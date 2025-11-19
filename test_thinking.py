import os
import base64
import json
import time
from datetime import datetime
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF, 用于 PDF 转图片
import matplotlib.pyplot as plt
from openai import OpenAI


# =========================
# 基本配置
# =========================
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV = "OPENAI_API_KEY"

DATA_DIR = "./data"
OUTPUT_DIR = Path("outputs")
OUTPUT_JSON = OUTPUT_DIR / "qwen_vl_benchmark_results.json"
OUTPUT_PLOT = OUTPUT_DIR / "avg_time_plot.png"
PROMPT_FILE = "prompt.txt"

# 并行线程数
MAX_WORKERS = 8

# 模型列表
MODELS = [
    "qwen-vl-plus",
    "qwen-vl-max",
    "qwen2.5-vl-3b-instruct",
    "qwen2.5-vl-7b-instruct",
    "qwen2.5-vl-32b-instruct",
    "qwen2.5-vl-72b-instruct",
    "qwen3-vl-8b-instruct",
    "qwen3-vl-30b-a3b-instruct",
]



# =========================
# 工具函数
# =========================
def now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串"""
    return datetime.utcnow().isoformat() + "Z"


def load_prompt() -> str:
    """读取 prompt.txt"""
    path = Path(PROMPT_FILE)
    if not path.exists():
        raise FileNotFoundError(f"未找到 {PROMPT_FILE}，请在脚本同级目录下创建该文件。")
    return path.read_text(encoding="utf-8")


def iter_docs(root_dir: str):
    """
    遍历 ./data 下的文件，只保留 jpg/jpeg/png/pdf
    返回 Path 对象
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root_dir}")

    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".pdf"):
            yield p


def make_data_url_from_bytes(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def encode_image_file_to_data_url(path: Path) -> str:
    """
    针对 jpg/jpeg/png 直接读取文件并转为 data URL
    """
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "application/octet-stream"

    with path.open("rb") as f:
        data = f.read()
    return make_data_url_from_bytes(data, mime)


def load_pdf_first_page_to_data_url(path: Path) -> tuple[int, str]:
    """
    打开 PDF，取第一页，渲染为 PNG，再转为 data URL。
    返回 (page_number, data_url)，page_number 从 1 开始。
    假设 PDF 都是单页，但代码支持有多页的情况（这里只取第一页）。
    """
    doc = fitz.open(path)
    if len(doc) == 0:
        raise ValueError(f"PDF 文件没有页面: {path}")
    page = doc[0]
    pix = page.get_pixmap()  # 默认分辨率
    png_bytes = pix.tobytes("png")
    data_url = make_data_url_from_bytes(png_bytes, "image/png")
    return 1, data_url  # 页码从 1 开始


def extract_json(text: str) -> dict:
    """
    强健 JSON 解析器：
    - 自动提取 ```json ... ``` 块
    - 识别最外层 { ... }
    - 自动修复中文符号、单引号、未加引号的 key、末尾逗号
    - 尝试使用正则给 key 补双引号
    """

    if not text:
        raise ValueError("Model output is empty")

    # 1. 找 ```json ... ``` 部分
    m = re.search(r"```json(.*?)```", text, re.S)
    if m:
        text = m.group(1)

    # 2. 只取最外层 {...}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")

    json_str = text[start:end+1]

    # ---------- 自动修复常见模型输出问题 ----------

    # （A）替换中文冒号、逗号
    json_str = json_str.replace("：", ":").replace("，", ",")

    # （B）统一替换单引号 -> 双引号  
    # 防止把值里的撇号也替换，但一般模型不会生成这种复杂情况
    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)

    # （C）给未加引号的 key 自动补引号：  
    # 例如：amount: 100  ->  "amount": 100
    json_str = re.sub(
        r'(?m)^\s*([A-Za-z0-9_\u4e00-\u9fa5]+)\s*:',
        r'"\1":',
        json_str
    )

    # (D) 修复行内未加引号 key，例如 { amount: 100 }
    json_str = re.sub(
        r'({|,)\s*([A-Za-z0-9_\u4e00-\u9fa5]+)\s*:',
        r'\1 "\2":',
        json_str
    )

    # （E）去掉多余的逗号，如： { "a":1, }
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    # （F）删除注释或尾部垃圾
    json_str = re.sub(r"//.*?$", "", json_str, flags=re.M)
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.S)

    # ---------- 尝试解析 ----------
    try:
        return json.loads(json_str)
    except Exception as e:
        print("========== RAW JSON STR ==========")
        print(json_str)
        print("==================================")
        raise ValueError(f"JSON parsing failed: {e}")



# =========================
# 核心处理逻辑
# =========================
def build_doc_items() -> list[dict]:
    """
    将 data 目录下的文件转成可供调用模型的任务列表：
    每个元素包含：
    {
        "file": Path,
        "page": int,       # 对图片就是 1，对 PDF 是页码（这里只取 1）
        "data_url": str,   # base64 data URL
    }
    """
    items: list[dict] = []
    for path in iter_docs(DATA_DIR):
        suffix = path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png"):
            data_url = encode_image_file_to_data_url(path)
            items.append({"file": path, "page": 1, "data_url": data_url})
        elif suffix == ".pdf":
            # 默认单页 PDF，这里只取第一页
            page_num, data_url = load_pdf_first_page_to_data_url(path)
            items.append({"file": path, "page": page_num, "data_url": data_url})
        else:
            # 正常不会走到这里，因为前面已经过滤
            continue

    if not items:
        raise RuntimeError(f"在 {DATA_DIR} 下没有找到 jpg/jpeg/png/pdf 文件。")
    return items


def call_model_on_doc(api_key, model_name, prompt, file, page, data_url):
    # print("THREAD ENV:", os.getenv("OPENAI_API_KEY"))

    # 每个线程独立 client 
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    start_ts = time.time()
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=800,
    )
    elapsed = time.time() - start_ts

    # content 提取和 JSON 
    content = completion.choices[0].message.content

    # DashScope 有时 content 可能是 list，这里做兼容
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        output_text = "".join(text_parts)
    else:
        output_text = content or ""
        # print("=== RAW OUTPUT ===")
        # print(output_text)
        # print("==================")


    parsed = extract_json(output_text)

    return model_name, file, page, parsed, elapsed


def plot_per_file_timing_multi_round(results_by_model, out_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import matplotlib

    # Set Chinese font
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # --- 收集所有文件名（保持顺序） ---
    all_files = []
    for model in results_by_model.values():
        for case in model["cases"]:
            fname = os.path.basename(case["file"])
            if fname not in all_files:
                all_files.append(fname)

    # --- 绘图 ---
    plt.figure(figsize=(16, 6))

    for model_name, model_data in results_by_model.items():
        # 记录模型对每个文件的多轮耗时
        file_time_map = {}  # fname → list[time_taken]

        for case in model_data["cases"]:
            fname = os.path.basename(case["file"])
            t = case.get("time_taken", 0)
            file_time_map.setdefault(fname, []).append(t)

        # 计算平均值（10 轮）
        avg_times = []
        for f in all_files:
            if f in file_time_map:
                avg_times.append(np.mean(file_time_map[f]))
            else:
                avg_times.append(0)

        plt.plot(all_files, avg_times, marker="o", label=model_name)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Time Taken (s)")
    plt.title("Per-File Timing Comparison (Averaged Over Multiple Rounds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # 读取 API Key
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError("未检测到环境变量 OPENAI_API_KEY")

    # 读取统一 prompt
    prompt = load_prompt()

    # 构建文档任务列表（支持 jpg/jpeg/png/pdf）
    doc_items = build_doc_items()

    task_start_iso = now_iso()

    # 结果结构：不包含任何时间字段（按你的要求）
    results_by_model: dict[str, dict] = {
        model: {"model": model, "cases": []} for model in MODELS
    }

    # 用于统计平均耗时（不写入 JSON，只用于绘图）
    time_records: dict[str, list[float]] = {model: [] for model in MODELS}

    # 并行执行：对每个模型 × 每个 doc_item 构建任务
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for model in MODELS:
            for item in doc_items:
                future = executor.submit(
                    call_model_on_doc,
                    api_key,
                    model,
                    prompt,
                    item["file"],
                    item["page"],
                    item["data_url"],
                )
                future.model_name = model   # ⭐绑定模型名
                futures.append(future)

        for fut in as_completed(futures):
            try:
                model_name, file, page, parsed, elapsed = fut.result()
            except Exception as e:
                print(f"[ERROR] 模型 {fut.model_name} 调用异常: {e}")  # ⭐打印模型名
                continue


            if "cases" not in results_by_model[model_name]:
                results_by_model[model_name]["cases"] = []

            results_by_model[model_name]["cases"].append(
                {
                    "file": str(file),
                    "page": page,
                    "output": parsed,
                    "time_taken": elapsed,   # ⭐ 保存每文件耗时
                }
            )

    task_end_iso = now_iso()

    # 组织最终 JSON（不包含任何时间字段）
    all_results = {
        "task_start": task_start_iso,
        "task_end": task_end_iso,
        "models": list(results_by_model.values()),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 计算每个模型的平均耗时
    avg_time = {}
    for model, times in time_records.items():
        if times:
            avg_time[model] = sum(times) / len(times)
        else:
            avg_time[model] = 0.0

    # 画图
    PLOT_MULTI = OUTPUT_DIR / "per_file_timing_multi_round.png"
    plot_per_file_timing_multi_round(results_by_model, PLOT_MULTI)
    print(f"📊 多轮平均耗时图: {PLOT_MULTI}")



if __name__ == "__main__":
    main()
