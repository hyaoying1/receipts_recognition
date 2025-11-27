import os
import base64
import json
import time
from datetime import datetime
from pathlib import Path
import re
from openai import OpenAI

# ================== 基本配置 ==================
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV = "DASHSCOPE_API_KEY"

# 图片所在目录（默认 ./pictures，下层所有 jpg/png 直接当作行程单来测）
ROOT_DIR = "./pictures/行程单"

# 结果输出路径
OUTPUT_PATH = Path("outputs/qwen_vl_trip_results.json")

# 需要测试的模型列表（你也可以删掉/增减）
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

# ================== Prompt：这里只测行程单 ==================
# 👉 这里你可以直接换成刚刚那段长的 SYSTEM_PROMPT / TRIP_QUERY
TRIP_QUERY = """你是一名票据解析助手。请从给定的行程单/打车票图片中读取信息，只输出一个合法的 JSON。

输出的 JSON 结构和字段含义如下（键名和嵌套结构必须保持完全一致）：

{
  "type": "行程单",
  "vendor": 开票方/平台/供应商名称，字符串；无法确定时为 null,
  "apply_date": 申请日期，格式为 "YYYY-MM-DD"，无法确定时为 null,
  "start_date": 行程开始日期（整个行程单的起始日期），格式为 "YYYY-MM-DD"，无法确定时为 null,
  "end_date": 行程结束日期（整个行程单的结束日期），格式为 "YYYY-MM-DD"，无法确定时为 null,
  "trips": [
    {
      "city": 本条行程所在城市名称（中文或英文均可），无法确定时为 null,
      "date": 本条行程日期，格式为 "YYYY-MM-DD"，无法确定时为 null,
      "start_time": 本条行程的出发时间，格式为 "YYYY-MM-DD HH:MM:SS"，无法确定时为 null,
      "line_amount": 此段金额数字（例如 23.50，精确到小数点后两位），无法确定时为 null,
      "currency": 币种（如 "CNY"），无法确定时为 null
    }
  ],
  "total_amount": 行程单总金额数字（例如 256.80，精确到小数点后两位），无法确定时为 null
}


补充规则（很重要）：

1. 关于 "start_date" 和 "end_date"
   - 大部分行程单在表头会明确给出整个行程的起止日期（通常是一个日期区间），例如：
     - "行程日期：2024-12-30 至 2025-01-02"
   - 如果表头**同时给出了起始日期和结束日期**：
     - 直接使用表头中最早的日期作为 "start_date"；
     - 使用表头中最晚的日期作为 "end_date"。
   - 如果表头只给出**单一日期**或模糊信息（例如“行程日期：2024-03-15”）：
     - 将所有明细日期中最早的那一天填入 "start_date"；
     - 将所有明细日期中最晚的那一天填入 "end_date"。
   - 如果表头完全不提供起止日期信息（没有任何日期区间）：
     - 将所有明细日期中最早的那一天填入 "start_date"；
     - 将所有明细日期中最晚的那一天填入 "end_date"。
   - 如果既无法从表头也无法从明细中推断任何日期，则 "start_date" 和 "end_date" 都填 null。

2. 关于 "date" 和 "start_time" 的补全与推断
   - 目标格式：
     - "date": "YYYY-MM-DD"
     - "start_time": "YYYY-MM-DD HH:MM:SS"
   - 如果明细中只给出了“月-日 时:分”（如 "12-31 23:50"），没有年份、没有秒：
     - 优先从表头的行程起止时间获取年份信息；
     - 若表头显示了行程起止日期区间（例如 "2024-12-30 至 2025-01-02"）：
       - 对于每一条只包含“月-日”的记录，应选择一个年份，使得组合后的完整日期 (YYYY-MM-DD) 尽量落在行程起止日期区间内；
       - 如果区间跨年，根据日期区间自行判断最合理的年份归属，使所有行程日期整体尽量落在表头给出的起止日期范围内。
     - 若秒数缺失，则补为 ":00"。
     - 例如：明细写的是 "12-31 23:50"，表头区间是 "2024-12-30 至 2025-01-02"，则：
       - "date": "2024-12-31"
       - "start_time": "2024-12-31 23:50:00"
   - 如果既找不到年份信息，又无法判断年份（例如票据上完全没有任何年份），则：
     - "date" 和 "start_time" 都填 null。


输出要求（严格执行）：
1）只输出一个符合上述结构的 JSON 对象；
2）键名必须全部使用上述英文名称，不能增删或改名；
3）所有字符串值必须用英文双引号包裹；
4）不要输出任何额外的解释性文字、注释或多余内容，只输出 JSON。
"""


# ================== 工具函数 ==================
def iter_images(root_dir: str):
    """遍历目录下所有 jpg / jpeg / png 图片"""
    root = Path(root_dir)
    for img_path in sorted(root.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            yield img_path


def encode_image_to_data_url(path: Path) -> str:
    """把本地图片转成 data:image/...;base64,xxx 形式，便于直接塞到 image_url 里"""
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "application/octet-stream"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_json(text: str) -> dict:
    """
    尝试从模型输出中提取 JSON：
    - 截取第一个 { 到最后一个 } 之间
    - 做一些简单替换（中文标点、尾逗号、单引号）
    """
    stripped = text.strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    json_str = stripped[start : end + 1]

    # 替换常见中文标点，防止简单错误
    json_str = json_str.replace("：", ":").replace("，", ",")

    # 去掉 } 或 ] 前多余的逗号：...,}
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    # 将单引号包裹的 key/value 转成双引号（模型偶尔会用）
    json_str = re.sub(r"\'", '"', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 再做一次非常有限的修正：删除不可见字符后重试
        cleaned = "".join(ch for ch in json_str if ch.isprintable())
        return json.loads(cleaned)


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ================== 主逻辑：仅行程单 ==================
def main():
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"请先在环境变量中设置 {API_KEY_ENV}")

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    # 收集所有要测的图片（全部视为行程单）
    images = list(iter_images(ROOT_DIR))
    print(f"[INFO] Found {len(images)} images under {ROOT_DIR}")

    task_start_time_iso = now_iso()
    task_start_ts = time.time()

    all_results = {
        "task_start": task_start_time_iso,
        "root_dir": ROOT_DIR,
        "models": [],
    }

    for model_name in MODELS:
        print(f"[MODEL] Running: {model_name}")
        model_start_time_iso = now_iso()
        model_start_ts = time.time()

        model_results = {
            "model": model_name,
            "start_time": model_start_time_iso,
            "cases": [],
        }

        for img_path in images:
            # ========= 每张图片开始计时 =========
            img_start_ts = time.time()

            data_url = encode_image_to_data_url(img_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": TRIP_QUERY,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            },
                        },
                    ],
                }
            ]

            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=800,
            )

            content = completion.choices[0].message.content

            # 有的兼容模式会返回 list，有的直接是字符串，这里做个兼容
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                output_text = "".join(text_parts)
            else:
                output_text = content

            # 解析 JSON（可能失败，所以用 try/except，方便后面排查）
            try:
                parsed = extract_json(output_text)
                parse_error = None
            except Exception as e:
                parsed = None
                parse_error = str(e)

            # ========= 每张图片结束计时 =========
            img_elapsed = time.time() - img_start_ts

            # 在终端打印每张图片的耗时
            print(f"[{model_name}] {img_path.name} parsed in {img_elapsed:.2f}s")

            case_result = {
                "_file": str(img_path),
                "output_raw": output_text,   # 原始文本输出，方便 debug
                "output": parsed,            # 解析后的 JSON（如失败则为 None）
                "parse_error": parse_error,  # 解析错误信息（如成功则为 None）
                "elapsed_seconds": img_elapsed,  # ✅ 每张图片的耗时
            }
            model_results["cases"].append(case_result)

        model_end_time_iso = now_iso()
        model_elapsed = time.time() - model_start_ts
        model_results["end_time"] = model_end_time_iso
        model_results["elapsed_seconds"] = model_elapsed

        all_results["models"].append(model_results)

    task_end_time_iso = now_iso()
    task_elapsed = time.time() - task_start_ts
    all_results["task_end"] = task_end_time_iso
    all_results["task_elapsed_seconds"] = task_elapsed

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()