import os
import base64
import json
import time
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Any, List, Tuple

from openai import OpenAI

# ================== 基本配置 ==================
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV = "DASHSCOPE_API_KEY"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  
# ✅ 这里改成你真实的 SFT 数据路径
#   就是你刚才展示那种每行一个 {"id": ... "messages": ...} 的 jsonl
DATA_PATH = PROJECT_ROOT / "data/cleaned/train_sft.jsonl"

# 图片路径是相对路径，比如 "artifacts/images/baidu_000001.jpg"
# 这里默认它们都是相对项目根目录
IMAGE_ROOT = PROJECT_ROOT

# 结果输出路径（评测明细 + 指标）
OUTPUT_PATH = PROJECT_ROOT / "outputs/qwen_3b_outputs.json"

# 需要测试的模型列表（你可以只留 1 个，比如 qwen3-vl-8b-instruct）
MODELS = [
    # "qwen-vl-plus",
    # "qwen-vl-max",
    "qwen2.5-vl-3b-instruct",
    # "qwen2.5-vl-7b-instruct",
    # "qwen2.5-vl-32b-instruct",
    # "qwen2.5-vl-72b-instruct",
    # "qwen3-vl-8b-instruct",
    # "qwen3-vl-30b-a3b-instruct",
]

# ================== Prompt（可选） ==================
# 实际上你在 jsonl 里已经把 system prompt 保存了，这里主要是保证和那一份一致
TRIP_SYSTEM_PROMPT = """你是一名票据解析助手。请从给定的行程单/打车票图片中读取信息，只输出一个合法的 JSON。

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

USER_QUERY = "请从这张行程单/打车票中提取字段，并用上面定义的 JSON 结构返回。"


# ================== 工具函数 ==================
def encode_image_to_data_url(path: Path) -> str:
    """把本地图片转成 data:image/...;base64,xxx 形式，给 image_url 用"""
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


# ========== 指标计算：字段级 & 样本级 ==========
TOP_FIELDS = ["type", "vendor", "apply_date", "start_date", "end_date", "total_amount"]
TRIP_FIELDS = ["city", "date", "start_time", "line_amount", "currency"]


def _equal_value(pred: Any, gold: Any, num_tol: float = 1e-2) -> bool:
    # 都是 None / null
    if pred is None and gold is None:
        return True

    # 数字：用 float 容忍一点点误差
    if isinstance(gold, (int, float)) or isinstance(pred, (int, float)):
        try:
            p = float(pred)
            g = float(gold)
            return abs(p - g) <= num_tol
        except Exception:
            return str(pred) == str(gold)

    # 其他直接字符串比较
    return str(pred) == str(gold)


def compare_record(pred: Dict[str, Any], gold: Dict[str, Any]) -> Tuple[bool, int, int]:
    """
    返回: (整条是否完全一致, 匹配字段数, 总字段数)
    简化假设：trips 不重排，按顺序一一对应。
    """
    match = True
    field_matches = 0
    field_total = 0

    # 顶层
    for key in TOP_FIELDS:
        field_total += 1
        if _equal_value(pred.get(key), gold.get(key)):
            field_matches += 1
        else:
            match = False

    # trips
    gold_trips = gold.get("trips", []) or []
    pred_trips = pred.get("trips", []) or []

    n = min(len(gold_trips), len(pred_trips))
    for i in range(n):
        g_trip = gold_trips[i]
        p_trip = pred_trips[i]
        for key in TRIP_FIELDS:
            field_total += 1
            if _equal_value(p_trip.get(key), g_trip.get(key)):
                field_matches += 1
            else:
                match = False

    return match, field_matches, field_total


# ========== 主评测逻辑：遍历 jsonl，逐条调用模型 ==========
def main(max_samples: int = None):
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"请先在环境变量中设置 {API_KEY_ENV}")

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    # 先把数据都读进来
    samples: List[Dict[str, Any]] = []
    start_n = 1
    end_n = 10
    id_re = re.compile(r"^[^_]+_(\d+)$")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = str(obj.get("id", ""))
            m = id_re.search(_id)
            if not m:
                continue 
            n = int(m.group(1))
            if start_n <= n <= end_n:
                samples.append(obj)

    print(f"[INFO] Loaded {len(samples)} labeled samples from {DATA_PATH} "
      f"(filtered *_ {start_n:06d} ~ *_ {end_n:06d})")

    task_start_time_iso = now_iso()
    task_start_ts = time.time()

    all_results = {
        "task_start": task_start_time_iso,
        "data_path": str(DATA_PATH),
        "models": [],
    }

    for model_name in MODELS:
        print(f"[MODEL] Evaluating: {model_name}")
        model_start_time_iso = now_iso()
        model_start_ts = time.time()

        model_results = {
            "model": model_name,
            "start_time": model_start_time_iso,
            "cases": [],
            "record_equal_count": 0,
            "sample_count": 0,
            "field_match_total": 0,
            "field_total": 0,
        }

        # 累计指标
        record_equal_count = 0
        sample_count = 0
        field_match_total = 0
        field_total = 0

        for idx, sample in enumerate(samples, start=1):
            if max_samples is not None and sample_count >= max_samples:
                break

            sample_id = sample.get("id")
            pages = sample.get('pages')
            if not pages:
                image_rel = sample.get('image')
                # 例如 "artifacts/images/baidu_000001.jpg"
                pages = [image_rel]
                    
            # gold label 是 messages 里最后一条 assistant 里的 text
            gold_label_text = sample["messages"][-1]["content"][0]["text"]
            try:
                gold = json.loads(gold_label_text)
            except Exception as e:
                print(f"[WARN] parse gold json failed for id={sample_id}: {e}")
                continue
            
            data_urls = []
            missing = False
            for rel in pages:
                img_path = IMAGE_ROOT/rel
                try:
                    data_urls.append(encode_image_to_data_url(img_path))
                except FileNotFoundError:
                    print(f"[WARN] image not found, skip sample id={sample_id}: {img_path}")
            # 可选：记录错误
            # errors.append({"id": sample_id, "error": "image_not_found", "path": str(img_path)})
                    missing = True
                    break
            if missing:
                continue

            # 构造发给 dashscope 的 messages，有错误的情况跳过
            user_content = [{"type": "text", "text": USER_QUERY}]
            for url in data_urls:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": TRIP_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]

            img_start_ts = time.time()
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=3000,
                )
                content = completion.choices[0].message.content
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    output_text = "".join(text_parts)
                else:
                    output_text = content

                # 解析预测 JSON
                try:
                    pred = extract_json(output_text)
                    parse_error = None
                except Exception as e:
                    pred = None
                    parse_error = str(e)

            except Exception as e:
                output_text = ""
                pred = None
                parse_error = f"model_call_error: {e}"

            img_elapsed = time.time() - img_start_ts

            sample_count += 1

            # 对比 gold & pred
            if pred is not None:
                record_equal, fm, ft = compare_record(pred, gold)
            else:
                record_equal, fm, ft = False, 0, len(TOP_FIELDS)  # 极简兜底

            if record_equal:
                record_equal_count += 1
            field_match_total += fm
            field_total += ft

            print(
                f"[{model_name}] id={sample_id} "
                f"record_equal={record_equal} field_match={fm}/{ft} "
                f"time={img_elapsed:.2f}s"
            )

            case_result = {
                "id": sample_id,
                "image": str(img_path),
                "gold": gold,
                "output_raw": output_text,
                "output": pred,
                "parse_error": parse_error,
                "record_equal": record_equal,
                "field_match": fm,
                "field_total": ft,
                "elapsed_seconds": img_elapsed,
            }
            model_results["cases"].append(case_result)

        model_end_time_iso = now_iso()
        model_elapsed = time.time() - model_start_ts

        model_results["end_time"] = model_end_time_iso
        model_results["elapsed_seconds"] = model_elapsed
        model_results["record_equal_count"] = record_equal_count
        model_results["sample_count"] = sample_count
        model_results["field_match_total"] = field_match_total
        model_results["field_total"] = field_total

        all_results["models"].append(model_results)

        # 在终端打印一份汇总
        print("\n===== MODEL SUMMARY:", model_name, "=====")
        if sample_count > 0:
            print(
                f"Record-level accuracy: "
                f"{record_equal_count}/{sample_count} = {record_equal_count / sample_count:.3f}"
            )
        if field_total > 0:
            print(
                f"Field-level accuracy: "
                f"{field_match_total}/{field_total} = {field_match_total / field_total:.3f}"
            )
        print("====================================\n")

    task_end_time_iso = now_iso()
    task_elapsed = time.time() - task_start_ts
    all_results["task_end"] = task_end_time_iso
    all_results["task_elapsed_seconds"] = task_elapsed

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote eval results to {OUTPUT_PATH}")


if __name__ == "__main__":
    # max_samples 可以先设个小数，比如 50，确认逻辑没问题再放开
    main(max_samples=None)