import os
import base64
import json
import time
from datetime import datetime
from pathlib import Path
import re
from openai import OpenAI


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV = "OPENAI_API_KEY"

ROOT_DIR = "./data"
OUTPUT_PATH = Path("outputs/qwen_vl_benchmark_results.json")


MODELS = [
    "qwen-vl-plus",
    # "qwen-vl-max",
    # "qwen2.5-vl-3b-instruct",
    # "qwen2.5-vl-7b-instruct",
    # "qwen2.5-vl-32b-instruct",
    # "qwen2.5-vl-72b-instruct",
    # "qwen3-vl-8b-instruct",
    # "qwen3-vl-30b-a3b-instruct",
]

# MODELS = [
#     # 通义千问（阿里）官方 VL 模型
#     "qwen-vl-plus",
#     "qwen-vl-max",
#     "qwen2.5-vl-3b-instruct",
#     "qwen2.5-vl-7b-instruct",
#     "qwen2.5-vl-32b-instruct",
#     "qwen2.5-vl-72b-instruct",
#     "qwen3-vl-8b-instruct",
#     "qwen3-vl-30b-a3b-instruct",

#     # 智谱（Zhipu AI）视觉模型
#     "glm-4v",
#     "glm-4v-flash",

#     # InternVL 系列（强力视觉模型）
#     "internvl2-1b",
#     "internvl2-4b",
#     "internvl2-8b",
#     "internvl2-26b",

#     # CogVLM 系列
#     "cogvlm2-llama3",
#     "cogvlm2-large",

#     # LLaVA-Qwen 系列
#     "llava-qwen-7b",
#     "llava-qwen-32b",

#     # 可选：Qwen 思考 + 视觉模型（需你自己确认可调用方式）
#     "qwen3-vl-72b-thinking",
#     "qwen3-vl-235b-a22b-thinking-fp8",
# ]



TRIP_QUERY = (
    "你是一名票据解析助手。请从给定的行程单/打车票图片中读取信息，只输出一个合法的 JSON："
    "字段含义如下（值可以是中文）："
    "{\n"
    "  \"type\": \"itinerary\",                  // 固定为行程单类型\n"
    "  \"attachment_name\": 原始附件或文件名，无则为 null,\n"
    "  \"vendor\": 开票方/平台/供应商名称，无则为 null,\n"
    "  \"transaction_date\": 交易日期(YYYY-MM-DD)或 null,\n"
    "  \"start_date\": 行程开始日期(YYYY-MM-DD)或 null,\n"
    "  \"end_date\": 行程结束日期(YYYY-MM-DD)或 null,\n"
    "  \"city\": 城市名称（中文或英文均可）或 null,\n"
    "  \"invoice_line_id\": 票号/订单号/发票号等标识或 null,\n"
    "  \"trips\": [\n"
    "    {\n"
    "      \"start_location\": 出发地名称（可为中文）或 null,\n"
    "      \"end_location\": 目的地名称（可为中文）或 null,\n"
    "      \"start_time\": 出发时间(HH:MM:SS)或 null,\n"
    "      \"line_amount\": 此段金额数字或 null,\n"
    "      \"currency\": 币种（如\"CNY\"）或 null\n"
    "    }\n"
    "  ],\n"
    "  \"total_amount\": 总金额数字或 null\n"
    "}\n"
    "要求：\n"
    "1）只输出上述结构的 JSON；\n"
    "2）键名保持英文不变；\n"
    "3）字符串值可以是中文，但必须用双引号；\n"
    "4）不要输出任何额外说明文字。"
)

HOTEL_QUERY = (
    "你是一名票据解析助手。请从给定的酒店水单/刷卡单图片中读取信息，只输出一个合法的 JSON："
    "字段含义如下（值可以是中文）："
    "{\n"
    "  \"type\": \"hotel_folio\",                // 固定为酒店账单类型\n"
    "  \"attachment_name\": 原始附件或文件名，无则为 null,\n"
    "  \"city\": 城市名称（可为中文）或 null,\n"
    "  \"checkin_date\": 入住日期(YYYY-MM-DD)或 null,\n"
    "  \"checkout_date\": 离店日期(YYYY-MM-DD)或 null,\n"
    "  \"currency\": 币种（如\"CNY\"）或 null,\n"
    "  \"amount\": 总金额数字或 null,\n"
    "  \"invoice_line_id\": 发票号/账单号等标识或 null\n"
    "}\n"
    "要求：\n"
    "1）只输出上述结构的 JSON；\n"
    "2）键名保持英文不变；\n"
    "3）字符串值可以是中文，但必须用双引号；\n"
    "4）不要输出任何额外说明文字。"
)


def iter_images(root_dir: str):
    root = Path(root_dir)
    for img_path in sorted(root.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            yield img_path


def classify_doc(path: Path) -> str:
    name = path.name
    if "行程" in name or "车票" in name:
        return "trip"
    if "酒店" in name:
        return "hotel"
    return ""


def encode_image_to_data_url(path: Path) -> str:
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
    stripped = text.strip()

    # 截取第一个 { 到 最后一个 } 之间的内容
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
    # 注意只做简单替换，复杂情况这里不管
    json_str = re.sub(r"\'", '"', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 再做一次非常有限的修正：删除控制字符和不可见字符后重试
        cleaned = "".join(ch for ch in json_str if ch.isprintable())
        return json.loads(cleaned)


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def main():
    api_key = os.getenv(API_KEY_ENV)

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    images = []
    for img_path in iter_images(ROOT_DIR):
        doc_type = classify_doc(img_path)
        if doc_type:
            images.append((doc_type, img_path))

    task_start_time_iso = now_iso()
    task_start_ts = time.time()

    all_results = {
        "task_start": task_start_time_iso,
        "models": [],
    }

    for model_name in MODELS:
        model_start_time_iso = now_iso()
        model_start_ts = time.time()

        model_results = {
            "model": model_name,
            "start_time": model_start_time_iso,
            "cases": [],
        }

        for doc_type, img_path in images:
            if doc_type == "trip":
                query = TRIP_QUERY
            elif doc_type == "hotel":
                query = HOTEL_QUERY
            else:
                continue

            data_url = encode_image_to_data_url(img_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query,
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
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                output_text = "".join(text_parts)
            else:
                output_text = content

            parsed = extract_json(output_text)

            case_result = {
                "_file": str(img_path),
                "_doc_type": doc_type,
                "output": parsed,
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


if __name__ == "__main__":
    main()