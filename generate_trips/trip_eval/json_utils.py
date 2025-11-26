# trip_eval/json_utils.py
import json
import re
from typing import Any, Dict

def extract_json(text: str) -> Dict[str, Any]:
    """
    从模型输出中尽量提取 JSON：
    - 截取第一个 { 到最后一个 } 之间
    - 做一些简单替换（中文标点、尾逗号、单引号）
    """
    stripped = text.strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    json_str = stripped[start : end + 1]

    # 替换常见中文标点
    json_str = json_str.replace("：", ":").replace("，", ",")

    # 去掉 } 或 ] 前多余的逗号：...,}
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    # 单引号换成双引号（不完美，但够用）
    json_str = json_str.replace("'", '"')

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        cleaned = "".join(ch for ch in json_str if ch.isprintable())
        return json.loads(cleaned)