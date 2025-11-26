import os
import json
from typing import List, Dict, Any
from pathlib import Path
import re, glob

# ======== 路径配置：按你的实际目录改 ========
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  
RAW_JSON_PATH = PROJECT_ROOT/"data/raw/fake_all.json"          # faker 原始数据
IMAGE_DIR = PROJECT_ROOT/"artifacts/images"                   # 图片目录：id.jpg
OUTPUT_JSONL = PROJECT_ROOT/"data/cleaned/train_sft.jsonl"    # 输出给 Qwen-VL SFT 用的 jsonl


# ======== 工具函数：从 'YYYY-MM-DD HH:MM(:SS)' 拆出 date 和完整 start_time ========
def split_datetime(dt: str):
    """
    输入：'YYYY-MM-DD HH:MM:SS' 或 'YYYY-MM-DD HH:MM'
    输出：(date_str, full_dt_str) 或 (None, None)
      - date_str: 'YYYY-MM-DD'
      - full_dt_str: 'YYYY-MM-DD HH:MM:SS'
    """
    if not dt:
        return None, None
    dt = dt.strip()
    parts = dt.split()
    if len(parts) != 2:
        return None, None
    date_part, time_part = parts
    # 补齐秒
    time_segs = time_part.split(":")
    if len(time_segs) == 2:
        # HH:MM -> HH:MM:00
        time_part = time_part + ":00"
    elif len(time_segs) == 1:
        # 只有小时，理论上不会出现，兜个底：HH -> HH:00:00
        time_part = time_part + ":00:00"
    # 如果已经是 HH:MM:SS 就直接用
    full_dt = f"{date_part} {time_part}"
    return date_part, full_dt


# ======== 你的 system prompt（这里先不改，用你前面确认好的 PROMPT/SYSTEM_PROMPT 即可） ========
SYSTEM_PROMPT = (
"""你是一名票据解析助手。请从给定的行程单/打车票图片中读取信息，只输出一个合法的 JSON。

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
      "line_amount": 此段金额数字（例如 23.50），无法确定时为 null,
      "currency": 币种（如 "CNY"），无法确定时为 null
    }
  ],
  "total_amount": 行程单总金额数字（例如 256.80），无法确定时为 null
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
""")
USER_QUERY = "请从这张行程单/打车票中提取字段，并用上面定义的 JSON 结构返回。"


def build_clean_label(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 faker 一条记录 rec 构造出“标准提取 JSON”（模型要学的东西）。
    新版结构：
    {
        "type": "行程单",
        "vendor": ...,
        "apply_date": "YYYY-MM-DD" 或 null,
        "start_date": "YYYY-MM-DD" 或 null,   # 所有 trips 中最早的日期（或 faker 里的 first_trip_date）
        "end_date": "YYYY-MM-DD" 或 null,     # 所有 trips 中最晚的日期（或 faker 里的 trip_date）
        "trips": [
            {
                "city": ...,
                "date": "YYYY-MM-DD" 或 null,
                "start_time": "YYYY-MM-DD HH:MM:SS" 或 null,
                "line_amount": 金额数字或 null,
                "currency": 币种或 null
            },
            ...
        ],
        "total_amount": 总金额或 null
    }
    """
    trips_raw: List[Dict[str, Any]] = rec.get("trips", []) or []

    trips_out = []
    date_list: List[str] = []

    currency = rec.get("currency", "CNY")

    for trip in trips_raw:
        # faker 里原来存的是完整时间字符串，比如 "2025-01-02 13:45:00"
        raw_dt = trip.get("start_time_str") or trip.get("start_time") or ""
        date_part, full_dt = split_datetime(raw_dt)

        if date_part:
            date_list.append(date_part)

        line_amount = trip.get("invoice_amount")
        city = trip.get("city")

        trips_out.append(
            {
                "city": city,
                "date": date_part,        # "YYYY-MM-DD" 或 None
                "start_time": full_dt,    # "YYYY-MM-DD HH:MM:SS" 或 None
                "line_amount": line_amount,
                "currency": currency,
            }
        )

    # 顶层 start_date / end_date：
    # 1) 优先从 date_list 里算最小/最大
    # date_list 来自 split_datetime(trip["start_time_str"]) 的 YYYY-MM-DD
    start_date = min(date_list) if date_list else None
    end_date   = max(date_list) if date_list else None

    # 如果 trips 全缺，再 fallback faker 顶层
    if not start_date:
        start_date = rec.get("first_trip_date")
    if not end_date:
        end_date = rec.get("trip_date")

    # 额外一致性保护（保险丝），确保顶层字段的起始时间正确
    if start_date and end_date and start_date > end_date:
        start_date, end_date = end_date, start_date

    cleaned = {
        "type": "行程单",
        "vendor": rec.get("platform_name") or rec.get("platform"),
        "apply_date": rec.get("apply_date"),      # YYYY-MM-DD 或 None
        "start_date": start_date,                # YYYY-MM-DD 或 None
        "end_date": end_date,                    # YYYY-MM-DD 或 None
        "trips": trips_out,
        "total_amount": rec.get("total_amount"),
    }
    return cleaned


def find_pages_for_id(rec_id: str, image_dir: Path, project_root: Path) -> List[str]:
    pattern = str(image_dir / f"{rec_id}_page_*.jpg")
    page_paths = glob.glob(pattern)

    page_re = re.compile(r"_page_(\d+)\.jpg$", re.I)
    tmp = []
    for p in page_paths:
        m = page_re.search(p)
        if m:
            tmp.append((int(m.group(1)), Path(p)))

    if tmp:
        tmp.sort(key=lambda x: x[0])
        return [str(p.relative_to(project_root)) for _, p in tmp]

    single_path = image_dir / f"{rec_id}.jpg"
    if single_path.exists():
        return [str(single_path.relative_to(project_root))]

    return []


def main():
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    print(f"[INFO] Loaded {len(records)} records from {RAW_JSON_PATH}")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for rec in records:
            rec_id = rec["id"]

            # ✅ 新：找到该 rec_id 的所有页（多页/单页兼容）
            pages = find_pages_for_id(rec_id, IMAGE_DIR, PROJECT_ROOT)

            if not pages:
                print(f"[WARN] image not found for id={rec_id} in {IMAGE_DIR}")
                continue

            # 2) 构造 label
            label_obj = build_clean_label(rec)
            label_text = json.dumps(label_obj, ensure_ascii=False)

            # 3) 构造 messages（把多页塞进同一个 user content）
            user_content = []
            for page_path in pages:
                user_content.append({
                    "type": "image",
                    "image": page_path,   # Dataset 后续会转绝对路径
                })
            user_content.append({
                "type": "text",
                "text": "以下是同一张行程单/打车票的多页图片，请综合所有页面提取字段，并用上面定义的 JSON 结构返回。"
            })

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": label_text}],
                },
            ]

            sample = {
                "id": rec_id,
                "pages": pages,      # ✅ 多页字段
                "image": pages[0],   # （可选）保留一个旧字段，方便你调试/兼容旧评测
                "messages": messages,
            }

            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote SFT dataset to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()