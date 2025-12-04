import asyncio
from pathlib import Path
import json
from datetime import datetime
import time

from src.pre_processor import preprocess_file
from src.run_model import run_one_file
from src.rulebased_classifier import (
    run_ocr_async,
    rule_classify,
)
from typing import List

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROMPT_DIR = Path("prompts")
OUTPUT_DIR = Path("outputs")

PROMPT_MAP = {
    "itinerary": PROMPT_DIR / "itinerary_prompt.txt",
    "hotel_invoice": PROMPT_DIR / "hotel_prompt.txt",
    "payment": PROMPT_DIR / "payment_prompt.txt",
    "other": PROMPT_DIR / "other_prompt.txt",
}



async def batch_ocr_and_classify(docs):
    """
    docs: List[List[Path]]
      - 外层：每个元素是一份文档
      - 内层：该文档的所有页面 jpg 路径

    返回：每个文档一条结果：
      {
        "doc_id": ...,
        "pages": [...],
        "type": ...,
      }
    """

    async def classify_one_doc(pages):
        # 保护一下，避免空列表
        if not pages:
            return {
                "doc_id": None,
                "pages": [],
                "type": "other",
            }

        first_page = pages[0]

        # 1. 用第一页做 OCR
        text = await run_ocr_async(first_page)

        # 2. 用 OCR 文本做规则分类
        doc_type = await rule_classify(text)

        # 3. 从第一页的文件名里推一个 doc_id
        stem = first_page.stem  # 比如 "trip_page1" 或 "hotel"
        if "_page" in stem:
            doc_id = stem.rsplit("_page", 1)[0]  # "trip_page1" -> "trip"
        else:
            doc_id = stem

        return {
            "doc_id": doc_id,
            "pages": pages,
            "type": doc_type,
        }

    # 所有文档并发分类
    tasks = [classify_one_doc(pages) for pages in docs]
    results = await asyncio.gather(*tasks)
    return list(results)


async def extract_one(pages: List[Path], doc_type: str):
    """
    对“同一份文档”的所有页面进行抽取：
    - pages: 这一份文档的所有页面 jpg 路径（至少有 1 个）
    - doc_type: 文档类型（行程单 / 酒店水单 / 支付记录 等）

    当前实现：把整份文档的所有 pages 列表直接传给 run_one_file，
    后续你会在 run_one_file 内部实现“多页合并 + 调用 LLM”的逻辑。
    """
    if doc_type not in PROMPT_MAP:
        return None

    if not pages:
        return None

    prompt_path = PROMPT_MAP[doc_type]

    # 代表这一份文档的“主文件名”，用第一页的名字即可
    first_page = pages[0]

    # 关键点：这里把「整份文档的所有页面」传给 run_one_file
    # 你后面会把 run_one_file 改成可以接收 List[Path] 并构造多页输入
    result = await run_one_file(pages, prompt_path)

    # Try parsing JSON
    try:
        output = result.get("output")
        if isinstance(output, str):
            result["output"] = json.loads(output)
    except Exception:
        pass

    return {
        # 兼容旧字段，用第一页名字做“代表文件名”
        "processed_file": first_page.name,
        # 额外返回这一份文档的所有页面名，方便调试 / 追溯
        "all_pages": [p.name for p in pages],
        "type": doc_type,
        "result": result,
    }



async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---- Load and preprocess files ----
    raw_files = sorted(RAW_DIR.iterdir())

    # 这里每个 preprocess_file 返回 List[Path]（一份文档的所有页面）
    processed_docs = [preprocess_file(f, PROCESSED_DIR) for f in raw_files]
    # processed_docs: List[List[Path]]

    # ---- Batch OCR + classification (doc-level) ----
    t0 = time.time()
    print("\n🔍 Running batch OCR + classification ...")
    batch_results = await batch_ocr_and_classify(processed_docs)
    t1 = time.time()
    print(f"🕒 Classification took: {t1 - t0:.2f} seconds")
    # batch_results 里每个 item:
    # {
    #   "doc_id": ...,
    #   "pages": [...],
    #   "type": ...,
    # }

    # ---- Extraction tasks ----
    extract_tasks = []
    for item in batch_results:
        doc_type = item["type"]
        if doc_type not in PROMPT_MAP:
            print(f"❌ Unknown type: {doc_type}, skipping doc {item['doc_id']}")
            continue

        pages = item["pages"]  # List[Path]，这一份文档的所有页面
        # 建议把 extract_one 改成按“文档级”来抽取：
        # async def extract_one(pages: List[Path], doc_type: str): ...
        extract_tasks.append(extract_one(pages, doc_type))
    
    t2 = time.time()
    # Run all extraction in parallel
    extracted = await asyncio.gather(*extract_tasks)
    t3 = time.time()
    print(f"🕒 Extraction (LLM) took: {t3 - t2:.2f} seconds")
    return [e for e in extracted if e is not None]



if __name__ == "__main__":
    start_time = time.time()

    results = asyncio.run(main())

    total_time = time.time() - start_time
    print(f"\n⏱ Total time taken: {total_time:.2f} seconds")

    now = datetime.now().strftime("%m%d%H%M")
    out_file = OUTPUT_DIR / f"output_{now}.json"

    output_json = {
        "total_time_seconds": round(total_time, 2),
        "results": results
    }

    out_file.write_text(
        json.dumps(output_json, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n🎉 ALL DONE")
    print(f"📄 Output saved to: {out_file}")
