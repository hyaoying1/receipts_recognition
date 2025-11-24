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

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROMPT_DIR = Path("prompts")
OUTPUT_DIR = Path("outputs")

PROMPT_MAP = {
    "itinerary": PROMPT_DIR / "itinerary_prompt.txt",
    "hotel_invoice": PROMPT_DIR / "hotel_prompt.txt",
    "payment": PROMPT_DIR / "payment_prompt.txt",
}



async def batch_ocr_and_classify(paths):
    # ---- Batch OCR ----
    ocr_tasks = [run_ocr_async(p) for p in paths]
    texts = await asyncio.gather(*ocr_tasks)

    # ---- Batch rule-based classification ----
    cls_tasks = [rule_classify(t) for t in texts]
    types = await asyncio.gather(*cls_tasks)

    return [
        {"file": str(p), "path": p, "text": t, "type": ty}
        for p, t, ty in zip(paths, texts, types)
    ]


async def extract_one(processed_path: Path, doc_type: str):
    if doc_type not in PROMPT_MAP:
        return None

    prompt_path = PROMPT_MAP[doc_type]
    result = await run_one_file(processed_path, prompt_path)

    # Try parsing JSON
    try:
        output = result.get("output")
        if isinstance(output, str):
            result["output"] = json.loads(output)
    except:
        pass

    return {
        "processed_file": processed_path.name,
        "type": doc_type,
        "result": result
    }



async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---- Load and preprocess files ----
    raw_files = sorted(RAW_DIR.iterdir())
    processed_paths = [preprocess_file(f, PROCESSED_DIR) for f in raw_files]

    # ---- Batch OCR + classification ----
    print("\n🔍 Running batch OCR + classification ...")
    batch_results = await batch_ocr_and_classify(processed_paths)

    # ---- Extraction tasks ----
    extract_tasks = []
    for item in batch_results:
        doc_type = item["type"]
        if doc_type not in PROMPT_MAP:
            print(f"❌ Unknown type: {doc_type}, skipping {item['file']}")
            continue
        extract_tasks.append(extract_one(item["path"], doc_type))

    # Run all extraction in parallel
    extracted = await asyncio.gather(*extract_tasks)
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
