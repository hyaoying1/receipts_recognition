import asyncio
from pathlib import Path
import json
from datetime import datetime
import time  

from src.pre_processor import preprocess_file
from src.run_model import run_one_file
from src.classifier import classify_file_async



RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROMPT_DIR = Path("prompts")
OUTPUT_DIR = Path("outputs")


PROMPT_MAP = {
    "itinerary": PROMPT_DIR / "itinerary_prompt.txt",
    "hotel_invoice": PROMPT_DIR / "hotel_prompt.txt",
    "payment": PROMPT_DIR / "payment_prompt.txt",
}


async def process_one_file(raw_file):
    print(f"\n📥 Processing raw file: {raw_file.name}")

    processed_path = preprocess_file(raw_file, PROCESSED_DIR)

    class_result = await classify_file_async(processed_path)
    doc_type = class_result.get("type")

    print(f"🧾 Classified as: {doc_type}")

    if doc_type not in PROMPT_MAP:
        print(f"❌ Unknown type: {doc_type}, skipping.")
        return None

    prompt_path = PROMPT_MAP[doc_type]

    # 🔥 This LLM call will run in parallel
    result = await run_one_file(processed_path, prompt_path)

    # Try parsing inner JSON
    try:
        if isinstance(result.get("output"), str):
            import json
            result["output"] = json.loads(result["output"])
    except:
        pass

    return {
        "raw_file": raw_file.name,
        "processed_file": processed_path.name,
        "type": doc_type,
        "result": result
    }


async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    tasks = []

    for raw_file in sorted(RAW_DIR.iterdir()):
        tasks.append(process_one_file(raw_file))

    all_results = await asyncio.gather(*tasks)

    # Remove None (unknown types)
    all_results = [x for x in all_results if x is not None]

    return all_results



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

    print(f"\n🎉 ALL DONE")
    print(f"📄 Output saved to: {out_file}")
