from pathlib import Path
import os
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor
from rapidocr_onnxruntime import RapidOCR


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY_ENV = "OPENAI_API_KEY"
MODEL_NAME = "qwen2.5-7b-instruct"

# Limit concurrency to 3 for classification (DashScope safe limit)
SEM_CLASSIFY = asyncio.Semaphore(3)

# OCR executor
ocr = RapidOCR()
ocr_executor = ThreadPoolExecutor(max_workers=8)


def run_ocr_sync(path: Path) -> str:
    result, _ = ocr(str(path))
    if result:
        return "\n".join([line[1] for line in result])
    return ""


async def run_ocr_async(path: Path) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(ocr_executor, run_ocr_sync, path)


async def classify_llm_async(text: str) -> str:
    if len(text.strip()) < 5:
        return "other"

    api_key = os.getenv(API_KEY_ENV)

    prompt = f"""
你是票据分类助手，请根据 OCR 文本判断票据类型，只输出：

- 行程单
- 酒店水单
- 支付记录
- 其他

OCR 文本：
{text}
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with SEM_CLASSIFY:  # limit concurrency
        # retry mechanism
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=25.0) as client:
                    resp = await client.post(BASE_URL, json=payload, headers=headers)
                    resp.raise_for_status()
                    answer = resp.json()["choices"][0]["message"]["content"].strip()
                    break
            except Exception as e:
                if attempt == 2:
                    return "other"
                await asyncio.sleep(1)

    if "行程" in answer:
        return "itinerary"
    if "酒店" in answer or "水单" in answer:
        return "hotel_invoice"
    if "支付" in answer:
        return "payment"

    return "other"


async def classify_file_async(image_path: Path) -> dict:
    print(f"\n📄 分类文件: {image_path.name}")

    text = await run_ocr_async(image_path)
    doc_type = await classify_llm_async(text)

    print(f"  分类结果: {doc_type}")

    return {"file": str(image_path), "type": doc_type}
