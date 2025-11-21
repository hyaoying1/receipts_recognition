import os
import base64
import asyncio
from pathlib import Path
import httpx


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY_ENV = "OPENAI_API_KEY"
MODEL_NAME = "qwen3-vl-8b-instruct"

# limit concurrency to 3 for image extraction
SEM_EXTRACT = asyncio.Semaphore(3)


def make_data_url_sync(path: Path):
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


async def make_data_url(path: Path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, make_data_url_sync, path)


async def run_one_file(image_path: Path, prompt_path: Path) -> dict:
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    api_key = os.getenv(API_KEY_ENV)

    data_url = await make_data_url(image_path)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 200,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with SEM_EXTRACT:  # limit concurrency
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(BASE_URL, json=payload, headers=headers)
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    return {"file": str(image_path), "output": content}
            except Exception as e:
                if attempt == 2:
                    return {"file": str(image_path), "error": str(e)}
                await asyncio.sleep(1)
