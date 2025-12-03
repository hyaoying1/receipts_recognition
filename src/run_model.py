import os
import base64
import json
import asyncio
from pathlib import Path
import httpx
from typing import List


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY_ENV = "OPENAI_API_KEY"
MODEL_NAME = "qwen3-vl-8b-instruct"

SEM_EXTRACT = asyncio.Semaphore(5)


def make_data_url_sync(path: Path):
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


async def make_data_url(path: Path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, make_data_url_sync, path)


async def run_one_file(image_paths: List[Path], prompt_path: Path) -> dict:
    """
    多页文档抽取：
    - image_paths: 一份文档的所有页面图片路径（按页顺序排好）
    - prompt_path: 对应文档类型的 prompt 文件路径
    """
    if not image_paths:
        raise ValueError("image_paths is empty")

    api_key = os.getenv(API_KEY_ENV)
    prompt = Path(prompt_path).read_text(encoding="utf-8")

    # 1. 并发把所有页面转成 data URL
    data_url_tasks = [make_data_url(p) for p in image_paths]
    data_urls = await asyncio.gather(*data_url_tasks)

    # 2. 构造多图输入：一个文本 prompt + 多个 image_url
    content = [
        {"type": "text", "text": prompt},
    ]
    for url in data_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": 4000,
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with SEM_EXTRACT:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=40) as client:
                    resp = await client.post(BASE_URL, json=payload, headers=headers)
                    resp.raise_for_status()

                raw = resp.json()["choices"][0]["message"]["content"]

                # 和之前一样，兼容 list 形式的 content
                if isinstance(raw, list):
                    text = ""
                    for c in raw:
                        if isinstance(c, dict) and "text" in c:
                            text += c["text"]
                    raw = text

                raw = str(raw).strip()

                return {
                    # 代表这份文档，用第一页文件名做主名称
                    "file": str(image_paths[0]),
                    # 同时把所有页面文件名记录下来，方便调试/追踪
                    "files": [str(p) for p in image_paths],
                    "output": raw,
                }

            except Exception as e:
                if attempt == 2:
                    return {
                        "file": str(image_paths[0]),
                        "files": [str(p) for p in image_paths],
                        "error": str(e),
                    }
                await asyncio.sleep(1)