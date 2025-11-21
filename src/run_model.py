import os
import base64
import json
import asyncio
from pathlib import Path
from openai import AsyncOpenAI



BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV = "OPENAI_API_KEY"
MODEL_NAME = "qwen3-vl-8b-instruct"

INPUT_DIR = Path("data/processed")
PROMPT_FILE = "prompt.txt"
CONCURRENCY = 30                



def load_prompt() -> str:
    path = Path(PROMPT_FILE)
    if not path.exists():
        raise FileNotFoundError(f"未找到 {PROMPT_FILE}")
    return path.read_text(encoding="utf-8")


def make_data_url(path: Path) -> str:
    """把图片转成 Base64 Data URL（jpg/png）"""
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        raise ValueError(f"不支持的格式: {suffix}")

    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def get_image_files(root_dir: Path):
    """遍历图片目录"""
    exts = [".jpg", ".jpeg", ".png"]
    return [p for p in sorted(root_dir.iterdir()) if p.suffix.lower() in exts]


async def run_inference(client: AsyncOpenAI, image_path: Path, prompt: str, semaphore: asyncio.Semaphore):

    data_url = make_data_url(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    async with semaphore:  # 限制并发数量
        try:
            resp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=200,     # 建议降低，提高速度
            )
        except Exception as e:
            return {
                "file": str(image_path),
                "error": str(e)
            }

    # 提取文本
    content = resp.choices[0].message.content
    if isinstance(content, list):
        text = "".join([c.get("text", "") for c in content if isinstance(c, dict)])
    else:
        text = content

    return {
        "file": str(image_path),
        "output": text
    }



async def main_async():

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError("未检测到 OPENAI_API_KEY")

    prompt = load_prompt()
    image_files = get_image_files(INPUT_DIR)

    if not image_files:
        raise RuntimeError(f"在 {INPUT_DIR} 下未找到图片文件")

    print(f"发现 {len(image_files)} 张图片，开始 asyncio 并发推理...")

    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)

    semaphore = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        asyncio.create_task(run_inference(client, img, prompt, semaphore))
        for img in image_files
    ]

    results = await asyncio.gather(*tasks)

    # 保存 JSON
    OUTPUT_PATH = Path("outputs/qwen3vl8b_async_results.json")
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n🎉 异步推理完成！结果已保存到: {OUTPUT_PATH}")



if __name__ == "__main__":
    asyncio.run(main_async())
