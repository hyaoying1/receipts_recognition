# ============================================================
# 票据分类器 - 极速精简版
# RapidOCR（最快） + Qwen LLM 分类
# 统计每张图片 OCR 用时 + LLM 用时
# ============================================================

from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from openai import OpenAI
import time

# ==== 配置 ====
API_KEY = "sk-88551cce573d49fe81aa466d78c21741"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen2.5-7b-instruct"

INPUT_DIR = Path("data/processed")

# 初始化 OCR（最快）
ocr = RapidOCR()

# 初始化 LLM 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============================================================
# OCR
# ============================================================
def run_ocr(path):
    start = time.time()
    result, _ = ocr(path)
    elapsed = time.time() - start

    if result:
        text = "\n".join([line[1] for line in result])
    else:
        text = ""

    return text, elapsed


# ============================================================
# LLM 分类
# ============================================================
def classify_llm(text: str):
    if len(text.strip()) < 5:
        return "other", 0

    prompt = f"""
你是票据分类助手，请根据 OCR 内容判断票据类型，只输出以下之一：

- 行程单
- 酒店水单
- 支付记录
- 其他

OCR 文本：
{text}
"""

    start = time.time()

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )
        elapsed = time.time() - start
        answer = resp.choices[0].message.content.strip()
    except Exception:
        return "other", 0

    if "行程" in answer:
        return "itinerary", elapsed
    if "酒店" in answer or "水单" in answer:
        return "hotel_folio", elapsed
    if "支付" in answer:
        return "payment", elapsed

    return "other", elapsed


# ============================================================
# 主函数：对单张图片进行处理
# ============================================================
def classify_image(image_path):
    print(f"\n📄 文件: {Path(image_path).name}")

    # OCR
    text, ocr_time = run_ocr(image_path)

    # LLM 分类
    category, llm_time = classify_llm(text)

    print(f"  OCR时间: {ocr_time:.4f}s   LLM时间: {llm_time:.4f}s   => 分类结果: {category}")

    return category


# ============================================================
# 批量处理
# ============================================================
def main():
    if not INPUT_DIR.exists():
        print(f"目录不存在: {INPUT_DIR}")
        return

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        images.extend(INPUT_DIR.glob(ext))

    if not images:
        print("未找到任何图片")
        return

    print(f"📁 找到 {len(images)} 张图片")

    stats = {"itinerary": 0, "hotel_folio": 0, "payment": 0, "other": 0}

    for img in images:
        category = classify_image(str(img))
        stats[category] += 1

    print("\n📊 统计结果")
    for k, v in stats.items():
        print(f"  {k}: {v} 张")


if __name__ == "__main__":
    main()
