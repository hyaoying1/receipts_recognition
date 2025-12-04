from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR(
    lang="ch",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

ocr_executor = ThreadPoolExecutor(max_workers=8)


def run_ocr(path: Path) -> str:
    result, _ = ocr(str(path))
    if result:
        return "\n".join([line[1] for line in result])
    return ""


async def run_ocr_async(path: Path) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(ocr_executor, run_ocr, path)



def fuzzy_contains(text: str, keyword: str, threshold: float = 0.75) -> bool:
    keyword = keyword.lower()

    # exact match
    if keyword in text:
        return True

    # fuzzy match sliding window
    window = len(keyword)
    for i in range(len(text) - window + 1):
        piece = text[i:i + window]
        if SequenceMatcher(None, piece, keyword).ratio() >= threshold:
            return True

    return False


async def rule_classify(text: str) -> str:
    if len(text.strip()) < 3:
        return "other"

    text = text.lower()

    # Keyword sets
    itinerary_keywords = [
        "起点", "终点", "公里", "行程", "出行",
        "快车", "特惠快车", "专车",
        "itinerary", "route", "km", "distance"
    ]

    hotel_keywords = [
        "酒店", "到店", "房间", "房费",
        "入住", "离店", "房价", "房号", "住宿",
        "hotel", "room", "check-in", "check out", "guest"
    ]

    payment_keywords = [
        "支付", "支付方式",
        "payment", "payment method",
        "付款", "金额", "交易"
    ]

    # scoring
    def score_keywords(keywords):
        return sum(1 for kw in keywords if fuzzy_contains(text, kw))

    itinerary_score = score_keywords(itinerary_keywords)
    hotel_score = score_keywords(hotel_keywords)
    payment_score = score_keywords(payment_keywords)

    scores = {
        "itinerary": itinerary_score,
        "hotel_invoice": hotel_score,
        "payment": payment_score,
    }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score >= 2:
        return best_type

    return "other"


async def classification_one(image_path: Path) -> dict:
    text = await run_ocr_async(image_path)
    doc_type = await rule_classify(text)
    return {"file": str(image_path), "type": doc_type}


async def classification_batch(image_paths: list[Path]) -> list[dict]:
    # parallel OCR
    ocr_tasks = [run_ocr_async(p) for p in image_paths]
    texts = await asyncio.gather(*ocr_tasks)

    # parallel classification
    classify_tasks = [rule_classify(t) for t in texts]
    types = await asyncio.gather(*classify_tasks)

    return [
        {"file": str(p), "type": t}
        for p, t in zip(image_paths, types)
    ]
