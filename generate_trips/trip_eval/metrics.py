# trip_eval/metrics.py
from typing import Any, Dict, List, Tuple

TOP_FIELDS = ["type", "vendor", "apply_date", "start_date", "end_date", "total_amount"]
TRIP_FIELDS = ["city", "date", "start_time", "line_amount", "currency"]


def equal_value(pred: Any, gold: Any, num_tol: float = 1e-2) -> bool:
    """容忍一点数值误差，其余按字符串精确比较。"""
    if pred is None and gold is None:
        return True

    if isinstance(gold, (int, float)) or isinstance(pred, (int, float)):
        try:
            p = float(pred)
            g = float(gold)
            return abs(p - g) <= num_tol
        except Exception:
            return str(pred) == str(gold)

    return str(pred) == str(gold)


def compare_record_keep_trip_order(
    pred: Dict[str, Any], gold: Dict[str, Any]
) -> Tuple[bool, int, int]:
    """
    顺序敏感版本：trips 按索引一一对应，顺序不同就算错。
    返回: (整条是否完全一致, 匹配字段数, 总字段数)
    """
    match = True
    field_matches = 0
    field_total = 0

    # 顶层字段
    for key in TOP_FIELDS:
        field_total += 1
        if equal_value(pred.get(key), gold.get(key)):
            field_matches += 1
        else:
            match = False

    gold_trips = gold.get("trips", []) or []
    pred_trips = pred.get("trips", []) or []

    n = min(len(gold_trips), len(pred_trips))
    for i in range(n):
        g_trip = gold_trips[i]
        p_trip = pred_trips[i]
        for key in TRIP_FIELDS:
            field_total += 1
            if equal_value(p_trip.get(key), g_trip.get(key)):
                field_matches += 1
            else:
                match = False

    return match, field_matches, field_total


def _trip_signature(trip: Dict[str, Any]) -> Tuple:
    """忽略顺序时，用这个 key 来排序 / 匹配 trip。"""
    return (
        trip.get("city"),
        trip.get("date"),
        trip.get("start_time"),
        float(trip.get("line_amount") or 0.0),
        trip.get("currency"),
    )


def compare_record_ignore_trip_order(
    pred: Dict[str, Any], gold: Dict[str, Any]
) -> Tuple[bool, int, int]:
    """
    顺序不敏感版本：顶层字段按 key 比，trips 当集合比，不要求顺序一致。
    """
    match = True
    field_matches = 0
    field_total = 0

    # 顶层字段
    for key in TOP_FIELDS:
        field_total += 1
        if equal_value(pred.get(key), gold.get(key)):
            field_matches += 1
        else:
            match = False

    gold_trips: List[Dict[str, Any]] = gold.get("trips", []) or []
    pred_trips: List[Dict[str, Any]] = pred.get("trips", []) or []

    gold_sorted = sorted(gold_trips, key=_trip_signature)
    pred_sorted = sorted(pred_trips, key=_trip_signature)

    n = min(len(gold_sorted), len(pred_sorted))
    for i in range(n):
        g_trip = gold_sorted[i]
        p_trip = pred_sorted[i]
        for key in TRIP_FIELDS:
            field_total += 1
            if equal_value(p_trip.get(key), g_trip.get(key)):
                field_matches += 1
            else:
                match = False

    # 如果条数不同，多出来的 trip 算所有字段都错
    if len(gold_sorted) != len(pred_sorted):
        match = False
        diff = abs(len(gold_sorted) - len(pred_sorted))
        field_total += diff * len(TRIP_FIELDS)

    return match, field_matches, field_total