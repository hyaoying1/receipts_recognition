import json
from pathlib import Path
from typing import Any, Dict, Tuple, List


# ===== 配置区：按你的文件名修改 =====
PRED_RESULTS_PATH = Path("outputs/qwen_vl_trip_results.json")  # 多模型跑出来的结果
GOLD_RESULTS_PATH = Path("outputs/standard_result.json")     # 你的“完全正确答案”那一版
GOLD_MODEL_NAME = "qwen-vl-max"  # 只是为了打印好看，不用于筛选（下方逻辑按 GOLD_RESULTS_PATH 来）


# ===== 工具：把嵌套 JSON 展平成 path -> value =====
def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """
    递归展平 JSON，得到一个 dict: { "trips[0].city": "北京", "total_amount": 103.0, ... }
    方便逐字段比较。
    """
    items: Dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            items.update(flatten_json(v, new_prefix))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]"
            items.update(flatten_json(v, new_prefix))
    else:
        # 标量（字符串、数字、布尔、null）
        items[prefix] = obj

    return items


# ===== 工具：值相等判定（更鲁棒一点） =====
def values_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    """
    判断两个值是否“相等”：
    - 数字之间按浮点容差比较；
    - 数字字符串 vs 数字，也尝试转成 float 比较；
    - 其他类型直接用 ==。
    """
    # 都是数字类型
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol

    # 数字字符串 vs 数字，或者两个都是数字串
    if isinstance(a, str) and isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) <= tol
        except ValueError:
            return False
    if isinstance(b, str) and isinstance(a, (int, float)):
        try:
            return abs(float(b) - float(a)) <= tol
        except ValueError:
            return False
    if isinstance(a, str) and isinstance(b, str):
        # 可以考虑 strip 一下空格
        return a.strip() == b.strip()

    # 其他类型就直接比较
    return a == b

# ====对单条 trip 做字段对比====
def compare_trip_fields(gold_trip: Dict[str, Any], pred_trip: Dict[str, Any]) -> Tuple[int, int]:
    """
    比较一条行程（一个 dict）：
    返回 (total_fields, correct_fields)
    - total_fields：gold_trip 里字段的数量
    - correct_fields：其中预测正确的字段数量
    """
    total = 0
    correct = 0
    for k, gold_val in gold_trip.items():
        total += 1
        pred_val = pred_trip.get(k, None)
        if values_equal(gold_val, pred_val):
            correct += 1
    return total, correct


# ====对 trips 列表做“忽略顺序”的匹配====
def evaluate_trips_list(gold_trips: List[Dict[str, Any]], pred_trips: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    对 trips 列表做顺序无关的评估：
    返回 (total_fields, correct_fields)

    规则：
    - 针对每一条 gold_trip，在所有尚未匹配的 pred_trip 中
      找“字段匹配数最多”的那一条进行配对。
    - 未配对上的 gold_trip 视为该条所有字段错误。
    - 多出的 pred_trip 当前不额外罚（只统计 gold 方向的召回）。
    """
    total_fields = 0
    correct_fields = 0

    if not gold_trips:
        return 0, 0

    # 标记哪些 pred_trip 已被匹配
    used_pred_indices = set()

    for gold_trip in gold_trips:
        best_match_idx = None
        best_match_correct = -1
        best_match_total = 0

        for i, pred_trip in enumerate(pred_trips):
            if i in used_pred_indices:
                continue
            t, c = compare_trip_fields(gold_trip, pred_trip)
            # 选择“正确字段数”最多的那个
            if c > best_match_correct:
                best_match_correct = c
                best_match_total = t
                best_match_idx = i

        if best_match_idx is not None:
            used_pred_indices.add(best_match_idx)
            total_fields += best_match_total
            correct_fields += best_match_correct
        else:
            # 没有任何可匹配的预测 trip，则这条 gold_trip 的字段全错
            total_fields += len(gold_trip)

    return total_fields, correct_fields

# ===== 主评估逻辑 =====
def load_gold_cases() -> Dict[str, Dict[str, Any]]:
    """
    读取标准答案文件，返回一个映射：
        gold_by_file[_file] = gold_output_json
    """
    with GOLD_RESULTS_PATH.open("r", encoding="utf-8") as f:
        gold_data = json.load(f)

    # 标准答案文件如果是单模型结构（有 "cases"）
    if "cases" in gold_data:
        cases = gold_data["cases"]
    # 如果是和 pred 一样的 task 结构（外面还有 "models"）
    elif "models" in gold_data:
        # 找到 GOLD_MODEL_NAME 对应的模型（如果没找到，就用第一个）
        models = gold_data["models"]
        target = None
        for m in models:
            if m.get("model") == GOLD_MODEL_NAME:
                target = m
                break
        if target is None:
            target = models[0]
        cases = target.get("cases", [])
    else:
        raise ValueError("Gold file 格式不对，既没有 'cases' 也没有 'models'。")

    gold_by_file: Dict[str, Dict[str, Any]] = {}
    for c in cases:
        _file = c["_file"]
        output = c.get("output")
        if output is None:
            # 没有 output 就没法当标准答案，跳过
            continue
        gold_by_file[_file] = output

    return gold_by_file


def evaluate_model(pred_model_entry: Dict[str, Any], gold_by_file: Dict[str, Dict[str, Any]]) -> Tuple[float, float, int, int]:
    """
    对单个模型做评估：
    - pred_model_entry: qwen_vl_trip_results.json 里 "models" 数组中的一个元素
    - gold_by_file: _file -> gold_output

    返回：
    - field_accuracy: 字段级准确率（0~1）
    - exact_match_ratio: 整个 JSON 完全一致的比例（0~1），这里的“完全一致”考虑 trips 顺序无关
    - num_cases_evaluated: 参与评估的 case 数
    - total_fields: 总字段数（分母）
    """
    cases = pred_model_entry.get("cases", [])

    total_fields = 0
    correct_fields = 0

    total_cases = 0
    exact_match_cases = 0

    for case in cases:
        _file = case.get("_file")
        pred_output = case.get("output")
        if _file not in gold_by_file:
            # 标准答案里没有这张票，就跳过
            continue

        gold_output = gold_by_file[_file]
        total_cases += 1

        # ===== 1. 先处理非 trips 的顶层字段 =====
        gold_top = {k: v for k, v in gold_output.items() if k != "trips"}
        pred_top = (pred_output or {})
        pred_top = {k: v for k, v in pred_top.items() if k != "trips"}

        case_all_match = True

        for k, gold_val in gold_top.items():
            total_fields += 1
            pred_val = pred_top.get(k, None)
            if values_equal(gold_val, pred_val):
                correct_fields += 1
            else:
                case_all_match = False

        # 如果标准答案里有字段，预测里完全没这个字段，也会被判错（上面已经加了 total_fields+1）

        # ===== 2. 再处理 trips 列表，顺序无关 =====
        gold_trips = gold_output.get("trips", []) or []
        pred_trips = (pred_output or {}).get("trips", []) or []

        t_fields, c_fields = evaluate_trips_list(gold_trips, pred_trips)
        total_fields += t_fields
        correct_fields += c_fields

        # 判断“整条 JSON 完全一致”（包括 trips，但忽略 trips 顺序）
        # 规则：非 trips 顶层字段全相等 + trips 列表长度相同且视为无序集合时完全一致
        if case_all_match:
            # 顶层字段已经全部命中，再检查 trips 集合是否完全一致
            if len(gold_trips) == len(pred_trips):
                # 将每条 trip 排序后的 key->value 转成 tuple，做 multiset 对比
                def normalize_trip(trip: Dict[str, Any]):
                    return tuple(sorted(trip.items()))

                gold_norm = sorted(normalize_trip(t) for t in gold_trips)
                pred_norm = sorted(normalize_trip(t) for t in pred_trips)
                if gold_norm == pred_norm:
                    # 顶层 + trips 都一致
                    exact_match_cases += 1
                else:
                    # 顶层对了但 trips 内容有偏差
                    pass
            else:
                # trips 数量不相等，肯定不是 exact match
                pass

    field_accuracy = correct_fields / total_fields if total_fields > 0 else 0.0
    exact_match_ratio = exact_match_cases / total_cases if total_cases > 0 else 0.0
    return field_accuracy, exact_match_ratio, total_cases, total_fields

def main():
    # 1. 读标准答案
    gold_by_file = load_gold_cases()
    print(f"[INFO] Gold cases loaded: {len(gold_by_file)} files")

    # 2. 读预测结果（多模型）
    with PRED_RESULTS_PATH.open("r", encoding="utf-8") as f:
        pred_data = json.load(f)

    models = pred_data.get("models", [])
    if not models:
        print("[WARN] No models found in pred results.")
        return

    results_summary: List[Dict[str, Any]] = []

    for m in models:
        model_name = m.get("model", "<unknown>")
        field_acc, em_ratio, num_cases, total_fields = evaluate_model(m, gold_by_file)
        elapsed = m.get("elapsed_seconds", None)

        results_summary.append(
            {
                "model": model_name,
                "field_accuracy": field_acc,
                "exact_match_ratio": em_ratio,
                "num_cases": num_cases,
                "total_fields": total_fields,
                "elapsed_seconds": elapsed,
            }
        )

    # 3. 按字段准确率排序（从高到低），精度相同的按耗时从小到大排
    results_summary.sort(key=lambda x: (-x["field_accuracy"], x["elapsed_seconds"] or 1e9))

    # 4. 打印排名
    print("\n=== 模型精确度排名（字段级） ===")
    print(f"{'Rank':<4} {'Model':<30} {'FieldAcc':>9} {'EM':>9} {'Cases':>7} {'Fields':>8} {'Elapsed(s)':>12}")
    for idx, r in enumerate(results_summary, start=1):
        print(
            f"{idx:<4} "
            f"{r['model']:<30} "
            f"{r['field_accuracy']*100:8.2f}% "
            f"{r['exact_match_ratio']*100:8.2f}% "
            f"{r['num_cases']:7d} "
            f"{r['total_fields']:8d} "
            f"{(r['elapsed_seconds'] or 0):12.2f}"
        )


if __name__ == "__main__":
    main()