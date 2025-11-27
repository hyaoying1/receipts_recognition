import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import itertools

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  
# 这里改成你真实的评测结果路径
EVAL_PATH = PROJECT_ROOT / "outputs/qwen_8b_outputs.json"

TOP_FIELDS = ["type", "vendor", "apply_date", "start_date", "end_date", "total_amount"]
TRIP_FIELDS = ["city", "date", "start_time", "line_amount", "currency"]


def _equal_value(pred: Any, gold: Any, num_tol: float = 1e-2) -> bool:
    """数值允许一点误差，其它用字符串直接比较。"""
    if pred is None and gold is None:
        return True

    # 数字比较
    if isinstance(gold, (int, float)) or isinstance(pred, (int, float)):
        try:
            p = float(pred)
            g = float(gold)
            return abs(p - g) <= num_tol
        except Exception:
            return str(pred) == str(gold)

    # 其他情况
    return str(pred) == str(gold)


def best_trip_alignment_dp(
    pred_trips: List[Dict[str, Any]],
    gold_trips: List[Dict[str, Any]],
) -> List[int]:
    """
    忽略顺序：找一个 pred->gold 的一一配对，使得 TRIP_FIELDS 命中总数最大。
    返回 perm：长度 n_common，perm[k] 是与 gold_trips[k] 对齐的 pred index。

    语义等价于 permutations 暴力最优，但复杂度 O(n^2 * 2^n)。
    """
    n_gold = len(gold_trips)
    n_pred = len(pred_trips)
    n_common = min(n_gold, n_pred)
    if n_common == 0:
        return []

    # 预计算得分矩阵 score[i][j]
    score = [[0] * n_pred for _ in range(n_common)]
    for i in range(n_common):
        g = gold_trips[i]
        for j in range(n_pred):
            p = pred_trips[j]
            s = 0
            for key in TRIP_FIELDS:
                if _equal_value(p.get(key), g.get(key)):
                    s += 1
            score[i][j] = s

    # dp[i][mask]：gold 前 i 条已匹配，pred 已用集合 mask 时最大得分
    dp = {0: 0}
    parent = {}  # (i, mask) -> (prev_mask, chosen_pred_idx)

    for i in range(n_common):
        new_dp = {}
        for mask, val in dp.items():
            for j in range(n_pred):
                if mask & (1 << j):
                    continue
                nm = mask | (1 << j)
                nv = val + score[i][j]
                if nm not in new_dp or nv > new_dp[nm]:
                    new_dp[nm] = nv
                    parent[(i + 1, nm)] = (mask, j)
        dp = new_dp

    # 取最优 mask
    best_mask = max(dp, key=lambda m: dp[m])

    # 回溯 perm
    perm = [None] * n_common
    mask = best_mask
    for i in range(n_common, 0, -1):
        pmask, j = parent[(i, mask)]
        perm[i - 1] = j
        mask = pmask

    return perm

def compare_record_ignore_trip_order(
    pred: Dict[str, Any],
    gold: Dict[str, Any],
) -> Tuple[bool, int, int, Dict[str, int], Dict[str, int]]:
    """
    trips 忽略顺序 + DP 最优对齐。

    record_equal 条件（与你注释一致）：
      * 顶层全部字段相等
      * len(pred_trips) == len(gold_trips)
      * 在最优对齐下，所有 trip 字段都相等

    field_total 口径：
      * 对齐的 n_common 部分逐字段计数
      * 长度不等时，多出来的 trip *字段数 计入 field_total 且视为全错
        （与 bad case 规则一致）
    """
    record_equal = True
    field_matches = 0
    field_total = 0

    top_match_by_key = {k: 0 for k in TOP_FIELDS}
    trip_match_by_key = {k: 0 for k in TRIP_FIELDS}

    # ===== 顶层字段 =====
    for key in TOP_FIELDS:
        field_total += 1
        if _equal_value(pred.get(key), gold.get(key)):
            field_matches += 1
            top_match_by_key[key] += 1
        else:
            record_equal = False

    # ===== trips 部分（忽略顺序 + DP最优对齐）=====
    gold_trips = gold.get("trips", []) or []
    pred_trips = pred.get("trips", []) or []

    n_gold = len(gold_trips)
    n_pred = len(pred_trips)
    n_common = min(n_gold, n_pred)

    # 1) 先做最优对齐（只对齐 common 部分）
    perm = best_trip_alignment_dp(pred_trips, gold_trips)

    trips_all_equal = True
    for k in range(n_common):
        g_trip = gold_trips[k]
        p_trip = pred_trips[perm[k]]

        for key in TRIP_FIELDS:
            field_total += 1
            if _equal_value(p_trip.get(key), g_trip.get(key)):
                field_matches += 1
                trip_match_by_key[key] += 1
            else:
                trips_all_equal = False

    # 2) 长度不等：多出的 trips 视为全错，并计入 field_total
    if n_gold != n_pred:
        trips_all_equal = False
        extra = abs(n_gold - n_pred)
        field_total += extra * len(TRIP_FIELDS)
        # field_matches 不加，因为视为全错

    if not trips_all_equal:
        record_equal = False

    return record_equal, field_matches, field_total, top_match_by_key, trip_match_by_key


def main():
    eval_path = EVAL_PATH
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    with eval_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    models = eval_data.get("models", [])
    if not models:
        print("[ERR] No models found in eval file.")
        return

    for model_entry in models:
        model_name = model_entry.get("model", "<unknown>")
        cases = model_entry.get("cases", [])
        print(f"\n===== Re-eval (trips order ignored) for model: {model_name} =====")

        record_equal_count = 0
        sample_count = 0
        global_field_matches = 0
        global_field_total = 0

        # 各字段统计：顶层 & trips
        top_correct = {k: 0 for k in TOP_FIELDS}
        top_total = {k: 0 for k in TOP_FIELDS}
        trip_correct = {k: 0 for k in TRIP_FIELDS}
        trip_total = {k: 0 for k in TRIP_FIELDS}

        for case in cases:
            gold = case.get("gold")
            pred = case.get("output")

            if gold is None:
                continue  # 理论上不会

            sample_count += 1

            if pred is None:
                # 完全解析失败：这条样本所有字段都视为错
                # 但为了保持逻辑简单，这里只给顶层加分母，不给 trips 加
                for key in TOP_FIELDS:
                    top_total[key] += 1
                record_equal = False
                field_matches = 0
                field_total = len(TOP_FIELDS)
            else:
                record_equal, fm, ft, top_match_by_key, trip_match_by_key = compare_record_ignore_trip_order(
                    pred, gold
                )
                field_matches = fm
                field_total = ft

                # 统计各字段命中数 & 分母
                for key in TOP_FIELDS:
                    top_total[key] += 1
                    if top_match_by_key.get(key, 0) > 0:
                        top_correct[key] += 1

                gold_trips = gold.get("trips", []) or []
                pred_trips = pred.get("trips", []) or []
                n_common = min(len(gold_trips), len(pred_trips))
                for key in TRIP_FIELDS:
                    trip_total[key] += n_common  # 每条样本，这个字段在 trips 中有 n_common 次比较
                    trip_correct[key] += trip_match_by_key.get(key, 0)

            if record_equal:
                record_equal_count += 1

            global_field_matches += field_matches
            global_field_total += field_total

        # ===== 打印整体指标 =====
        print(f"[INFO] Samples evaluated: {sample_count}")
        if sample_count > 0:
            print(
                f"Record-level accuracy (trips 忽略顺序): "
                f"{record_equal_count}/{sample_count} = {record_equal_count / sample_count:.3f}"
            )
        if global_field_total > 0:
            print(
                f"Field-level accuracy (整体字段，micro-avg): "
                f"{global_field_matches}/{global_field_total} = {global_field_matches / global_field_total:.3f}"
            )

        # ===== 打印各字段正确率 =====
        print("\n[Top-level fields accuracy]")
        for key in TOP_FIELDS:
            tot = top_total[key]
            cor = top_correct[key]
            acc = cor / tot if tot > 0 else 0.0
            print(f"  {key:12s}: {cor}/{tot} = {acc:.3f}")

        print("\n[Trip fields accuracy]  (trips 内部忽略顺序，对齐 min(len(pred), len(gold)))")
        for key in TRIP_FIELDS:
            tot = trip_total[key]
            cor = trip_correct[key]
            acc = cor / tot if tot > 0 else 0.0
            print(f"  {key:12s}: {cor}/{tot} = {acc:.3f}")

        print("============================================================\n")


if __name__ == "__main__":
    main()