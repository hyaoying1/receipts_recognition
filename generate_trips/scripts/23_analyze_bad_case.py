import json
from pathlib import Path
from typing import Any, Dict, List
import itertools
from pathlib import Path
import sys
# 1. 项目根目录：.../OCR_test_2
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 2. 加入 sys.path，这样就能 import trip_eval 了
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from trip_eval.paths import PROJECT_ROOT, OUTPUT_DIR

EVAL_PATH = OUTPUT_DIR / "qwen_8b_outputs.json"
OUTPUT_PATH = OUTPUT_DIR/'qwen_8b_bad_cases.json'

TOP_FIELDS = ["type", "vendor", "apply_date", "start_date", "end_date", "total_amount"]
TRIP_FIELDS = ["city", "date", "start_time", "line_amount", "currency"]


def _equal_value(pred: Any, gold: Any, num_tol: float = 0) -> bool:
    """数值允许一点误差，其它用字符串比较。"""
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


def best_trip_alignment(
    pred_trips: List[Dict[str, Any]],
    gold_trips: List[Dict[str, Any]],
) -> List[int]:
    """
    忽略顺序：在 pred_trips 里找一个一一配对，使得和 gold_trips 对齐时总命中字段数最大。
    返回长度 = n_common 的索引列表 perm，表示 gold[k] 对齐 pred[perm[k]]。

    语义与原先 permutations 暴力完全一致，但复杂度从 O(n!) 降到 O(n^2 * 2^n)
    """
    n_gold = len(gold_trips)
    n_pred = len(pred_trips)
    n_common = min(n_gold, n_pred)
    if n_common == 0:
        return []

    # 预计算得分矩阵 score[i][j] = gold i 与 pred j 命中的字段数
    score = [[0]*n_pred for _ in range(n_common)]
    for i in range(n_common):
        g = gold_trips[i]
        for j in range(n_pred):
            p = pred_trips[j]
            s = 0
            for key in TRIP_FIELDS:
                if _equal_value(p.get(key), g.get(key)):
                    s += 1
            score[i][j] = s

    # dp[i][mask]：gold 前 i 条已匹配，使用 pred 子集 mask 时的最大得分
    dp = {0: 0}
    parent = {}  # (i, mask) -> (prev_mask, chosen_pred_index)

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
                    parent[(i+1, nm)] = (mask, j)
        dp = new_dp

    # 找到得分最高的 mask
    best_mask = max(dp, key=lambda m: dp[m])

    # 回溯得到 perm
    perm = [None] * n_common
    mask = best_mask
    for i in range(n_common, 0, -1):
        pmask, j = parent[(i, mask)]
        perm[i-1] = j
        mask = pmask

    return perm


def analyze_case(gold: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    返回该样本的错误详情：
    {
      "top_wrong": ["start_date", ...],
      "trip_len_mismatch": bool,
      "trip_wrong": [
        {
          "gold_idx": 0,
          "pred_idx": 1,
          "wrong_fields": ["line_amount", "date"]
        },
        ...
      ]
    }
    """
    result = {
        "top_wrong": [],
        "trip_len_mismatch": False,
        "trip_wrong": [],
    }

    # 顶层字段
    for key in TOP_FIELDS:
        if not _equal_value(pred.get(key), gold.get(key)):
            result["top_wrong"].append(key)

    gold_trips = gold.get("trips", []) or []
    pred_trips = pred.get("trips", []) or []

    n_gold = len(gold_trips)
    n_pred = len(pred_trips)
    n_common = min(n_gold, n_pred)

    if n_gold != n_pred:
        result["trip_len_mismatch"] = True

    if n_common == 0:
        return result

    perm = best_trip_alignment(pred_trips, gold_trips)

    for k in range(n_common):
        g_trip = gold_trips[k]
        p_trip = pred_trips[perm[k]]
        wrong_fields = []
        for key in TRIP_FIELDS:
            if not _equal_value(p_trip.get(key), g_trip.get(key)):
                wrong_fields.append(key)
        if wrong_fields:
            result["trip_wrong"].append(
                {
                    "gold_idx": k,
                    "pred_idx": perm[k],
                    "wrong_fields": wrong_fields,
                }
            )

    return result


def main():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Eval file not found: {EVAL_PATH}")

    with EVAL_PATH.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    models = eval_data.get("models", [])
    if not models:
        print("[ERR] No models in eval file.")
        return

    for model_entry in models:
        model_name = model_entry.get("model", "<unknown>")
        cases = model_entry.get("cases", [])
        print(f"\n===== BAD CASE ANALYSIS for model: {model_name} =====")

        bad_cases = []
        # 为了按字段看 bad case，再做一个字段 -> [id,...] 的索引
        top_bad_index = {k: [] for k in TOP_FIELDS}
        trip_bad_index = {k: [] for k in TRIP_FIELDS}

        for case in cases:
            gold = case.get("gold")
            pred = case.get("output")

            if gold is None:
                continue

            if pred is None:
                # 完全失败：所有顶层字段 + trips 视为错
                wrong = {
                    "top_wrong": TOP_FIELDS[:],
                    "trip_len_mismatch": True,
                    "trip_wrong": [],
                }
            else:
                wrong = analyze_case(gold, pred)

            # 判断是否 bad case
            if wrong["top_wrong"] or wrong["trip_len_mismatch"] or wrong["trip_wrong"]:
                bad_entry = {
                    "id": case.get("id"),
                    "image": case.get("image"),
                    "top_wrong": wrong["top_wrong"],
                    "trip_len_mismatch": wrong["trip_len_mismatch"],
                    "trip_wrong": wrong["trip_wrong"],
                }
                bad_cases.append(bad_entry)

                sid = case.get("id")
                # 顶层字段错误索引
                for key in wrong["top_wrong"]:
                    top_bad_index[key].append(sid)
                # trips 字段错误索引
                if wrong["trip_len_mismatch"]:
                    # 长度不等：所有 trip 字段都算有问题
                    for key in TRIP_FIELDS:
                        trip_bad_index[key].append(sid)
                else:
                    # 具体哪几个字段错
                    wrong_fields_set = set()
                    for tw in wrong["trip_wrong"]:
                        wrong_fields_set.update(tw["wrong_fields"])
                    for key in wrong_fields_set:
                        trip_bad_index[key].append(sid)

        print(f"[INFO] Total bad cases: {len(bad_cases)}")

        # 1) 终端打印：每个字段有哪些样本错了（只打印 id 列表）
        print("\n[Top-level field bad cases]")
        for key in TOP_FIELDS:
            ids = top_bad_index[key]
            print(f"  {key:12s} ({len(ids)}): {ids}")

        print("\n[Trip field bad cases]  (trips 顺序已忽略)")
        for key in TRIP_FIELDS:
            ids = trip_bad_index[key]
            print(f"  {key:12s} ({len(ids)}): {ids}")

        # 2) 写一份详细 bad case JSON，方便你之后用 jq / Python 再分析
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(bad_cases, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote bad cases eval results to {OUTPUT_PATH}")

        


if __name__ == "__main__":
    main()