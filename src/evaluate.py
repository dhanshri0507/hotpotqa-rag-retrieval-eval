#!/usr/bin/env python3
"""
evaluate.py — Compute retrieval metrics for BM25, Dense, and Hybrid methods.

Metrics: Recall@1, Recall@5, Recall@10, nDCG@10
Broken down overall and by question type (bridge / comparison).

Run from project root:  python -m src.evaluate
"""

import csv
import json
import math
import sys

from src.utils import (
    load_config,
    load_questions,
    load_qrels,
    load_rankings,
    resolve_path,
    check_file_overwrite,
    ensure_dir,
)


# ── Metric functions ─────────────────────────────────────────────────────────

def recall_at_k(ranked_list: list[tuple[str, float]], gold_ids: set[str], k: int) -> float:
    """
    Binary recall: 1.0 if at least one gold doc_id is in the top-k, else 0.0.
    """
    top_k_ids = {doc_id for doc_id, _ in ranked_list[:k]}
    return 1.0 if top_k_ids & gold_ids else 0.0


def dcg_at_k(ranked_list: list[tuple[str, float]], gold_ids: set[str], k: int) -> float:
    """
    DCG@k with binary relevance: rel(i) = 1 if doc_id is gold, else 0.
    DCG = sum_{i=1}^{k} rel(i) / log2(i + 1)
    """
    dcg = 0.0
    for i, (doc_id, _) in enumerate(ranked_list[:k]):
        if doc_id in gold_ids:
            dcg += 1.0 / math.log2(i + 2)  # i is 0-based, so position = i+1, log2(i+2)
    return dcg


def ndcg_at_k(ranked_list: list[tuple[str, float]], gold_ids: set[str], k: int) -> float:
    """
    nDCG@k = DCG@k / IDCG@k, with binary relevance.

    IDCG is the DCG of a perfect ranking (all gold docs first).
    """
    dcg = dcg_at_k(ranked_list, gold_ids, k)

    # Ideal: all gold docs ranked first
    n_gold = len(gold_ids)
    n_ideal = min(n_gold, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ── Self-test: validate metric implementations with a toy example ────────────

def _validate_metrics() -> None:
    """
    Hand-verified toy example:
      3 documents: A, B, C
      1 query with gold = {A, C}
      Ranked list: [(B, 0.9), (A, 0.7), (C, 0.5)]

    Expected:
      Recall@1 = 0.0 (B is not gold)
      Recall@2 = 1.0 (A is gold, in top-2)
      Recall@3 = 1.0

      DCG@3:
        pos 1 (B): 0 / log2(2) = 0
        pos 2 (A): 1 / log2(3) = 0.6309...
        pos 3 (C): 1 / log2(4) = 0.5
        DCG@3 = 1.1309...

      IDCG@3 (ideal: A, C at positions 1,2):
        pos 1: 1 / log2(2) = 1.0
        pos 2: 1 / log2(3) = 0.6309...
        IDCG = 1.6309...

      nDCG@3 = 1.1309... / 1.6309... = 0.6934...
    """
    ranked = [("B", 0.9), ("A", 0.7), ("C", 0.5)]
    gold = {"A", "C"}

    assert recall_at_k(ranked, gold, 1) == 0.0, "recall@1 toy test failed"
    assert recall_at_k(ranked, gold, 2) == 1.0, "recall@2 toy test failed"
    assert recall_at_k(ranked, gold, 3) == 1.0, "recall@3 toy test failed"

    expected_dcg3 = 1.0 / math.log2(3) + 1.0 / math.log2(4)
    actual_dcg3 = dcg_at_k(ranked, gold, 3)
    assert abs(actual_dcg3 - expected_dcg3) < 1e-9, f"dcg@3 toy test failed: {actual_dcg3}"

    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected_ndcg3 = expected_dcg3 / expected_idcg
    actual_ndcg3 = ndcg_at_k(ranked, gold, 3)
    assert abs(actual_ndcg3 - expected_ndcg3) < 1e-9, f"ndcg@3 toy test failed: {actual_ndcg3}"

    print("  ✓ Metric self-test passed (toy example verified)")


# ── Evaluation logic ─────────────────────────────────────────────────────────

def compute_metrics(
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, list[str]],
    question_ids: list[str],
    eval_k_values: list[int],
    ndcg_k: int,
) -> dict:
    """
    Compute Recall@k and nDCG@k for a single method.

    Returns:
        {
            "Recall@1": float, "Recall@5": float, "Recall@10": float,
            "nDCG@10": float,
            "per_query": {qid: {"recall@1": ..., ..., "ndcg@10": ..., "gold": [...], "top10": [...]}}
        }
    """
    recall_sums = {k: 0.0 for k in eval_k_values}
    ndcg_sum = 0.0
    per_query = {}
    n = len(question_ids)

    for qid in question_ids:
        gold_ids = set(qrels.get(qid, []))
        ranked_list = rankings.get(qid, [])

        query_result = {
            "gold_doc_ids": sorted(gold_ids),
            "top_10_retrieved": [did for did, _ in ranked_list[:10]],
        }

        for k in eval_k_values:
            r = recall_at_k(ranked_list, gold_ids, k)
            recall_sums[k] += r
            query_result[f"recall@{k}"] = r

        nd = ndcg_at_k(ranked_list, gold_ids, ndcg_k)
        ndcg_sum += nd
        query_result[f"ndcg@{ndcg_k}"] = nd

        per_query[qid] = query_result

    metrics = {}
    for k in eval_k_values:
        metrics[f"Recall@{k}"] = recall_sums[k] / n
    metrics[f"nDCG@{ndcg_k}"] = ndcg_sum / n
    metrics["per_query"] = per_query

    return metrics


def compute_metrics_by_type(
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, list[str]],
    questions: list[dict],
    eval_k_values: list[int],
    ndcg_k: int,
) -> dict[str, dict]:
    """
    Compute metrics broken down by question type (bridge / comparison).

    Returns: {"bridge": {metric: val, ...}, "comparison": {metric: val, ...}}
    """
    by_type: dict[str, list[str]] = {}
    for q in questions:
        qtype = q["type"]
        by_type.setdefault(qtype, []).append(q["_id"])

    result = {}
    for qtype, qids in sorted(by_type.items()):
        m = compute_metrics(rankings, qrels, qids, eval_k_values, ndcg_k)
        result[qtype] = {k: v for k, v in m.items() if k != "per_query"}
        result[qtype]["count"] = len(qids)

    return result


# ── Output writers ───────────────────────────────────────────────────────────

def save_metrics_overall(all_metrics: dict, path) -> None:
    """Save overall metrics (one entry per method)."""
    out = {}
    for method, m in all_metrics.items():
        out[method] = {k: v for k, v in m.items() if k != "per_query"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def save_metrics_by_type(all_by_type: dict, path) -> None:
    """Save metrics broken down by question type."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_by_type, f, ensure_ascii=False, indent=2)


def save_comparison_table(all_metrics: dict, path) -> None:
    """Save formatted CSV comparison table."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Retrieval Method", "Recall@1", "Recall@5", "Recall@10", "nDCG@10"])
        for method in ["BM25", "Dense", "Hybrid"]:
            m = all_metrics[method]
            writer.writerow([
                method,
                f"{m['Recall@1']:.4f}",
                f"{m['Recall@5']:.4f}",
                f"{m['Recall@10']:.4f}",
                f"{m['nDCG@10']:.4f}",
            ])


def save_query_level_results(all_metrics: dict, questions: list[dict], path) -> None:
    """Save per-query, per-method results to JSONL."""
    qid_to_type = {q["_id"]: q["type"] for q in questions}
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            qid = q["_id"]
            record = {
                "question_id": qid,
                "question": q["question"],
                "type": q["type"],
                "methods": {},
            }
            for method, m in all_metrics.items():
                pq = m["per_query"].get(qid, {})
                record["methods"][method] = pq
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_comparison_table(all_metrics: dict) -> None:
    """Pretty-print the comparison table to console."""
    header = f"{'Method':<10} {'Recall@1':>9} {'Recall@5':>9} {'Recall@10':>10} {'nDCG@10':>9}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for method in ["BM25", "Dense", "Hybrid"]:
        m = all_metrics[method]
        print(
            f"{method:<10} {m['Recall@1']:>9.4f} {m['Recall@5']:>9.4f} "
            f"{m['Recall@10']:>10.4f} {m['nDCG@10']:>9.4f}"
        )
    print(sep)


def print_type_breakdown(all_by_type: dict) -> None:
    """Print metrics by question type."""
    for method in ["BM25", "Dense", "Hybrid"]:
        print(f"\n  {method}:")
        for qtype in sorted(all_by_type[method].keys()):
            m = all_by_type[method][qtype]
            print(
                f"    {qtype:<12} (n={m['count']:>3})  "
                f"R@1={m['Recall@1']:.4f}  R@5={m['Recall@5']:.4f}  "
                f"R@10={m['Recall@10']:.4f}  nDCG@10={m['nDCG@10']:.4f}"
            )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()

    questions = load_questions(cfg)
    qrels = load_qrels(cfg)
    eval_k_values = cfg["eval_k_values"]
    ndcg_k = cfg["ndcg_k"]

    print(f"Loaded {len(questions)} questions, {len(qrels)} qrels entries")
    print(f"Eval: Recall@{eval_k_values}, nDCG@{ndcg_k}")

    # Self-test before trusting metrics on real data
    print("\nMetric validation:")
    _validate_metrics()

    # Define methods and their ranking files
    methods = {
        "BM25": resolve_path(cfg["output_paths"]["bm25_rankings"]),
        "Dense": resolve_path(cfg["output_paths"]["dense_rankings"]),
        "Hybrid": resolve_path(cfg["output_paths"]["hybrid_rankings"]),
    }

    question_ids = [q["_id"] for q in questions]
    all_metrics = {}
    all_by_type = {}

    for method_name, ranking_path in methods.items():
        print(f"\nEvaluating {method_name} ({ranking_path.name}) ...")
        rankings = load_rankings(ranking_path)

        # Verify all questions are present
        missing = set(question_ids) - set(rankings.keys())
        if missing:
            print(f"  ✗ {len(missing)} questions missing from {method_name} rankings!")
            sys.exit(1)

        metrics = compute_metrics(rankings, qrels, question_ids, eval_k_values, ndcg_k)
        all_metrics[method_name] = metrics

        by_type = compute_metrics_by_type(rankings, qrels, questions, eval_k_values, ndcg_k)
        all_by_type[method_name] = by_type

        print(f"  Recall@1={metrics['Recall@1']:.4f}  Recall@5={metrics['Recall@5']:.4f}  "
              f"Recall@10={metrics['Recall@10']:.4f}  nDCG@10={metrics['nDCG@10']:.4f}")

    # ── Save all output files ────────────────────────────────────────────
    print("\nSaving results ...")

    overall_path = resolve_path(cfg["output_paths"]["metrics_overall"])
    by_type_path = resolve_path(cfg["output_paths"]["metrics_by_question_type"])
    table_path = resolve_path(cfg["output_paths"]["comparison_table"])
    query_path = resolve_path(cfg["output_paths"]["query_level_results"])

    for p, name in [
        (overall_path, "evaluate"),
        (by_type_path, "evaluate"),
        (table_path, "evaluate"),
        (query_path, "evaluate"),
    ]:
        check_file_overwrite(p, name)

    ensure_dir(overall_path.parent)
    save_metrics_overall(all_metrics, overall_path)
    print(f"  -> {overall_path.name}")

    save_metrics_by_type(all_by_type, by_type_path)
    print(f"  -> {by_type_path.name}")

    save_comparison_table(all_metrics, table_path)
    print(f"  -> {table_path.name}")

    save_query_level_results(all_metrics, questions, query_path)
    print(f"  -> {query_path.name}")

    # ── Print comparison table ───────────────────────────────────────────
    print("\n\n=== OVERALL COMPARISON ===\n")
    print_comparison_table(all_metrics)

    print("\n=== BY QUESTION TYPE ===")
    print_type_breakdown(all_by_type)

    print("\n\nAll result files saved to results/")


if __name__ == "__main__":
    main()
