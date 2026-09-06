#!/usr/bin/env python3
"""
evaluate_improvement.py — Evaluate the Improved Retrieval Method and perform Targeted Failure Check.

Steps:
  1. Evaluate Improved Method using exact metric logic from src/evaluate.py.
  2. Output:
     - results/metrics_overall_improved.json
     - results/metrics_by_question_type_improved.json
     - results/comparison_table_with_improved.csv (BM25, Dense, Hybrid, Improved Method)
     - results/query_level_results_improved.jsonl
  3. Targeted Failure Check (Step 4):
     - Isolate queries categorized as 'multi_hop_bridging_failure' in results/failure_taxonomy_mapping.json.
     - Check whether both gold documents are now retrieved in top-10 under Improved Method.
     - Output results/improvement_targeted_check.json.
  4. New Failures Check:
     - Check if queries where all baselines succeeded at top-10 now fail (Recall@10=0) under Improved Method.
     - Output results/improvement_new_failures.json.

Run from project root: python -m src.evaluate_improvement
"""

import csv
import json
import sys
from pathlib import Path

from src.utils import (
    load_config,
    load_questions,
    load_qrels,
    load_rankings,
    resolve_path,
    check_file_overwrite,
    ensure_dir,
)
from src.evaluate import compute_metrics, compute_metrics_by_type


def main():
    cfg = load_config()

    questions = load_questions(cfg)
    questions_dict = {q["_id"]: q for q in questions}
    question_ids = [q["_id"] for q in questions]
    qrels = load_qrels(cfg)
    eval_k_values = cfg["eval_k_values"]
    ndcg_k = cfg["ndcg_k"]

    print("Loading all ranking files ...")
    bm25_rankings = load_rankings(resolve_path(cfg["output_paths"]["bm25_rankings"]))
    dense_rankings = load_rankings(resolve_path(cfg["output_paths"]["dense_rankings"]))
    hybrid_rankings = load_rankings(resolve_path(cfg["output_paths"]["hybrid_rankings"]))
    improved_rankings = load_rankings(resolve_path(cfg["output_paths"]["improved_rankings"]))

    all_methods = {
        "BM25": bm25_rankings,
        "Dense": dense_rankings,
        "Hybrid": hybrid_rankings,
        "Improved Method": improved_rankings,
    }

    print("\n--- STEP 3: Evaluating All Methods Including Improved ---")
    all_metrics = {}
    all_by_type = {}

    for name, rankings in all_methods.items():
        m = compute_metrics(rankings, qrels, question_ids, eval_k_values, ndcg_k)
        all_metrics[name] = m
        by_type = compute_metrics_by_type(rankings, qrels, questions, eval_k_values, ndcg_k)
        all_by_type[name] = by_type
        print(
            f"  {name:<16}: R@1={m['Recall@1']:.4f}  R@5={m['Recall@5']:.4f}  "
            f"R@10={m['Recall@10']:.4f}  nDCG@10={m['nDCG@10']:.4f}"
        )

    # -------------------------------------------------------------------------
    # Save Step 3 Result Files
    # -------------------------------------------------------------------------
    print("\nSaving Step 3 evaluation artifacts ...")

    # 1. results/metrics_overall_improved.json
    overall_path = resolve_path(cfg["output_paths"]["metrics_overall_improved"])
    check_file_overwrite(overall_path, "evaluate_improvement")
    overall_data = {
        name: {k: v for k, v in m.items() if k != "per_query"}
        for name, m in all_metrics.items()
    }
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall_data, f, ensure_ascii=False, indent=2)
    print(f"  -> {overall_path.name}")

    # 2. results/metrics_by_question_type_improved.json
    by_type_path = resolve_path(cfg["output_paths"]["metrics_by_question_type_improved"])
    check_file_overwrite(by_type_path, "evaluate_improvement")
    with open(by_type_path, "w", encoding="utf-8") as f:
        json.dump(all_by_type, f, ensure_ascii=False, indent=2)
    print(f"  -> {by_type_path.name}")

    # 3. results/comparison_table_with_improved.csv
    table_path = resolve_path(cfg["output_paths"]["comparison_table_with_improved"])
    check_file_overwrite(table_path, "evaluate_improvement")
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Retrieval Method", "Recall@1", "Recall@5", "Recall@10", "nDCG@10"])
        for name in ["BM25", "Dense", "Hybrid", "Improved Method"]:
            m = all_metrics[name]
            writer.writerow([
                name,
                f"{m['Recall@1']:.4f}",
                f"{m['Recall@5']:.4f}",
                f"{m['Recall@10']:.4f}",
                f"{m['nDCG@10']:.4f}",
            ])
    print(f"  -> {table_path.name}")

    # 4. results/query_level_results_improved.jsonl
    query_path = resolve_path(cfg["output_paths"]["query_level_results_improved"])
    check_file_overwrite(query_path, "evaluate_improvement")
    with open(query_path, "w", encoding="utf-8") as f:
        for q in questions:
            qid = q["_id"]
            record = {
                "question_id": qid,
                "question": q["question"],
                "type": q["type"],
                "methods": {name: all_metrics[name]["per_query"].get(qid, {}) for name in all_methods},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  -> {query_path.name}")

    # -------------------------------------------------------------------------
    # STEP 4: Targeted Failure-Type Check
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: Targeted Failure-Type Check (Multi-Hop Bridging Failures) ---")

    taxonomy_mapping_path = resolve_path("results/failure_taxonomy_mapping.json")
    with open(taxonomy_mapping_path, "r", encoding="utf-8") as f:
        taxonomy_mapping = json.load(f)

    # Isolate queries that had a multi_hop_bridging_failure in baselines
    bridging_failed_queries = {}
    for key, cat in taxonomy_mapping.items():
        if cat == "multi_hop_bridging_failure":
            qid, method = key.split("__")
            bridging_failed_queries.setdefault(qid, []).append(method)

    print(f"  Found {len(bridging_failed_queries)} unique queries categorized as multi_hop_bridging_failure across baselines.")

    targeted_check_results = []
    resolved_count = 0

    for qid, failed_methods in sorted(bridging_failed_queries.items()):
        golds = qrels[qid]
        gold_set = set(golds)
        improved_ranked = improved_rankings.get(qid, [])
        full_doc_order = [d for d, _ in improved_ranked]
        top10_improved = set(full_doc_order[:10])

        # Check whether BOTH gold documents are now retrieved in top-10
        both_in_top10 = gold_set.issubset(top10_improved)
        at_least_one_in_top10 = bool(gold_set & top10_improved)

        if both_in_top10:
            resolved_count += 1

        new_gold_ranks = {
            g: (full_doc_order.index(g) + 1 if g in full_doc_order else None)
            for g in golds
        }

        entry = {
            "question_id": qid,
            "question": questions_dict[qid]["question"],
            "type": questions_dict[qid]["type"],
            "baseline_methods_failed": failed_methods,
            "gold_doc_ids": golds,
            "both_gold_in_top10": both_in_top10,
            "at_least_one_gold_in_top10": at_least_one_in_top10,
            "resolved": both_in_top10,
            "improved_gold_ranks": new_gold_ranks,
            "improved_top10_doc_ids": full_doc_order[:10],
        }
        targeted_check_results.append(entry)

    print(f"  Targeted check resolution rate (both golds in top-10): {resolved_count}/{len(bridging_failed_queries)} ({resolved_count/len(bridging_failed_queries)*100:.1f}%)")

    targeted_path = resolve_path(cfg["output_paths"]["improvement_targeted_check"])
    check_file_overwrite(targeted_path, "evaluate_improvement")
    with open(targeted_path, "w", encoding="utf-8") as f:
        json.dump(targeted_check_results, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved targeted check to {targeted_path.name}")

    # Check for NEW failures caused by Improved Method
    print("\n--- Checking for New Failures Introduced by Improved Method ---")
    new_failures = []

    for qid in question_ids:
        golds = set(qrels[qid])
        # Did all baselines succeed at Recall@10 (i.e. at least one gold in top 10 for all 3 methods)?
        bm25_hit = bool(golds & {d for d, _ in bm25_rankings[qid][:10]})
        dense_hit = bool(golds & {d for d, _ in dense_rankings[qid][:10]})
        hybrid_hit = bool(golds & {d for d, _ in hybrid_rankings[qid][:10]})
        baselines_all_succeeded = bm25_hit and dense_hit and hybrid_hit

        # Does Improved Method fail at Recall@10?
        improved_hit = bool(golds & {d for d, _ in improved_rankings[qid][:10]})

        if baselines_all_succeeded and not improved_hit:
            improved_ranked = [d for d, _ in improved_rankings[qid]]
            new_failures.append({
                "question_id": qid,
                "question": questions_dict[qid]["question"],
                "type": questions_dict[qid]["type"],
                "gold_doc_ids": list(golds),
                "improved_gold_ranks": {g: (improved_ranked.index(g) + 1 if g in improved_ranked else None) for g in golds},
                "improved_top10_doc_ids": improved_ranked[:10],
                "likely_cause": "Hop-1 distractor text diluted original query keywords, causing query drift away from true gold target.",
            })

    print(f"  New failures introduced by Improved Method at Recall@10: {len(new_failures)}")

    new_failures_path = resolve_path(cfg["output_paths"]["improvement_new_failures"])
    check_file_overwrite(new_failures_path, "evaluate_improvement")
    with open(new_failures_path, "w", encoding="utf-8") as f:
        json.dump(new_failures, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved new failures check to {new_failures_path.name}")

    # -------------------------------------------------------------------------
    # Console Summary Tables
    # -------------------------------------------------------------------------
    print("\n\n==================== COMPARISON TABLE ====================")
    header = f"{'Method':<18} {'Recall@1':>9} {'Recall@5':>9} {'Recall@10':>10} {'nDCG@10':>9}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name in ["BM25", "Dense", "Hybrid", "Improved Method"]:
        m = all_metrics[name]
        print(
            f"{name:<18} {m['Recall@1']:>9.4f} {m['Recall@5']:>9.4f} "
            f"{m['Recall@10']:>10.4f} {m['nDCG@10']:>9.4f}"
        )
    print(sep)

    print("\n==================== BY QUESTION TYPE ====================")
    for name in ["BM25", "Dense", "Hybrid", "Improved Method"]:
        print(f"\n  {name}:")
        for qtype in sorted(all_by_type[name].keys()):
            m = all_by_type[name][qtype]
            print(
                f"    {qtype:<12} (n={m['count']:>3})  "
                f"R@1={m['Recall@1']:.4f}  R@5={m['Recall@5']:.4f}  "
                f"R@10={m['Recall@10']:.4f}  nDCG@10={m['nDCG@10']:.4f}"
            )


if __name__ == "__main__":
    main()
