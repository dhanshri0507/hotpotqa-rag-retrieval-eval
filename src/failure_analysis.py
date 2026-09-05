#!/usr/bin/env python3
"""
failure_analysis.py — Systematic Failure Analysis for HotpotQA RAG Retrieval.

Implements:
  1. Predefined, reproducible threshold selection (Recall@10=0 widened to Recall@5=0).
  2. Extraction of raw failure cases to results/failure_cases_raw.jsonl.
  3. Structured taxonomy classification covering 100% of failure cases.
  4. Output of results/failure_taxonomy_mapping.json and results/failure_taxonomy_summary.json.

Run from project root: python -m src.failure_analysis
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

from src.utils import (
    load_config,
    load_corpus,
    load_questions,
    load_qrels,
    load_rankings,
    resolve_path,
    check_file_overwrite,
    ensure_dir,
)

# -----------------------------------------------------------------------------
# STEP 1: Predefined Threshold Selection Rule
# -----------------------------------------------------------------------------

def evaluate_thresholds(
    qrels: dict[str, list[str]],
    bm25_rankings: dict,
    dense_rankings: dict,
    hybrid_rankings: dict,
    min_unique_queries: int = 20,
) -> tuple[int, list[tuple[str, str]], set[str]]:
    """
    Evaluate thresholds starting from Recall@10 = 0.
    Widens to Recall@5 = 0, then Recall@1 = 0 if needed to reach min_unique_queries floor.

    Returns:
        (chosen_k, failure_pairs, unique_failing_qids)
        where failure_pairs is list of (question_id, method_name).
    """
    methods = [("bm25", bm25_rankings), ("dense", dense_rankings), ("hybrid", hybrid_rankings)]

    for k in [10, 5, 1]:
        failing_pairs = []
        unique_qids = set()

        for method_name, rankings in methods:
            for qid, gold_titles in qrels.items():
                gold_set = set(gold_titles)
                top_k_retrieved = {doc_id for doc_id, _ in rankings[qid][:k]}
                # Recall@k == 0 means no gold document retrieved in top-k
                if not (top_k_retrieved & gold_set):
                    failing_pairs.append((qid, method_name))
                    unique_qids.add(qid)

        print(f"  Evaluating threshold Recall@{k}=0: found {len(unique_qids)} unique queries across {len(failing_pairs)} (query, method) instances")

        if len(unique_qids) >= min_unique_queries:
            print(f"  -> Threshold criterion satisfied at Recall@{k}=0 ({len(unique_qids)} unique queries >= {min_unique_queries} floor).")
            return k, failing_pairs, unique_qids

    # Fallback if even k=1 didn't reach floor
    return 1, failing_pairs, unique_qids


# -----------------------------------------------------------------------------
# STEP 2: Extract Raw Failure Case Data
# -----------------------------------------------------------------------------

def extract_failure_cases(
    failing_pairs: list[tuple[str, str]],
    questions_dict: dict[str, dict],
    qrels: dict[str, list[str]],
    all_rankings: dict[str, dict],
) -> list[dict]:
    """
    Extract detailed failure records for all failing instances.
    """
    records = []

    for qid, method in failing_pairs:
        q_meta = questions_dict[qid]
        gold_titles = qrels[qid]
        method_ranked_list = all_rankings[method][qid]

        # Map document positions (1-based) across full ranking
        full_doc_order = [doc_id for doc_id, _ in method_ranked_list]
        doc_position_lookup = {doc_id: pos + 1 for pos, doc_id in enumerate(full_doc_order)}

        gold_ranks = {
            title: doc_position_lookup.get(title, None)
            for title in gold_titles
        }

        top_10 = [
            {"doc_id": doc_id, "score": float(score), "rank": rank + 1}
            for rank, (doc_id, score) in enumerate(method_ranked_list[:10])
        ]

        record = {
            "question_id": qid,
            "question": q_meta["question"],
            "type": q_meta["type"],
            "method": method,
            "gold_doc_ids": gold_titles,
            "top_10_retrieved": top_10,
            "gold_ranks": gold_ranks,
        }
        records.append(record)

    return records


# -----------------------------------------------------------------------------
# STEP 3: Taxonomy Categorization Logic
# -----------------------------------------------------------------------------

TAXONOMY_DEFINITIONS = {
    "multi_hop_bridging_failure": (
        "The query requires reasoning across multiple documents, but one of the gold documents "
        "(typically the second hop) is not mentioned and cannot be semantically matched from the query "
        "text alone without first retrieving and reading the bridge document."
    ),
    "topical_distraction_and_semantic_drift": (
        "The corpus contains multiple competing documents in the same broad topical domain, genre, or "
        "series (e.g. other Olympic competitions, other musical albums or songs by the same artist, or "
        "same-category entities) that outrank the true gold target."
    ),
    "lexical_and_syntactic_mismatch": (
        "The query uses vocabulary, paraphrasing, phonetic descriptions, or informal formulations (including "
        "spelling errors or descriptive aliases) that fail to match the formal lexical terms in the gold document, "
        "heavily penalizing exact keyword matching."
    ),
    "entity_confusion_and_partial_overlap": (
        "A distractor document contains partial or verbatim overlap with entity names or keywords in the query "
        "(such as shared namesake substrings, adaptations across media, or namesake locations) and outranks "
        "the specific target entity document."
    ),
}

# Empirical assignment based on detailed qualitative analysis of all 42 failure records
TAXONOMY_ASSIGNMENTS = {
    # 1. Multi-hop bridging failure
    "5abc19705542993a06baf86e__bm25": "multi_hop_bridging_failure",
    "5abc19705542993a06baf86e__hybrid": "multi_hop_bridging_failure",
    "5a7344e95542991f9a20c6ce__dense": "multi_hop_bridging_failure",
    "5a7344e95542991f9a20c6ce__hybrid": "multi_hop_bridging_failure",
    "5a7344e95542991f9a20c6ce__bm25": "multi_hop_bridging_failure",
    "5a8f503c5542992414482a34__bm25": "multi_hop_bridging_failure",
    "5a721a7655429971e9dc9271__bm25": "multi_hop_bridging_failure",
    "5a8cfa2e554299585d9e378b__hybrid": "multi_hop_bridging_failure",
    "5a8cfa2e554299585d9e378b__bm25": "multi_hop_bridging_failure",
    "5a89bbb05542992e4fca83a3__bm25": "multi_hop_bridging_failure",
    "5ab4304a55429942dd415ec5__bm25": "multi_hop_bridging_failure",
    "5abc030e554299642a094bdc__bm25": "multi_hop_bridging_failure",
    "5ac2d85e55429921a00ab06b__bm25": "multi_hop_bridging_failure",
    "5a8dee2455429917b4a5bce1__bm25": "multi_hop_bridging_failure",

    # 2. Topical distraction and semantic drift
    "5a7f54665542992097ad2f1a__dense": "topical_distraction_and_semantic_drift",
    "5a809e9f5542996402f6a5b1__dense": "topical_distraction_and_semantic_drift",
    "5a87c13f5542996e4f30890c__dense": "topical_distraction_and_semantic_drift",
    "5ae306115542992decbdcdc4__dense": "topical_distraction_and_semantic_drift",
    "5ab5c263554299488d4d9a18__bm25": "topical_distraction_and_semantic_drift",
    "5a7e32905542991319bc943b__bm25": "topical_distraction_and_semantic_drift",
    "5ae497f15542995ad6573db8__hybrid": "topical_distraction_and_semantic_drift",
    "5ae497f15542995ad6573db8__bm25": "topical_distraction_and_semantic_drift",
    "5ab9b7d555429970cfb8eb7a__bm25": "topical_distraction_and_semantic_drift",
    "5a7f567e5542992097ad2f22__bm25": "topical_distraction_and_semantic_drift",

    # 3. Lexical and syntactic mismatch
    "5ae7cea355429952e35ea9c1__dense": "lexical_and_syntactic_mismatch",
    "5ae7cea355429952e35ea9c1__hybrid": "lexical_and_syntactic_mismatch",
    "5adbcc085542996e6852523c__bm25": "lexical_and_syntactic_mismatch",
    "5ab31864554299233954ff06__bm25": "lexical_and_syntactic_mismatch",
    "5a851ba95542997175ce1f81__bm25": "lexical_and_syntactic_mismatch",
    "5a8effb55542997ba9cb317f__bm25": "lexical_and_syntactic_mismatch",

    # 4. Entity confusion and partial overlap
    "5adde73f5542992200553b94__hybrid": "entity_confusion_and_partial_overlap",
    "5adde73f5542992200553b94__bm25": "entity_confusion_and_partial_overlap",
    "5a8c493e554299653c1aa020__hybrid": "entity_confusion_and_partial_overlap",
    "5a8c493e554299653c1aa020__bm25": "entity_confusion_and_partial_overlap",
    "5ae32e125542991a06ce9946__hybrid": "entity_confusion_and_partial_overlap",
    "5ae32e125542991a06ce9946__bm25": "entity_confusion_and_partial_overlap",
    "5ac3e0f7554299194317388b__bm25": "entity_confusion_and_partial_overlap",
    "5a88b3b4554299206df2b336__bm25": "entity_confusion_and_partial_overlap",
    "5a840e395542992ef85e239d__bm25": "entity_confusion_and_partial_overlap",
    "5a7aa0a55542990198eaf165__bm25": "entity_confusion_and_partial_overlap",
    "5a81711455429938b614233e__bm25": "entity_confusion_and_partial_overlap",
    "5a865e8a55429960ec39b67a__bm25": "entity_confusion_and_partial_overlap",
}


# -----------------------------------------------------------------------------
# STEP 4: Build Taxonomy Mapping and Summary
# -----------------------------------------------------------------------------

def build_taxonomy_artifacts(
    failure_records: list[dict],
) -> tuple[dict[str, str], dict]:
    """
    Produce the mapping and statistical summary.
    """
    mapping = {}
    cat_counts = Counter()
    cat_methods = defaultdict(set)

    for rec in failure_records:
        key = f"{rec['question_id']}__{rec['method']}"
        if key not in TAXONOMY_ASSIGNMENTS:
            raise ValueError(f"Missing taxonomy assignment for {key}")
        cat = TAXONOMY_ASSIGNMENTS[key]
        mapping[key] = cat
        cat_counts[cat] += 1
        cat_methods[cat].add(rec["method"])

    total_cases = len(failure_records)
    summary = {}

    for cat_name, definition in TAXONOMY_DEFINITIONS.items():
        count = cat_counts[cat_name]
        pct = round((count / total_cases) * 100, 2) if total_cases > 0 else 0.0
        summary[cat_name] = {
            "count": count,
            "percentage_of_failures": pct,
            "methods_affected": sorted(list(cat_methods[cat_name])),
            "definition": definition,
        }

    return mapping, summary


# -----------------------------------------------------------------------------
# Sanity Checks
# -----------------------------------------------------------------------------

def run_sanity_checks(
    failure_records: list[dict],
    questions_dict: dict[str, dict],
    corpus_titles: set[str],
    expected_count: int,
    mapping: dict[str, str],
) -> bool:
    """
    Validate all constraints specified in prompt.
    """
    print("\nRunning sanity checks on failure analysis artifacts:")
    all_ok = True

    def check(ok: bool, msg: str):
        nonlocal all_ok
        sym = "✓" if ok else "✗"
        print(f"  {sym} {msg}")
        if not ok:
            all_ok = False

    # 1. Row count matches expected count
    check(len(failure_records) == expected_count, f"total row count ({len(failure_records)}) matches expected ({expected_count})")

    # 2. Every question_id exists in sampled_questions
    unknown_qids = [r["question_id"] for r in failure_records if r["question_id"] not in questions_dict]
    check(len(unknown_qids) == 0, f"all question_ids exist in sampled_questions.json (missing: {len(unknown_qids)})")

    # 3. Every gold title exists in corpus
    missing_golds = []
    for r in failure_records:
        for gold in r["gold_doc_ids"]:
            if gold not in corpus_titles:
                missing_golds.append(gold)
    check(len(missing_golds) == 0, f"all gold document titles exist in corpus.jsonl (missing: {len(missing_golds)})")

    # 4. No duplicate (question_id, method) pairs
    keys = [f"{r['question_id']}__{r['method']}" for r in failure_records]
    check(len(keys) == len(set(keys)), f"no duplicate (question_id, method) pairs ({len(keys)} unique)")

    # 5. Mapping covers 100% of rows
    unmapped = [k for k in keys if k not in mapping]
    check(len(unmapped) == 0, f"taxonomy mapping covers 100% of failure cases ({len(mapping)} mapped)")

    return all_ok


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    cfg = load_config()

    print("Loading artifacts and rankings...")
    corpus_docs = load_corpus(cfg)
    corpus_titles = {d["doc_id"] for d in corpus_docs}
    questions = load_questions(cfg)
    questions_dict = {q["_id"]: q for q in questions}
    qrels = load_qrels(cfg)

    all_rankings = {
        "bm25": load_rankings(resolve_path(cfg["output_paths"]["bm25_rankings"])),
        "dense": load_rankings(resolve_path(cfg["output_paths"]["dense_rankings"])),
        "hybrid": load_rankings(resolve_path(cfg["output_paths"]["hybrid_rankings"])),
    }

    # Step 1: Predefined Threshold Evaluation
    print("\n--- STEP 1: Predefined Threshold Evaluation ---")
    chosen_k, failing_pairs, unique_qids = evaluate_thresholds(
        qrels,
        all_rankings["bm25"],
        all_rankings["dense"],
        all_rankings["hybrid"],
        min_unique_queries=20,
    )

    print(f"\nFinal Selection: Threshold Recall@{chosen_k}=0 chosen.")
    print(f"Total failure instances: {len(failing_pairs)} across {len(unique_qids)} unique queries.")

    # Step 2: Extract Raw Failure Case Data
    print("\n--- STEP 2: Extract Raw Failure Case Data ---")
    failure_records = extract_failure_cases(failing_pairs, questions_dict, qrels, all_rankings)

    # Step 3 & 4: Taxonomy Mapping & Summary
    print("\n--- STEP 3 & 4: Categorize and Generate Taxonomy Artifacts ---")
    mapping, summary = build_taxonomy_artifacts(failure_records)

    # Sanity checks
    passed = run_sanity_checks(failure_records, questions_dict, corpus_titles, len(failing_pairs), mapping)
    if not passed:
        print("\n✗ Sanity checks FAILED. Aborting output writing.")
        sys.exit(1)

    # Save artifacts
    raw_path = resolve_path("results/failure_cases_raw.jsonl")
    mapping_path = resolve_path("results/failure_taxonomy_mapping.json")
    summary_path = resolve_path("results/failure_taxonomy_summary.json")

    check_file_overwrite(raw_path, "failure_analysis")
    ensure_dir(raw_path.parent)

    with open(raw_path, "w", encoding="utf-8") as f:
        for rec in failure_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  -> Saved {len(failure_records)} raw failure records to {raw_path.name}")

    check_file_overwrite(mapping_path, "failure_analysis")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved taxonomy mapping to {mapping_path.name}")

    check_file_overwrite(summary_path, "failure_analysis")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved taxonomy summary to {summary_path.name}")

    print("\n=== FAILURE TAXONOMY SUMMARY ===")
    for cat, info in summary.items():
        print(f"  {cat:<42}: {info['count']:>2} cases ({info['percentage_of_failures']:>5.1f}%) | Methods: {', '.join(info['methods_affected'])}")


if __name__ == "__main__":
    main()
