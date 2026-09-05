#!/usr/bin/env python3
"""
hybrid_retrieval.py — Reciprocal Rank Fusion (RRF) of BM25 and Dense rankings.

Reads the saved BM25 and Dense ranking files (never recomputes them),
fuses them via RRF, and saves a combined ranking.
"""

import sys
from pathlib import Path

from src.utils import (
    load_config,
    load_rankings,
    save_rankings,
    resolve_path,
    check_file_overwrite,
)


def reciprocal_rank_fusion(
    bm25_rankings: dict[str, list[tuple[str, float]]],
    dense_rankings: dict[str, list[tuple[str, float]]],
    top_k_for_fusion: int,
    rrf_k: int,
) -> dict[str, list[tuple[str, float]]]:
    """
    Apply RRF to merge BM25 and Dense ranked lists.

    For each document d:
        score(d) = sum over methods of 1 / (rrf_k + rank_in_that_method)

    Only the top_k_for_fusion entries from each method are used.

    Args:
        bm25_rankings: {qid: [(doc_id, score), ...]} sorted desc by score.
        dense_rankings: same format.
        top_k_for_fusion: how many top results to take from each method.
        rrf_k: the RRF constant (typically 60).

    Returns:
        {qid: [(doc_id, fused_score), ...]} sorted desc by fused score.
    """
    # Get all question IDs from both sources
    all_qids = set(bm25_rankings.keys()) | set(dense_rankings.keys())
    fused = {}

    for qid in all_qids:
        doc_scores: dict[str, float] = {}

        # BM25 contribution
        bm25_list = bm25_rankings.get(qid, [])[:top_k_for_fusion]
        for rank_0based, (doc_id, _score) in enumerate(bm25_list):
            rank = rank_0based + 1  # 1-based rank
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        # Dense contribution
        dense_list = dense_rankings.get(qid, [])[:top_k_for_fusion]
        for rank_0based, (doc_id, _score) in enumerate(dense_list):
            rank = rank_0based + 1
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        # Sort by fused score descending
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused[qid] = [(doc_id, score) for doc_id, score in ranked]

    return fused


def run_sanity_checks(
    fused_rankings: dict,
    bm25_rankings: dict,
    dense_rankings: dict,
) -> bool:
    """
    Sanity checks for hybrid rankings:
      a. Every query has a fused ranking
      b. No duplicate doc_ids in any query's fused ranking
      c. All doc_ids in fused ranking come from BM25 or Dense input rankings
    """
    all_ok = True

    def check(ok: bool, msg: str) -> None:
        nonlocal all_ok
        sym = "✓" if ok else "✗"
        print(f"  {sym} {msg}")
        if not ok:
            all_ok = False

    # (a) Every query from both inputs has a fused ranking
    all_input_qids = set(bm25_rankings.keys()) | set(dense_rankings.keys())
    missing = all_input_qids - set(fused_rankings.keys())
    check(len(missing) == 0, f"every query has a fused ranking ({len(fused_rankings)} queries)")

    # (b) No duplicate doc_ids within any query's fused ranking
    dup_found = False
    for qid, ranked_list in fused_rankings.items():
        doc_ids = [did for did, _ in ranked_list]
        if len(doc_ids) != len(set(doc_ids)):
            print(f"    duplicate doc_ids found for query {qid}")
            dup_found = True
    check(not dup_found, "no duplicate doc_ids in any fused ranking")

    # (c) All fused doc_ids come from at least one input ranking
    invalid = False
    for qid, ranked_list in fused_rankings.items():
        bm25_doc_ids = {did for did, _ in bm25_rankings.get(qid, [])}
        dense_doc_ids = {did for did, _ in dense_rankings.get(qid, [])}
        valid_ids = bm25_doc_ids | dense_doc_ids
        for doc_id, _ in ranked_list:
            if doc_id not in valid_ids:
                print(f"    fused doc_id '{doc_id}' not in either input for query {qid}")
                invalid = True
    check(not invalid, "all fused doc_ids come from input rankings")

    return all_ok


def main() -> None:
    cfg = load_config()

    # Paths from config
    bm25_path = resolve_path(cfg["output_paths"]["bm25_rankings"])
    dense_path = resolve_path(cfg["output_paths"]["dense_rankings"])
    output_path = resolve_path(cfg["output_paths"]["hybrid_rankings"])

    top_k = cfg["top_k_for_fusion"]
    rrf_k = cfg["rrf_k"]

    print(f"Loading BM25 rankings from {bm25_path.name} ...")
    bm25_rankings = load_rankings(bm25_path)
    print(f"  {len(bm25_rankings)} queries loaded")

    print(f"Loading Dense rankings from {dense_path.name} ...")
    dense_rankings = load_rankings(dense_path)
    print(f"  {len(dense_rankings)} queries loaded")

    print(f"\nApplying RRF (k={rrf_k}, top_k_for_fusion={top_k}) ...")
    fused_rankings = reciprocal_rank_fusion(
        bm25_rankings, dense_rankings, top_k, rrf_k
    )
    print(f"  Fused rankings for {len(fused_rankings)} queries")

    # Sanity checks
    print("\nSanity checks:")
    passed = run_sanity_checks(fused_rankings, bm25_rankings, dense_rankings)
    if not passed:
        print("\n✗ Sanity checks FAILED. Aborting.")
        sys.exit(1)

    # Save
    check_file_overwrite(output_path, "hybrid_retrieval")
    save_rankings(fused_rankings, output_path)
    print(f"\nHybrid retrieval complete. Saved {len(fused_rankings)} queries to {output_path.name}")


if __name__ == "__main__":
    main()
