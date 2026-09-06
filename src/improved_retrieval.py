#!/usr/bin/env python3
"""
improved_retrieval.py — Two-step iterative Hybrid retrieval with query expansion.

Pipeline:
  1. Hop 1: Load precomputed baseline Hybrid rankings (results/hybrid_rankings.jsonl).
  2. Query expansion: Take top hop1_top_n documents from Hop 1 ranking, fetch full text
     from corpus.jsonl, and concatenate with original query text.
  3. Hop 2:
     - Run BM25 retrieval on expanded queries.
     - Run Dense retrieval on expanded queries (sentence-transformers + FAISS IndexFlatIP).
     - Fuse hop-2 BM25 and Dense rankings via RRF (k=60) -> hop-2 Hybrid ranking.
  4. Final ranking:
     - Fuse hop-1 Hybrid ranking and hop-2 Hybrid ranking via RRF (k=60).
  5. Validate with sanity checks and save to results/improved_rankings.jsonl.

Run from project root:  python -m src.improved_retrieval
"""

import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.utils import (
    load_config,
    load_corpus,
    load_questions,
    load_rankings,
    save_rankings,
    resolve_path,
    check_file_overwrite,
    ensure_dir,
)


def tokenize(text: str) -> list[str]:
    """Lowercase, then split on word characters only (same as BM25 baseline)."""
    return re.findall(r"\w+", text.lower())


def build_expanded_queries(
    questions: list[dict],
    hop1_rankings: dict[str, list[tuple[str, float]]],
    corpus_dict: dict[str, dict],
    top_n: int,
) -> dict[str, str]:
    """
    Build expanded query strings by concatenating the original question with the
    full paragraph text of the top_n documents from hop-1 Hybrid ranking.
    """
    expanded_queries = {}
    for q in questions:
        qid = q["_id"]
        original_q = q["question"]
        hop1_ranked = hop1_rankings.get(qid, [])
        top_docs = hop1_ranked[:top_n]

        context_pieces = []
        for did, _score in top_docs:
            if did in corpus_dict:
                context_pieces.append(corpus_dict[did]["text"])

        if context_pieces:
            expanded_text = original_q + " " + " ".join(context_pieces)
        else:
            expanded_text = original_q

        expanded_queries[qid] = expanded_text

    return expanded_queries


def retrieve_bm25_hop2(
    bm25: BM25Okapi,
    doc_ids: list[str],
    questions: list[dict],
    expanded_queries: dict[str, str],
) -> dict[str, list[tuple[str, float]]]:
    """Score all corpus documents with BM25 using expanded queries."""
    rankings = {}
    for i, q in enumerate(questions):
        qid = q["_id"]
        tokenized_q = tokenize(expanded_queries[qid])
        scores = bm25.get_scores(tokenized_q)
        ranked_list = sorted(
            zip(doc_ids, scores.tolist()), key=lambda x: x[1], reverse=True
        )
        rankings[qid] = ranked_list
        if (i + 1) % 100 == 0:
            print(f"    BM25 hop-2 scored {i + 1}/{len(questions)} queries")
    return rankings


def retrieve_dense_hop2(
    model: SentenceTransformer,
    corpus_embeddings: np.ndarray,
    doc_ids: list[str],
    questions: list[dict],
    expanded_queries: dict[str, str],
) -> dict[str, list[tuple[str, float]]]:
    """Retrieve full rankings via FAISS IndexFlatIP using encoded expanded queries."""
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embeddings)

    texts = [expanded_queries[q["_id"]] for q in questions]
    qids = [q["_id"] for q in questions]

    print(f"    Encoding {len(texts)} expanded queries (batch_size=64) ...")
    query_embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

    k = len(doc_ids)
    scores_matrix, indices_matrix = index.search(query_embeddings, k)

    rankings = {}
    for i, qid in enumerate(qids):
        ranked_list = []
        for j in range(k):
            idx = int(indices_matrix[i, j])
            score = float(scores_matrix[i, j])
            ranked_list.append((doc_ids[idx], score))
        rankings[qid] = ranked_list

    return rankings


def fuse_rankings_rrf(
    ranking_a: dict[str, list[tuple[str, float]]],
    ranking_b: dict[str, list[tuple[str, float]]],
    top_k_for_fusion: int,
    rrf_k: int,
) -> dict[str, list[tuple[str, float]]]:
    """Fuse two ranked lists using Reciprocal Rank Fusion."""
    all_qids = set(ranking_a.keys()) | set(ranking_b.keys())
    fused = {}

    for qid in all_qids:
        doc_scores: dict[str, float] = {}

        list_a = ranking_a.get(qid, [])[:top_k_for_fusion]
        for rank_0based, (did, _) in enumerate(list_a):
            rank = rank_0based + 1
            doc_scores[did] = doc_scores.get(did, 0.0) + 1.0 / (rrf_k + rank)

        list_b = ranking_b.get(qid, [])[:top_k_for_fusion]
        for rank_0based, (did, _) in enumerate(list_b):
            rank = rank_0based + 1
            doc_scores[did] = doc_scores.get(did, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused[qid] = [(did, score) for did, score in ranked]

    return fused


def run_sanity_checks(
    improved_rankings: dict[str, list[tuple[str, float]]],
    questions: list[dict],
    expanded_queries: dict[str, str],
) -> bool:
    """
    Sanity checks:
      1. Every query has a final improved ranking (300 queries)
      2. No duplicate doc_ids within any query's improved ranking
      3. Improved ranking is sorted descending by fused score
      4. Expanded query text is non-empty for every query
    """
    all_ok = True

    def check(ok: bool, msg: str) -> None:
        nonlocal all_ok
        sym = "✓" if ok else "✗"
        print(f"  {sym} {msg}")
        if not ok:
            all_ok = False

    # 1. Query count
    check(
        len(improved_rankings) == len(questions),
        f"every query has an improved ranking ({len(improved_rankings)}/{len(questions)})",
    )

    # 2. No duplicate doc_ids
    dups_found = False
    for qid, ranked_list in improved_rankings.items():
        dids = [d for d, _ in ranked_list]
        if len(dids) != len(set(dids)):
            print(f"    duplicate doc_ids in query {qid}")
            dups_found = True
            break
    check(not dups_found, "no duplicate doc_ids in any query's improved ranking")

    # 3. Sorted descending
    unsorted = False
    for qid, ranked_list in improved_rankings.items():
        scores = [s for _, s in ranked_list]
        if not all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)):
            print(f"    unsorted ranking in query {qid}")
            unsorted = True
            break
    check(not unsorted, "improved rankings sorted descending by fused score")

    # 4. Expanded queries non-empty
    empty_expanded = [qid for qid, text in expanded_queries.items() if not text.strip()]
    check(
        len(empty_expanded) == 0,
        f"expanded query text non-empty for all queries ({len(expanded_queries)} non-empty)",
    )

    return all_ok


def main() -> None:
    cfg = load_config()

    corpus_docs = load_corpus(cfg)
    corpus_dict = {d["doc_id"]: d for d in corpus_docs}
    doc_ids = [d["doc_id"] for d in corpus_docs]
    questions = load_questions(cfg)

    print(f"Loaded {len(corpus_docs)} corpus documents and {len(questions)} questions.")

    # 1. Hop 1: Load precomputed baseline Hybrid rankings
    hop1_path = resolve_path(cfg["output_paths"]["hybrid_rankings"])
    print(f"\nStep 1: Loading Hop 1 Hybrid rankings from {hop1_path.name} ...")
    hop1_rankings = load_rankings(hop1_path)
    print(f"  Loaded {len(hop1_rankings)} queries.")

    # 2. Query expansion
    hop1_top_n = cfg["improvement"]["hop1_top_n"]
    print(f"\nStep 2: Building expanded queries using top-{hop1_top_n} Hop 1 documents ...")
    expanded_queries = build_expanded_queries(questions, hop1_rankings, corpus_dict, hop1_top_n)
    sample_qid = questions[0]["_id"]
    print(f"  Sample original query: {questions[0]['question']}")
    print(f"  Sample expanded length: {len(expanded_queries[sample_qid])} chars")

    # 3. Hop 2 Retrieval
    print("\nStep 3: Running Hop 2 retrieval on expanded queries ...")
    # BM25 Hop 2
    k1 = cfg["bm25"]["k1"]
    b = cfg["bm25"]["b"]
    print(f"  Building BM25 index (k1={k1}, b={b}) ...")
    tokenized_corpus = [tokenize(d["text"]) for d in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    print("  Scoring expanded queries with BM25 ...")
    hop2_bm25 = retrieve_bm25_hop2(bm25, doc_ids, questions, expanded_queries)

    # Dense Hop 2
    emb_path = resolve_path(cfg["embedding_paths"]["corpus_embeddings"])
    print(f"  Loading cached corpus embeddings from {emb_path.name} ...")
    corpus_embeddings = np.load(str(emb_path))
    model_name = cfg["embedding_model"]
    print(f"  Loading embedding model {model_name} ...")
    model = SentenceTransformer(model_name)
    print("  Scoring expanded queries with Dense retrieval ...")
    hop2_dense = retrieve_dense_hop2(model, corpus_embeddings, doc_ids, questions, expanded_queries)

    # Hop 2 Fusion
    top_k = cfg["top_k_for_fusion"]
    rrf_k = cfg["rrf_k"]
    print(f"  Fusing Hop 2 (BM25 + Dense) via RRF (k={rrf_k}, top_k={top_k}) ...")
    hop2_hybrid = fuse_rankings_rrf(hop2_bm25, hop2_dense, top_k, rrf_k)

    # 4. Final Ranking: Fuse Hop 1 Hybrid + Hop 2 Hybrid
    print(f"\nStep 4: Fusing Hop 1 Hybrid + Hop 2 Hybrid via RRF (k={rrf_k}) ...")
    improved_rankings = fuse_rankings_rrf(hop1_rankings, hop2_hybrid, top_k, rrf_k)

    # 5. Sanity checks
    print("\nStep 5: Sanity checks ...")
    if not run_sanity_checks(improved_rankings, questions, expanded_queries):
        print("\n✗ Sanity checks FAILED. Aborting.")
        sys.exit(1)

    # Save final improved rankings
    out_path = resolve_path(cfg["output_paths"]["improved_rankings"])
    check_file_overwrite(out_path, "improved_retrieval")
    save_rankings(improved_rankings, out_path)
    print(f"\nImproved retrieval complete. Saved {len(improved_rankings)} queries to {out_path.name}")


if __name__ == "__main__":
    main()
