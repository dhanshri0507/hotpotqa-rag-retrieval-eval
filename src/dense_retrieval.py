#!/usr/bin/env python3
"""
dense_retrieval.py — Dense retrieval with sentence-transformers + FAISS.

Embeds corpus (cached) and queries, retrieves via exact inner-product search.
Run from project root:  python -m src.dense_retrieval
"""

import json
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import (
    load_config,
    load_corpus,
    load_questions,
    save_rankings,
    resolve_path,
    check_file_overwrite,
    ensure_dir,
)


def load_or_build_corpus_embeddings(
    model: SentenceTransformer, corpus_docs: list[dict], cfg: dict
) -> tuple[np.ndarray, list[str]]:
    """
    Embed corpus documents, caching to disk so re-runs don't recompute.

    Cache files:
        embeddings/corpus_embeddings.npy
        embeddings/doc_ids.json

    Validates cache by checking that saved doc_ids match the current corpus
    (same list, same order). Recomputes on mismatch.
    """
    doc_ids = [doc["doc_id"] for doc in corpus_docs]
    texts = [doc["text"] for doc in corpus_docs]

    emb_path = resolve_path(cfg["embedding_paths"]["corpus_embeddings"])
    ids_path = resolve_path(cfg["embedding_paths"]["doc_ids"])

    # Try loading cache
    if emb_path.exists() and ids_path.exists():
        print(f"  Found cached embeddings at {emb_path.name}")
        with open(ids_path, "r", encoding="utf-8") as f:
            cached_ids = json.load(f)
        if cached_ids == doc_ids:
            embeddings = np.load(str(emb_path))
            print(f"  Cache valid — loaded {embeddings.shape[0]} embeddings")
            return embeddings, doc_ids
        else:
            print("  WARNING: cached doc_ids don't match current corpus. Recomputing ...")

    # Compute embeddings
    print(f"  Encoding {len(texts)} documents (batch_size=64) ...")
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Save cache
    ensure_dir(emb_path.parent)
    np.save(str(emb_path), embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False)
    print(f"  Cached embeddings -> {emb_path.name}, doc_ids -> {ids_path.name}")

    return embeddings, doc_ids


def embed_queries(
    model: SentenceTransformer, questions: list[dict]
) -> tuple[np.ndarray, list[str]]:
    """Embed all queries with the same model and normalization."""
    texts = [q["question"] for q in questions]
    qids = [q["_id"] for q in questions]

    print(f"  Encoding {len(texts)} queries ...")
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings, qids


def retrieve_all(
    corpus_embeddings: np.ndarray,
    doc_ids: list[str],
    query_embeddings: np.ndarray,
    query_ids: list[str],
) -> dict[str, list[tuple[str, float]]]:
    """
    Build FAISS IndexFlatIP, search all docs for every query.

    Returns full ranked lists: {qid: [(doc_id, score), ...]} sorted desc.
    """
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")

    k = len(doc_ids)  # retrieve ALL documents (full ranking)
    scores_matrix, indices_matrix = index.search(query_embeddings, k)

    rankings = {}
    for i, qid in enumerate(query_ids):
        ranked_list = []
        for j in range(k):
            idx = int(indices_matrix[i, j])
            score = float(scores_matrix[i, j])
            ranked_list.append((doc_ids[idx], score))
        rankings[qid] = ranked_list

    return rankings


def run_sanity_checks(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    corpus_docs: list[dict],
    questions: list[dict],
    doc_ids: list[str],
    rankings: dict,
    cfg: dict,
) -> bool:
    """
    Sanity checks:
      a. Corpus embedding count == corpus doc count
      b. Query embedding count == 300
      c. Embedding dimension == 384
      d. Every ranked doc_id maps to a real corpus doc_id
      e. Ranked lists sorted descending by score
    """
    all_ok = True
    expected_dim = cfg["embedding_dim"]

    def check(ok: bool, msg: str) -> None:
        nonlocal all_ok
        sym = "✓" if ok else "✗"
        print(f"  {sym} {msg}")
        if not ok:
            all_ok = False

    # (a) Corpus embeddings count
    check(
        corpus_embeddings.shape[0] == len(corpus_docs),
        f"corpus embeddings count = {corpus_embeddings.shape[0]} == {len(corpus_docs)} docs",
    )

    # (b) Query embeddings count
    check(
        query_embeddings.shape[0] == len(questions),
        f"query embeddings count = {query_embeddings.shape[0]} == {len(questions)} questions",
    )

    # (c) Embedding dimension
    check(
        corpus_embeddings.shape[1] == expected_dim
        and query_embeddings.shape[1] == expected_dim,
        f"embedding dim = {expected_dim}",
    )

    # (d) Every ranked doc_id is valid
    valid_ids = set(doc_ids)
    invalid = False
    for qid, ranked_list in rankings.items():
        for did, _ in ranked_list:
            if did not in valid_ids:
                print(f"    invalid doc_id '{did}' in query {qid}")
                invalid = True
                break
        if invalid:
            break
    check(not invalid, "all ranked doc_ids map to real corpus documents")

    # (e) Sorted descending
    unsorted = False
    for qid, ranked_list in rankings.items():
        scores = [s for _, s in ranked_list]
        if not all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)):
            unsorted = True
            break
    check(not unsorted, "all ranked lists sorted descending by score")

    return all_ok


def main() -> None:
    cfg = load_config()

    corpus_docs = load_corpus(cfg)
    questions = load_questions(cfg)
    print(f"Loaded {len(corpus_docs)} corpus documents, {len(questions)} questions")

    model_name = cfg["embedding_model"]
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("\nCorpus embeddings:")
    corpus_embeddings, doc_ids = load_or_build_corpus_embeddings(model, corpus_docs, cfg)

    print("\nQuery embeddings:")
    query_embeddings, query_ids = embed_queries(model, questions)

    print("\nRetrieval:")
    rankings = retrieve_all(corpus_embeddings, doc_ids, query_embeddings, query_ids)

    print("\nSanity checks:")
    if not run_sanity_checks(
        corpus_embeddings, query_embeddings, corpus_docs, questions, doc_ids, rankings, cfg
    ):
        print("\n✗ Sanity checks FAILED. Aborting.")
        sys.exit(1)

    out_path = resolve_path(cfg["output_paths"]["dense_rankings"])
    check_file_overwrite(out_path, "dense_retrieval")
    save_rankings(rankings, out_path)
    print(
        f"\nDense retrieval complete. Saved {len(rankings)} queries "
        f"to {out_path.name}"
    )


if __name__ == "__main__":
    main()
