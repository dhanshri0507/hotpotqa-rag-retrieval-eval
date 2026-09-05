#!/usr/bin/env python3
"""
bm25_retrieval.py — BM25 sparse retrieval over the frozen HotpotQA dataset.

Run from project root:  python -m src.bm25_retrieval
"""

import re
import sys

from rank_bm25 import BM25Okapi

from src.utils import (
    load_config,
    load_corpus,
    load_questions,
    save_rankings,
    resolve_path,
    check_file_overwrite,
)


def tokenize(text: str) -> list[str]:
    """Lowercase, then split on word characters only. No stemming, no stopwords."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(
    corpus_docs: list[dict], k1: float, b: float
) -> tuple[BM25Okapi, list[str]]:
    """
    Tokenize every corpus document and build a BM25Okapi index.

    Returns:
        (bm25_index, doc_ids) where doc_ids[i] corresponds to the i-th doc.
    """
    tokenized_corpus = [tokenize(doc["text"]) for doc in corpus_docs]
    doc_ids = [doc["doc_id"] for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    return bm25, doc_ids


def retrieve_all(
    bm25: BM25Okapi, doc_ids: list[str], questions: list[dict]
) -> dict[str, list[tuple[str, float]]]:
    """
    For each question, score all documents and return a full ranked list.

    Returns:
        {question_id: [(doc_id, score), ...]} sorted descending by score.
    """
    rankings = {}
    for i, q in enumerate(questions):
        qid = q["_id"]
        tokenized_query = tokenize(q["question"])
        scores = bm25.get_scores(tokenized_query)
        # Pair each doc_id with its score, sort descending
        ranked_list = sorted(
            zip(doc_ids, scores.tolist()), key=lambda x: x[1], reverse=True
        )
        rankings[qid] = ranked_list
        if (i + 1) % 50 == 0:
            print(f"  Scored {i + 1}/{len(questions)} queries")
    return rankings


def run_sanity_checks(
    rankings: dict, questions: list[dict], corpus_docs: list[dict]
) -> bool:
    """
    Sanity checks:
      a. Every query has a ranked list
      b. Every doc_id in rankings exists in corpus
      c. No duplicate doc_ids within any query's ranked list
      d. Every ranked list is sorted descending by score
    """
    corpus_doc_ids = {doc["doc_id"] for doc in corpus_docs}
    all_ok = True

    def check(ok: bool, msg: str) -> None:
        nonlocal all_ok
        sym = "✓" if ok else "✗"
        print(f"  {sym} {msg}")
        if not ok:
            all_ok = False

    # (a) Every query has a ranked list
    question_ids = {q["_id"] for q in questions}
    missing = question_ids - set(rankings.keys())
    check(
        len(rankings) == len(questions) and len(missing) == 0,
        f"every query has a ranked list ({len(rankings)}/{len(questions)})",
    )

    # (b) Every doc_id in rankings exists in corpus
    invalid_docs = False
    for qid, ranked_list in rankings.items():
        for doc_id, _ in ranked_list:
            if doc_id not in corpus_doc_ids:
                invalid_docs = True
                break
        if invalid_docs:
            break
    check(not invalid_docs, "all doc_ids exist in corpus")

    # (c) No duplicate doc_ids within any query's ranked list
    dup_found = False
    for qid, ranked_list in rankings.items():
        ids = [did for did, _ in ranked_list]
        if len(ids) != len(set(ids)):
            dup_found = True
            print(f"    duplicate doc_ids in query {qid}")
            break
    check(not dup_found, "no duplicate doc_ids in any ranked list")

    # (d) Every ranked list is sorted descending by score
    unsorted = False
    for qid, ranked_list in rankings.items():
        scores = [s for _, s in ranked_list]
        if not all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)):
            unsorted = True
            print(f"    unsorted ranked list for query {qid}")
            break
    check(not unsorted, "all ranked lists sorted descending by score")

    return all_ok


def main() -> None:
    cfg = load_config()

    corpus_docs = load_corpus(cfg)
    questions = load_questions(cfg)
    print(f"Loaded {len(corpus_docs)} corpus documents, {len(questions)} questions")

    k1 = cfg["bm25"]["k1"]
    b = cfg["bm25"]["b"]
    print(f"Building BM25 index (k1={k1}, b={b}) ...")
    bm25, doc_ids = build_bm25_index(corpus_docs, k1, b)

    print("Retrieving ranked lists for all queries ...")
    rankings = retrieve_all(bm25, doc_ids, questions)

    print("\nSanity checks:")
    if not run_sanity_checks(rankings, questions, corpus_docs):
        print("\n✗ Sanity checks FAILED. Aborting.")
        sys.exit(1)

    out_path = resolve_path(cfg["output_paths"]["bm25_rankings"])
    check_file_overwrite(out_path, "bm25_retrieval")
    save_rankings(rankings, out_path)
    print(
        f"\nBM25 retrieval complete. Saved {len(rankings)} queries "
        f"to {out_path.name}"
    )


if __name__ == "__main__":
    main()
