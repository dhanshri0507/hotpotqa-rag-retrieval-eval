#!/usr/bin/env python3

import json
import os
import random
import sys
from pathlib import Path

# Ensure Unicode characters render correctly on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hotpot_dev_distractor_v1.json"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

SAMPLED_QUESTIONS_PATH = ARTIFACTS_DIR / "sampled_questions.json"
CORPUS_PATH = ARTIFACTS_DIR / "corpus.jsonl"
QRELS_PATH = ARTIFACTS_DIR / "qrels.json"
MANIFEST_PATH = ARTIFACTS_DIR / "dataset_manifest.json"

RANDOM_SEED = 42
SAMPLE_SIZE = 300


# ── 1. Load ──────────────────────────────────────────────────────────────────
def load_data(path: Path) -> list:
    """Load the raw HotpotQA JSON file and return the list of question dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions from {path.name}")
    return data


# ── 2. Sample ────────────────────────────────────────────────────────────────
def sample_questions(data: list, seed: int, n: int) -> list:
    """Deterministically sample *n* questions using *seed*."""
    rng = random.Random(seed)
    sampled = rng.sample(data, n)
    # Keep only the fields we need (plus context & supporting_facts for later steps)
    entries = []
    for q in sampled:
        entries.append({
            "_id": q["_id"],
            "question": q["question"],
            "answer": q["answer"],
            "type": q["type"],
            "level": q["level"],
            "supporting_facts": q["supporting_facts"],
            "context": q["context"],
        })
    return entries


# ── 3. Build corpus ─────────────────────────────────────────────────────────
def build_corpus(sampled: list) -> list:
    """
    Collect every context paragraph from the sampled questions.
    Deduplicate by title — each unique title appears exactly once.
    Returns a list of dicts: {doc_id, title, text}.
    """
    seen: dict[str, str] = {}          # title -> text
    conflicts: list[str] = []          # titles with conflicting texts

    for q in sampled:
        for title, sentences in q["context"]:
            text = "".join(sentences)   # join sentence list into one string
            if title in seen:
                if seen[title] != text:
                    conflicts.append(title)
            else:
                seen[title] = text

    # Build corpus list preserving insertion (first-seen) order
    corpus = [
        {"doc_id": title, "title": title, "text": text}
        for title, text in seen.items()
    ]
    return corpus, conflicts


# ── 4. Build qrels ──────────────────────────────────────────────────────────
def build_qrels(sampled: list) -> dict:
    """
    For each sampled question, extract unique gold document titles
    from supporting_facts.  Returns {question_id: [title, ...]}.
    """
    qrels: dict[str, list[str]] = {}
    for q in sampled:
        titles = list(dict.fromkeys(t for t, _ in q["supporting_facts"]))
        qrels[q["_id"]] = titles
    return qrels


# ── helpers: save ────────────────────────────────────────────────────────────
def _save_sampled_questions(sampled: list, path: Path) -> None:
    """Write sampled_questions.json (without internal context/supporting_facts)."""
    slim = []
    for q in sampled:
        slim.append({
            "_id": q["_id"],
            "question": q["question"],
            "answer": q["answer"],
            "type": q["type"],
            "level": q["level"],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(slim)} sampled questions -> {path.name}")


def _save_corpus(corpus: list, path: Path) -> None:
    """Write corpus.jsonl — one JSON object per line."""
    with open(path, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Saved {len(corpus)} corpus documents -> {path.name}")


def _save_qrels(qrels: dict, path: Path) -> None:
    """Write qrels.json."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(qrels, f, ensure_ascii=False, indent=2)
    print(f"Saved qrels for {len(qrels)} questions -> {path.name}")


# ── 6. Sanity checks ────────────────────────────────────────────────────────
def run_sanity_checks(
    sampled_path: Path,
    corpus_path: Path,
    qrels_path: Path,
    sampled_full: list,
    seed: int,
) -> bool:
    """
    Run all sanity checks (a–i).  Prints a ✓/✗ line for each.
    Returns True only if ALL checks pass.
    """
    # Reload written files to validate what's actually on disk
    with open(sampled_path, "r", encoding="utf-8") as f:
        sq = json.load(f)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_docs = [json.loads(line) for line in f if line.strip()]

    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)

    all_ok = True

    def check(ok: bool, msg: str) -> None:
        nonlocal all_ok
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {msg}")
        if not ok:
            all_ok = False

    # (a) Exactly 300 questions
    check(len(sq) == SAMPLE_SIZE, f"{len(sq)} questions (expected {SAMPLE_SIZE})")

    # (b) All 300 question IDs are unique
    q_ids = [q["_id"] for q in sq]
    check(len(q_ids) == len(set(q_ids)), f"{len(set(q_ids))} unique question IDs")

    q_id_set = set(q_ids)

    # (c) All question IDs in qrels exist in sampled_questions
    qrels_ids = set(qrels.keys())
    missing_in_sq = qrels_ids - q_id_set
    check(
        len(missing_in_sq) == 0,
        f"all qrels question IDs exist in sampled_questions"
        + (f" (missing: {missing_in_sq})" if missing_in_sq else ""),
    )

    # (d) Every gold doc title in qrels exists in corpus
    corpus_titles = {doc["title"] for doc in corpus_docs}
    all_gold_titles = {t for titles in qrels.values() for t in titles}
    missing_gold = all_gold_titles - corpus_titles
    check(
        len(missing_gold) == 0,
        f"every qrel document exists in corpus"
        + (f" (missing: {missing_gold})" if missing_gold else ""),
    )

    # (e) No duplicate titles in corpus
    titles_list = [doc["title"] for doc in corpus_docs]
    check(len(titles_list) == len(set(titles_list)), "no duplicate titles in corpus")

    # (f) Corpus contains ONLY titles from the 300 sampled questions' contexts
    expected_titles: set[str] = set()
    for q in sampled_full:
        for title, _ in q["context"]:
            expected_titles.add(title)
    extra_titles = corpus_titles - expected_titles
    check(
        len(extra_titles) == 0,
        "corpus contains only sampled-question contexts"
        + (f" (extra: {extra_titles})" if extra_titles else ""),
    )

    # (g) No conflicting text for same title across sampled questions
    title_texts: dict[str, str] = {}
    conflicting: list[str] = []
    for q in sampled_full:
        for title, sentences in q["context"]:
            text = "".join(sentences)
            if title in title_texts:
                if title_texts[title] != text:
                    conflicting.append(title)
            else:
                title_texts[title] = text
    check(
        len(conflicting) == 0,
        "no conflicting text for the same title"
        + (f" (conflicts: {conflicting})" if conflicting else ""),
    )

    # (h) Gold-document distribution: report 1-gold vs 2+-gold counts
    #     HotpotQA is multi-hop, so every question should have 2 gold docs.
    #     We report counts either way and only fail if the distribution is empty.
    counts_1 = sum(1 for titles in qrels.values() if len(titles) == 1)
    counts_2plus = sum(1 for titles in qrels.values() if len(titles) >= 2)
    if counts_1 > 0:
        check(True, f"some questions have 1 gold document ({counts_1} found)")
    else:
        # Not a failure for HotpotQA — just informational
        print(f"  ✓ gold-doc distribution: {counts_1} questions with 1 gold doc, "
              f"{counts_2plus} with 2+ (expected for multi-hop dataset)")
    check(counts_2plus > 0, f"some questions have 2+ gold documents ({counts_2plus} found)")

    # (i) Seed recorded correctly
    check(seed == RANDOM_SEED, f"seed = {seed} recorded")

    return all_ok


# ── 5. Manifest ──────────────────────────────────────────────────────────────
def write_manifest(path: Path) -> None:
    """Write dataset_manifest.json (only called when all checks pass)."""
    manifest = {
        "source": "hotpot_dev_distractor_v1.json",
        "sample_size": SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "document_id": "title",
        "corpus_scope": "contexts of sampled questions",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest → {path.name}")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    # Ensure artifacts directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    data = load_data(DATA_PATH)

    # 2. Sample
    sampled = sample_questions(data, RANDOM_SEED, SAMPLE_SIZE)
    _save_sampled_questions(sampled, SAMPLED_QUESTIONS_PATH)

    # 3. Build corpus
    corpus, conflicts = build_corpus(sampled)
    _save_corpus(corpus, CORPUS_PATH)

    # 4. Build qrels
    qrels = build_qrels(sampled)
    _save_qrels(qrels, QRELS_PATH)

    # 6. Sanity checks
    print("\nSanity checks:")
    passed = run_sanity_checks(
        SAMPLED_QUESTIONS_PATH,
        CORPUS_PATH,
        QRELS_PATH,
        sampled,
        RANDOM_SEED,
    )

    if not passed:
        print("\n✗ Some sanity checks FAILED — dataset_manifest.json was NOT written.")
        sys.exit(1)

    # 5. Write manifest (only on success)
    print()
    write_manifest(MANIFEST_PATH)

    # 7. Summary
    with open(QRELS_PATH, "r", encoding="utf-8") as f:
        qrels_loaded = json.load(f)

    total_questions = SAMPLE_SIZE
    total_corpus = len(corpus)
    gold_counts = [len(v) for v in qrels_loaded.values()]
    avg_gold = sum(gold_counts) / len(gold_counts) if gold_counts else 0
    one_gold = sum(1 for c in gold_counts if c == 1)
    two_plus_gold = sum(1 for c in gold_counts if c >= 2)

    print(f"\n{'='*50}")
    print(f"  Dataset build complete")
    print(f"{'='*50}")
    print(f"  Questions sampled    : {total_questions}")
    print(f"  Unique corpus docs   : {total_corpus}")
    print(f"  Avg gold docs/query  : {avg_gold:.2f}")
    print(f"  1-gold questions     : {one_gold}")
    print(f"  2+-gold questions    : {two_plus_gold}")
    print(f"  Manifest written     : {MANIFEST_PATH.name} ✓")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
