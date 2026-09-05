#!/usr/bin/env python3
"""
utils.py — Shared helpers for the HotpotQA RAG retrieval pipeline.

Provides:
  - Config loading (config/config.yaml)
  - Artifact loading (corpus, questions, qrels)
  - Rankings I/O (save/load JSONL ranking files)
  - Directory helpers
"""

import json
import sys
from pathlib import Path

import yaml

# ── Windows console UTF-8 fix ───────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the centralised config.yaml and return it as a dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to project root."""
    return PROJECT_ROOT / relative_path


# ── Artifact loaders ────────────────────────────────────────────────────────

def load_corpus(cfg: dict) -> list[dict]:
    """
    Load artifacts/corpus.jsonl.
    Returns a list of dicts, each with keys: doc_id, title, text.
    """
    path = resolve_path(cfg["artifact_paths"]["corpus"])
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def load_questions(cfg: dict) -> list[dict]:
    """
    Load artifacts/sampled_questions.json.
    Returns a list of dicts, each with keys: _id, question, answer, type, level.
    """
    path = resolve_path(cfg["artifact_paths"]["sampled_questions"])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_qrels(cfg: dict) -> dict[str, list[str]]:
    """
    Load artifacts/qrels.json.
    Returns {question_id: [gold_title_1, gold_title_2, ...]}.
    """
    path = resolve_path(cfg["artifact_paths"]["qrels"])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Rankings I/O ─────────────────────────────────────────────────────────────

def save_rankings(rankings: dict[str, list[tuple[str, float]]], path: Path) -> None:
    """
    Save per-query ranked lists to a JSONL file.

    Args:
        rankings: {question_id: [(doc_id, score), ...]} — sorted descending by score.
        path: output file path.
    """
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked_list in rankings.items():
            record = {
                "question_id": qid,
                "rankings": [[doc_id, score] for doc_id, score in ranked_list],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_rankings(path: Path) -> dict[str, list[tuple[str, float]]]:
    """
    Load per-query ranked lists from a JSONL file.

    Returns:
        {question_id: [(doc_id, score), ...]}
    """
    rankings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = record["question_id"]
            ranked_list = [(doc_id, score) for doc_id, score in record["rankings"]]
            rankings[qid] = ranked_list
    return rankings


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dir(directory: Path) -> None:
    """Create directory (and parents) if it doesn't exist."""
    directory.mkdir(parents=True, exist_ok=True)


def check_file_overwrite(path: Path, script_name: str) -> None:
    """
    Warn clearly if an output file already exists.
    Proceeds with overwrite after printing a warning.
    """
    if path.exists():
        print(f"[{script_name}] WARNING: {path.name} already exists and will be overwritten.")
