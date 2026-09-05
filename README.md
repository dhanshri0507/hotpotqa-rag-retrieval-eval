# HotpotQA RAG Retrieval Evaluation

A mini research project evaluating three retrieval methods: **BM25**, **Dense (sentence-transformers)**, and **Hybrid (RRF)**, on a 300-question subset of HotpotQA (distractor split), with Recall@k and nDCG@10 metrics.

## Project Structure

```
├── data/
│   └── hotpot_dev_distractor_v1.json      # Raw HotpotQA dev/distractor file
├── artifacts/                              # Frozen dataset (do NOT modify)
│   ├── sampled_questions.json              # 300 deterministically sampled questions
│   ├── corpus.jsonl                        # 2988 deduplicated context paragraphs
│   ├── qrels.json                          # Gold labels: {qid: [title1, title2]}
│   └── dataset_manifest.json              # Build metadata
├── config/
│   └── config.yaml                         # All experiment parameters (single source of truth)
├── src/
│   ├── utils.py                            # Shared helpers (loading, saving, config)
│   ├── bm25_retrieval.py                   # Sparse retrieval
│   ├── dense_retrieval.py                  # Dense retrieval (sentence-transformers + FAISS)
│   ├── hybrid_retrieval.py                 # Hybrid retrieval (Reciprocal Rank Fusion)
│   └── evaluate.py                         # Metrics computation and output
├── embeddings/                             # Cached corpus embeddings (auto-generated)
├── results/                                # All experiment outputs (auto-generated)
│   ├── bm25_rankings.jsonl                 # BM25 per-query ranked lists
│   ├── dense_rankings.jsonl                # Dense per-query ranked lists
│   ├── hybrid_rankings.jsonl               # Hybrid (RRF) per-query ranked lists
│   ├── metrics_overall.json                # Recall@1/5/10, nDCG@10 per method
│   ├── metrics_by_question_type.json       # Same metrics, split by bridge/comparison
│   ├── comparison_table.csv                # Formatted comparison table
│   └── query_level_results.jsonl           # Per-query, per-method detail
├── build_dataset.py                        # Dataset construction (already run)
├── RESEARCH_PLAN.md                        # Pre-registered research plan
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## Prerequisites

- **Python 3.10+**
- The raw HotpotQA validation file `data/hotpot_dev_distractor_v1.json` must be present. It is the distractor dev split from the official HotpotQA project:
  - Official project page: [https://hotpotqa.github.io/](https://hotpotqa.github.io/)
  - Canonical download URL: [http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json) (also mirrored on [Hugging Face Datasets: hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa))
- The frozen dataset in `artifacts/` must already be built (see Step 2 below).

## Step 1: Environment Setup

```bash
pip install -r requirements.txt
```

This installs: `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `numpy`, `pyyaml`, and `torch` (required by sentence-transformers).

## Step 2: Dataset Preparation

The dataset was built once using `build_dataset.py` and is **frozen**: all retrieval methods operate on the identical artifacts.

If you need to rebuild from scratch (produces identical output given the same input file):

```bash
python build_dataset.py
```

This samples 300 questions (seed=42), builds the corpus and gold labels, runs sanity checks, and writes `artifacts/`. See `RESEARCH_PLAN.md` Section 3 for details.

> **Note:** Do not modify anything in `artifacts/`. All three methods read these files as-is.

## Step 3: BM25 Retrieval

```bash
python -m src.bm25_retrieval
```

- Tokenizes corpus and queries: lowercase + regex `\w+` (no stemming, no stopwords)
- Builds a BM25Okapi index with k1=1.5, b=0.75
- Produces a full ranked list per query
- Runs sanity checks, then saves to `results/bm25_rankings.jsonl`

## Step 4: Dense Retrieval

```bash
python -m src.dense_retrieval
```

- Embeds corpus using `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized)
- Caches embeddings to `embeddings/` (skips recomputation on subsequent runs)
- Builds FAISS IndexFlatIP (exact inner-product search)
- Produces a full ranked list per query
- Runs sanity checks, then saves to `results/dense_rankings.jsonl`

## Step 5: Hybrid Retrieval

```bash
python -m src.hybrid_retrieval
```

- Reads saved BM25 and Dense rankings (does not recompute them)
- Applies Reciprocal Rank Fusion: `score(d) = Σ 1/(60 + rank)` using top-100 from each method
- Runs sanity checks, then saves to `results/hybrid_rankings.jsonl`

## Step 6: Evaluation

```bash
python -m src.evaluate
```

- Loads gold labels from `artifacts/qrels.json`
- Computes Recall@1, Recall@5, Recall@10, and nDCG@10 for all three methods
- Also computes metrics broken down by question type (bridge vs. comparison)
- Validates its metric implementation with a toy hand-example before running on real data
- Saves four result files (see below) and prints the comparison table to console

## Result Files

| File | Contents |
|------|----------|
| `results/metrics_overall.json` | Recall@1/5/10 and nDCG@10 for each method |
| `results/metrics_by_question_type.json` | Same metrics, split by bridge / comparison, per method |
| `results/comparison_table.csv` | Formatted CSV: one row per method, columns = metrics |
| `results/query_level_results.jsonl` | Per-query detail: top-10 retrieved, gold docs, hit/miss at each k |
| `results/bm25_rankings.jsonl` | Full BM25 ranked lists (all docs scored per query) |
| `results/dense_rankings.jsonl` | Full Dense ranked lists |
| `results/hybrid_rankings.jsonl` | Fused RRF ranked lists |

## Configuration

All experiment parameters are in **`config/config.yaml`**, the single source of truth. Key values:

| Parameter | Value | Source |
|-----------|-------|--------|
| Random seed | 42 | RESEARCH_PLAN.md §3 |
| BM25 k1, b | 1.5, 0.75 | RESEARCH_PLAN.md §4 |
| Embedding model | `all-MiniLM-L6-v2` | RESEARCH_PLAN.md §4 |
| Embedding dim | 384 | Model spec |
| FAISS index | IndexFlatIP (exact) | RESEARCH_PLAN.md §4 |
| RRF k | 60 | RESEARCH_PLAN.md §4 |
| Top-k for fusion | 100 | RESEARCH_PLAN.md §4 |
| Eval metrics | Recall@1/5/10, nDCG@10 | RESEARCH_PLAN.md §5 |

These are fixed, pre-registered defaults (not tuned against evaluation results).

## Reproducibility

- The dataset is deterministic (seed=42, same input file -> identical artifacts).
- All scripts read from `config/config.yaml`; no hardcoded parameters elsewhere.
- Corpus embeddings are cached; re-running Dense retrieval skips recomputation.
- Each script includes built-in sanity checks that fail loudly on any inconsistency.
- Running the full pipeline twice produces identical results.

## Notes

- All three methods operate on the same frozen 300 questions, same corpus, and same gold labels.
- No parameter tuning is performed; all values are literature defaults.
- Scope is strictly retrieval evaluation. No answer generation or RAG chatbot components.
- See `RESEARCH_PLAN.md` for research questions, hypotheses, and expected risks.
