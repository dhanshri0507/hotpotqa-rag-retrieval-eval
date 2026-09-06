# HotpotQA RAG Retrieval Evaluation and Failure-Driven Improvement

A research project evaluating retrieval architectures (**BM25**, **Dense retrieval via sentence-transformers**, and **Hybrid RRF fusion**) on a frozen 300-question subset of HotpotQA (distractor split), followed by a systematic failure analysis and an iterative hybrid retrieval improvement with query expansion.

## Overview and Key Results

All experiments operate on a frozen benchmark of 300 questions (254 bridge, 46 comparison) and 2,988 candidate Wikipedia paragraphs with binary relevance ground truth (`qrels.json`).

| Retrieval Method | Recall@1 | Recall@5 | Recall@10 | nDCG@10 |
| :--- | :---: | :---: | :---: | :---: |
| BM25 | 0.6933 | 0.9067 | 0.9733 | 0.6894 |
| Dense (all-MiniLM-L6-v2) | 0.8333 | 0.9800 | 0.9900 | 0.7726 |
| Hybrid (RRF Baseline) | 0.7533 | 0.9733 | 0.9933 | 0.7581 |
| **Improved Method (Two-Step Hybrid)** | **0.6533** | **0.9633** | **0.9933** | **0.7184** |

### Key Findings:
- **Dense Dominance at Top Ranks:** Dense retrieval achieved the highest Recall@1 (0.8333) and nDCG@10 (0.7726), particularly outperforming BM25 on comparison questions (0.9348 vs. 0.5435 Recall@1).
- **Hybrid Coverage:** Hybrid retrieval achieved the highest baseline Recall@10 (0.9933), retrieving gold documents for 298 out of 300 queries.
- **Targeted Improvement (40% Resolution):** The two-step iterative hybrid retrieval with query expansion successfully resolved 40.0% (4 of 10) of the hardest Multi-Hop Bridging Failures by bringing both gold documents into the top 10, while matching the peak Recall@10 (0.9933).

---

## Project Structure

```
├── data/
│   └── hotpot_dev_distractor_v1.json      # Raw HotpotQA dev/distractor file (download link below)
├── artifacts/                              # Frozen dataset (deterministic, seed 42)
│   ├── sampled_questions.json              # 300 sampled questions (254 bridge, 46 comparison)
│   ├── corpus.jsonl                        # 2988 deduplicated Wikipedia paragraphs
│   ├── qrels.json                          # Gold labels: {qid: [title1, title2]}
│   └── dataset_manifest.json              # Build metadata and sanity check logs
├── config/
│   └── config.yaml                         # Centralized parameters (single source of truth)
├── docs/
│   ├── FAILURE_ANALYSIS.md                 # Systematic taxonomy and failure case diagnosis
│   ├── IMPROVEMENT.md                      # Pre-registered two-step retrieval improvement design
│   └── IMPROVEMENT_ANALYSIS.md             # Empirical analysis of improvement results and trade-offs
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── utils.py                            # Shared helpers (loading, saving, config resolution)
│   ├── bm25_retrieval.py                   # Sparse retrieval baseline
│   ├── dense_retrieval.py                  # Dense retrieval baseline (sentence-transformers + FAISS)
│   ├── hybrid_retrieval.py                 # Hybrid RRF fusion baseline
│   ├── evaluate.py                         # Baseline evaluation script
│   ├── failure_analysis.py                 # Systematic failure extraction and taxonomy mapping
│   ├── improved_retrieval.py               # Two-step iterative hybrid retrieval with query expansion
│   └── evaluate_improvement.py            # Evaluation of improved method and targeted checks
├── embeddings/                             # Cached 384-dim corpus embeddings and doc IDs
├── results/                                # Output rankings, evaluation tables, and taxonomy artifacts
│   ├── bm25_rankings.jsonl                 # BM25 full ranked lists
│   ├── dense_rankings.jsonl                # Dense full ranked lists
│   ├── hybrid_rankings.jsonl               # Hybrid baseline ranked lists
│   ├── improved_rankings.jsonl             # Improved two-step hybrid ranked lists
│   ├── metrics_overall.json                # Baseline metrics summary
│   ├── metrics_by_question_type.json       # Baseline metrics split by bridge/comparison
│   ├── comparison_table.csv                # Baseline CSV comparison table
│   ├── metrics_overall_improved.json       # All-method metrics summary including improved
│   ├── metrics_by_question_type_improved.json # Subgroup metrics including improved
│   ├── comparison_table_with_improved.csv  # 4-method CSV comparison table
│   ├── query_level_results.jsonl           # Per-query baseline detail
│   ├── query_level_results_improved.jsonl  # Per-query detail for all methods
│   ├── failure_cases_raw.jsonl             # Extracted failure case records (Recall@5 = 0)
│   ├── failure_taxonomy_mapping.json       # 100% mapping of failure instances to taxonomy
│   ├── failure_taxonomy_summary.json       # Per-category failure statistics
│   ├── improvement_targeted_check.json     # Resolution audit for multi-hop bridging failures
│   └── improvement_new_failures.json       # Audit of newly introduced retrieval failures
├── build_dataset.py                        # Dataset construction script
├── RESEARCH_PLAN.md                        # Pre-registered experimental plan
├── RESULTS.md                              # Baseline results report
├── ANALYSIS.md                             # Baseline interpretation report (H1, H4, Limitations)
├── requirements.txt                        # Pinned Python dependencies
└── README.md                               # This reproduction guide
```

---

## Prerequisites and Setup

- **Python 3.10+**
- The raw HotpotQA validation file `data/hotpot_dev_distractor_v1.json` is needed only if rebuilding the dataset from scratch:
  - Official project page: [https://hotpotqa.github.io/](https://hotpotqa.github.io/)
  - Canonical download URL: [http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json) (also mirrored on [Hugging Face Datasets: hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa))
- All frozen evaluation artifacts in `artifacts/` and precomputed embeddings in `embeddings/` are tracked in the repository.

### Install Dependencies
```bash
pip install -r requirements.txt
```
Installed packages include `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `numpy`, `pyyaml`, and `torch`.

---

## Step-by-Step Reproduction Guide

Execute all commands from the project root directory.

### Step 1: Dataset Construction (Optional: Already Built and Frozen)
The dataset artifacts are frozen in `artifacts/`. To verify deterministic reconstruction:
```bash
python build_dataset.py
```
Samples 300 questions (seed 42), builds the 2,988-document corpus, validates 9 sanity checks, and records MD5 manifests.

### Step 2: BM25 Sparse Retrieval
```bash
python -m src.bm25_retrieval
```
- Tokenization: lowercase regex `\w+` (no stemming, no stopwords).
- Parameters: BM25Okapi with k1 = 1.5, b = 0.75.
- Output: `results/bm25_rankings.jsonl`.

### Step 3: Dense Retrieval
```bash
python -m src.dense_retrieval
```
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2-normalized).
- Index: Exact inner-product search with FAISS `IndexFlatIP`.
- Output: `results/dense_rankings.jsonl` (uses cached vectors in `embeddings/`).

### Step 4: Hybrid Baseline Retrieval (RRF)
```bash
python -m src.hybrid_retrieval
```
- Fuses top-100 results from BM25 and Dense using Reciprocal Rank Fusion:
  $$\text{score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{60 + \text{rank}_m(d)}$$
- Output: `results/hybrid_rankings.jsonl`.

### Step 5: Baseline Evaluation
```bash
python -m src.evaluate
```
- Runs self-testing metric validation with a toy hand-calculated example before evaluation.
- Computes Recall@1, Recall@5, Recall@10, and nDCG@10 overall and by question type.
- Outputs: `metrics_overall.json`, `metrics_by_question_type.json`, `comparison_table.csv`, and `query_level_results.jsonl`.

### Step 6: Systematic Failure Analysis
```bash
python -m src.failure_analysis
```
- Evaluates failure cases across methods using an objective, code-defined criterion.
- Widens threshold to Recall@5 = 0 to satisfy the 20-query evaluation floor (yielded 33 unique queries, 42 total failure instances).
- Generates: `results/failure_cases_raw.jsonl`, `results/failure_taxonomy_mapping.json`, and `results/failure_taxonomy_summary.json`.
- Full analysis report: `docs/FAILURE_ANALYSIS.md`.

### Step 7: Two-Step Iterative Hybrid Retrieval (Targeted Improvement)
```bash
python -m src.improved_retrieval
```
- Hop 1: Loads baseline Hybrid rankings.
- Query Expansion: Appends full text of top-3 Hop 1 documents to the query.
- Hop 2: Scores expanded queries with BM25 and Dense, fusing via RRF ($k=60$).
- Final Fusion: Combines Hop 1 and Hop 2 via RRF ($k=60$).
- Output: `results/improved_rankings.jsonl`.

### Step 8: Improvement Evaluation and Targeted Audit
```bash
python -m src.evaluate_improvement
```
- Evaluates the Improved Method under identical metric logic.
- Conducts a targeted audit on all queries categorized under Multi-Hop Bridging Failure.
- Checks for newly introduced failure cases.
- Outputs: `metrics_overall_improved.json`, `metrics_by_question_type_improved.json`, `comparison_table_with_improved.csv`, `query_level_results_improved.jsonl`, `improvement_targeted_check.json`, and `improvement_new_failures.json`.
- Full analysis report: `docs/IMPROVEMENT_ANALYSIS.md`.

---

## Research Documentation Map

- **[RESEARCH_PLAN.md](RESEARCH_PLAN.md)**: Pre-registered experimental plan, research questions (RQ1 to RQ4), hypotheses (H1 to H4), and risk analysis.
- **[RESULTS.md](RESULTS.md)**: Detailed report of baseline experimental metrics and query-level overlap statistics.
- **[ANALYSIS.md](ANALYSIS.md)**: Baseline interpretation analyzing why Dense outperforms Hybrid at Rank 1 and why Comparison questions show a wide performance gap.
- **[docs/FAILURE_ANALYSIS.md](docs/FAILURE_ANALYSIS.md)**: Systematic failure taxonomy (Multi-Hop Bridging, Entity Confusion, Topical Distraction, Lexical Mismatch) with concrete query case studies.
- **[docs/IMPROVEMENT.md](docs/IMPROVEMENT.md)**: Pre-registered design document for iterative hybrid retrieval with query expansion.
- **[docs/IMPROVEMENT_ANALYSIS.md](docs/IMPROVEMENT_ANALYSIS.md)**: Evaluation of the improvement, documenting the 40% bridging resolution rate and the precision-coverage trade-off.

---

## Reproducibility Guarantees

1. **Deterministic Pipeline:** Random seed 42 freezes the 300 sampled questions. BM25, exact FAISS inner-product search, and RRF fusion contain zero stochastic elements.
2. **Centralized Parameters:** All parameters live exclusively in `config/config.yaml`. No hardcoded values exist in the source code.
3. **Automated Sanity Checks:** Every script includes assertions that verify ranking completeness, absence of duplicates, and score monotonicity before saving.
4. **Independent Execution:** Running the full sequence of commands from Step 1 to Step 8 reproduces bit-for-bit identical ranking files and metrics tables.
