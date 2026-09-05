# Baseline Retrieval Results

This document records the official baseline evaluation results for the HotpotQA RAG retrieval evaluation project. All evaluations were conducted on the frozen 300-question sample using the exact pre-registered parameters from config/config.yaml.

## 1. Experimental Setup Summary

- Dataset: HotpotQA validation set (distractor split), sampled with random_seed = 42
- Queries: Exactly 300 questions (254 bridge, 46 comparison)
- Corpus: 2,988 unique Wikipedia paragraphs (title + full text)
- Ground Truth: qrels.json (each query has exactly 2 gold documents)
- Metrics:
  - Recall@k (k = 1, 5, 10): 1.0 if at least one gold document is in the top-k, else 0.0
  - nDCG@10: Normalized Discounted Cumulative Gain at rank 10 with binary relevance
- Retrieval Methods:
  - BM25: BM25Okapi (k1 = 1.5, b = 0.75, lowercase regex tokenization)
  - Dense: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, L2 normalized, FAISS IndexFlatIP exact search)
  - Hybrid: Reciprocal Rank Fusion (RRF, k = 60, top-100 inputs per method)

## 2. Overall Performance Comparison

Below are the aggregate performance metrics across all 300 evaluation questions:

| Retrieval Method | Recall@1 | Recall@5 | Recall@10 | nDCG@10 |
| :--- | :---: | :---: | :---: | :---: |
| BM25 | 0.6933 | 0.9067 | 0.9733 | 0.6894 |
| Dense | 0.8333 | 0.9800 | 0.9900 | 0.7726 |
| Hybrid (RRF) | 0.7533 | 0.9733 | 0.9933 | 0.7581 |

### Key Overall Findings:
1. Dense retrieval achieves the highest Recall@1 (0.8333) and highest nDCG@10 (0.7726).
2. Hybrid retrieval achieves the highest overall Recall@10 (0.9933), successfully surfacing gold documents in 298 out of 300 questions.
3. BM25 performs respectably at Recall@10 (0.9733) but struggles significantly at top ranks (Recall@1 = 0.6933, nDCG@10 = 0.6894).

## 3. Results by Question Type

HotpotQA questions are categorized into two structural patterns:
- Bridge questions (n = 254): Require following an intermediate entity link from one document to another.
- Comparison questions (n = 46): Compare two entities on a shared property (e.g., birthplaces, release years).

### Breakdown Table:

| Method | Question Type | Count | Recall@1 | Recall@5 | Recall@10 | nDCG@10 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| BM25 | Bridge | 254 | 0.7205 | 0.9094 | 0.9764 | 0.6952 |
| BM25 | Comparison | 46 | 0.5435 | 0.8913 | 0.9565 | 0.6570 |
| Dense | Bridge | 254 | 0.8150 | 0.9764 | 0.9882 | 0.7480 |
| Dense | Comparison | 46 | 0.9348 | 1.0000 | 1.0000 | 0.9083 |
| Hybrid | Bridge | 254 | 0.7559 | 0.9685 | 0.9921 | 0.7432 |
| Hybrid | Comparison | 46 | 0.7391 | 1.0000 | 1.0000 | 0.8402 |

### Subgroup Insights:
1. Comparison Questions: Dense retrieval shows outstanding performance, achieving 0.9348 Recall@1 and 1.0000 (perfect) Recall@5 and Recall@10. In contrast, BM25 drops sharply to 0.5435 Recall@1.
2. Bridge Questions: The performance gap between BM25 (0.7205) and Dense (0.8150) is narrower (9.45% delta) than on comparison questions (39.13% delta).
3. Hybrid Robustness: Hybrid retrieval achieves a perfect 1.0000 Recall@10 on comparison questions and 0.9921 on bridge questions, indicating strong coverage across both query types.

## 4. Method Agreement and Query Level Overlap

Analysis of query-level predictions across the 300 queries reveals the interplay between lexical and semantic retrieval:

- Recall@1 Agreement:
  - Both BM25 and Dense retrieved a gold document at Rank 1: 184 queries (61.3%)
  - Dense won alone at Rank 1: 66 queries (22.0%)
  - BM25 won alone at Rank 1: 24 queries (8.0%)
  - Neither method retrieved a gold document at Rank 1: 26 queries (8.7%)

- Recall@10 Error Recovery:
  - BM25 missed 8 queries entirely at top 10.
  - Dense missed 3 queries entirely at top 10.
  - Hybrid missed only 2 queries at top 10 (recovering 7 of BM25's 8 misses and 1 of Dense's 3 misses).

## 5. Artifact Reference

All corresponding raw and aggregated data files are saved in the results/ directory:
- results/comparison_table.csv (CSV comparison table)
- results/metrics_overall.json (Machine-readable overall scores)
- results/metrics_by_question_type.json (Machine-readable subgroup breakdown)
- results/query_level_results.jsonl (Granular per-query ranking and hit/miss data)
