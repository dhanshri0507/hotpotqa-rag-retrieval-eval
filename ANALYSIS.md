# Baseline Retrieval Analysis and Interpretation

This document interprets the overall evaluation results (RQ1, RQ4) against the hypotheses in RESEARCH_PLAN.md. Failure analysis (RQ2) and the targeted improvement design (RQ3) are covered separately in docs/FAILURE_ANALYSIS.md, which uses a predefined, reproducible failure-selection criterion rather than illustrative examples.

## 1. Evaluation of Hypotheses

### Hypothesis 1 (RQ1: Performance Across Methods)
Prediction: Hybrid retrieval will outperform both BM25 and Dense retrieval individually across Recall@1, Recall@5, Recall@10, and nDCG@10.

Status: Partially Supported.

Detailed Finding:
- Recall@10: Supported. Hybrid achieved the highest Recall@10 (0.9933), retrieving gold documents for 298 out of 300 queries, outperforming Dense (0.9900) and BM25 (0.9733).
- Recall@1 and nDCG@10: Not Supported. Dense retrieval outperformed Hybrid at Rank 1 (0.8333 vs. 0.7533) and on nDCG@10 (0.7726 vs. 0.7581).
- Root Cause Analysis: Reciprocal Rank Fusion treats rankings symmetrically. When BM25 ranks an irrelevant distractor paragraph high due to high term frequency (e.g. repeated keyword mentions), that distractor receives a strong fusion score that can demote Dense retrieval's true top-1 hit down to ranks 2 or 3. However, at broader ranks (Recall@10), the complementary nature of BM25 and Dense shines, yielding the highest overall discovery rate.

---

### Hypothesis 4 (RQ4: Bridge vs. Comparison Performance Gap)
Prediction: Comparison questions will show a smaller performance gap between BM25 and Dense retrieval than Bridge questions do.

H4 Evaluation: Not supported.

Detailed Finding:
- On Bridge Questions: The performance gap at Recall@1 between Dense (0.8150) and BM25 (0.7205) was 0.0945 (9.45%).
- On Comparison Questions: The performance gap at Recall@1 between Dense (0.9348) and BM25 (0.5435) exploded to 0.3913 (39.13%).
- Root Cause Analysis: The initial reasoning assumed comparison questions rely on direct entity overlap favoring BM25. In reality, comparison questions mention two distinct entities (e.g., "Which movie did Disney produce first, The Many Adventures of Winnie the Pooh or Ride a Wild Pony?"). BM25 frequently retrieved distractor passages that heavily repeated common words ("Disney", "produce") or sub-characters (e.g. "The Wonderful Thing About Tiggers"), missing the main entity articles. Dense retrieval, operating on sentence-level contextual embeddings, mapped the comparison query directly to the canonical entity articles with near-flawless precision (1.0000 Recall@5 and Recall@10).

---

## 2. Limitations and Threats to Validity

1. Sample Size: The evaluation set contains 300 questions (46 comparison questions). While overall metrics have adequate sample support, subgroup conclusions on comparison questions have wider confidence intervals.
2. Distractor Setting: HotpotQA's distractor setting provides a pool of 2,988 candidate paragraphs rather than full Wikipedia (millions of documents). While this accurately reflects a scoped domain RAG corpus, absolute recall numbers are higher than would be expected at web scale.
3. Fixed Default Hyperparameters: As pre-registered in RESEARCH_PLAN.md, BM25 (k1=1.5, b=0.75) and RRF (k=60) were not tuned on validation data. Parameter tuning could adjust the relative margin between methods.
