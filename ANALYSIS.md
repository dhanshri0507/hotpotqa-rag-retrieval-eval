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

### Hypothesis 2 (RQ2: Systematic Failure Taxonomy and Vulnerability)
Prediction: Baseline retrieval failures will systematically fall into at least four distinct categories (Lexical Mismatch, Multi-Hop Bridging, Topical Distraction, Entity Confusion), each affecting retrieval methods differently based on their underlying scoring mechanisms.

Status: Supported.

Detailed Finding:
- Pre-Registered Categories Confirmed: Across 42 failure instances at Recall@5 = 0, all four hypothesized categories were empirically confirmed:
  1. Multi-Hop Bridging Failure: 14 cases (33.33%), representing the single largest failure mode.
  2. Entity Confusion: 12 cases (28.57%), primarily affecting BM25 and Hybrid.
  3. Topical Distraction: 10 cases (23.81%), primarily affecting Dense retrieval where sibling entities cluster tightly in embedding space.
  4. Lexical Mismatch: 6 cases (14.29%), heavily concentrated in BM25.
- Emergent Findings: Analysis revealed two unpredicted sub-phenomena: (1) Complete Second-Hop Invisibility (target documents sharing zero substantive tokens with the query), and (2) User Typographical and Phonetic Noise ("Hawaiin", "John-a-kite") causing catastrophic BM25 score collapse.
- Detailed Case Studies and Evidence: Fully documented in docs/FAILURE_ANALYSIS.md.

---

### Hypothesis 3 (RQ3: Targeted Retrieval Improvement via Iterative Hybrid Retrieval)
Prediction: A targeted retrieval improvement designed from the observed failure analysis will improve performance specifically on the targeted failure type, and will improve overall Hybrid retrieval performance relative to the original untuned Hybrid baseline, without meaningfully hurting other failure types.

Status: Partially Supported.

Detailed Finding:
- Targeted Bridging Resolution (Supported): The two-step iterative hybrid retrieval with query expansion resolved 40.0% (4 of 10) of the hardest multi-hop bridging failures by bringing both gold documents into the top 10, and improved 90.0% (9 of 10) of them to have at least one gold document in the top 10.
- Overall Coverage (Partially Supported): The Improved Method matched peak baseline Recall@10 at 0.9933 (298 of 300 queries retrieved).
- General Precision and Side Effects (Not Supported): Universal query expansion introduced query drift on single-hop and already-accurate queries, reducing Recall@1 from 0.7533 to 0.6533 (-10.00%) and nDCG@10 from 0.7581 to 0.7184 (-0.0397), and introduced 1 newly failing query at Recall@10.
- Detailed Trade-Off Analysis: Fully documented in docs/IMPROVEMENT.md and docs/IMPROVEMENT_ANALYSIS.md.

---

### Hypothesis 4 (RQ4: Bridge vs. Comparison Performance Gap)
Prediction: Comparison questions will show a smaller performance gap between BM25 and Dense retrieval than Bridge questions do.

Status: Not Supported.

Detailed Finding:
- On Bridge Questions: The performance gap at Recall@1 between Dense (0.8150) and BM25 (0.7205) was 0.0945 (9.45%).
- On Comparison Questions: The performance gap at Recall@1 between Dense (0.9348) and BM25 (0.5435) exploded to 0.3913 (39.13%), directly contradicting the hypothesis.
- Root Cause Analysis: The initial reasoning assumed comparison questions rely on direct entity overlap favoring BM25. In reality, comparison questions mention two distinct entities (e.g., "Which movie did Disney produce first, The Many Adventures of Winnie the Pooh or Ride a Wild Pony?"). BM25 frequently retrieved distractor passages that heavily repeated common words ("Disney", "produce") or sub-characters (e.g. "The Wonderful Thing About Tiggers"), missing the main entity articles. Dense retrieval, operating on sentence-level contextual embeddings, mapped the comparison query directly to the canonical entity articles with near-flawless precision (1.0000 Recall@5 and Recall@10).

---

### Summary of Hypotheses and Outcomes

| Hypothesis | Research Question | Prediction Summary | Outcome | Primary Document Reference |
| :--- | :--- | :--- | :---: | :--- |
| **H1** | RQ1: Baseline Performance | Hybrid outperforms BM25 and Dense across all metrics | **Partially Supported** | Section 1 (above) & RESULTS.md |
| **H2** | RQ2: Failure Taxonomy | Failures fall into 4 distinct modes with differential method impact | **Supported** | docs/FAILURE_ANALYSIS.md |
| **H3** | RQ3: Targeted Improvement | Iterative expansion fixes bridging without hurting other metrics | **Partially Supported** | docs/IMPROVEMENT_ANALYSIS.md |
| **H4** | RQ4: Question Type Gap | Comparison questions exhibit a smaller BM25 vs. Dense gap | **Not Supported** | Section 1 (above) & RESULTS.md |

---

## 2. Limitations and Threats to Validity

1. Sample Size: The evaluation set contains 300 questions (46 comparison questions). While overall metrics have adequate sample support, subgroup conclusions on comparison questions have wider confidence intervals.
2. Distractor Setting: HotpotQA's distractor setting provides a pool of 2,988 candidate paragraphs rather than full Wikipedia (millions of documents). While this accurately reflects a scoped domain RAG corpus, absolute recall numbers are higher than would be expected at web scale.
3. Fixed Default Hyperparameters: As pre-registered in RESEARCH_PLAN.md, BM25 (k1=1.5, b=0.75) and RRF (k=60) were not tuned on validation data. Parameter tuning could adjust the relative margin between methods.
