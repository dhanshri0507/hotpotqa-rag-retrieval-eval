# Baseline Retrieval Analysis and Interpretation

This document provides in-depth interpretation and failure analysis of the baseline retrieval experiments on the HotpotQA dataset, directly addressing the research questions and hypotheses formulated in RESEARCH_PLAN.md.

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

## 2. Failure Analysis Taxonomy (RQ2)

Examination of query-level predictions identified four distinct failure modes matching the pre-registered categories in Hypothesis 2:

### Category 1: Multi-Hop Bridging Failures (2nd-hop Invisibility)
This represents the most severe failure mode in HotpotQA. Both gold documents are required to answer the question, but the second document is structurally invisible from the query text alone.

- Example Query (ID: 5abc19705542993a06baf86e):
  - Question: "Black Book starred the actress and writer of what heritage?"
  - Gold Documents: "Black Book (film)" and "Halina Reijn"
  - Failure Mechanism: The name "Halina Reijn" appears nowhere in the query. A single-step retriever can easily find "Black Book (film)", but has no textual bridge to rank "Halina Reijn" in the top 10 without first reading the cast list of "Black Book".
  - Outcome: Both BM25 and Dense missed the second document, and Hybrid failed to recover it.

- Example Query (ID: 5a7344e95542991f9a20c6ce):
  - Question: "What song was number 4 on the charts when a song from FutureSex/LoveSounds was number 1?"
  - Gold Documents: "Rudebox (song)" and "SexyBack"
  - Failure Mechanism: Neither "Rudebox" nor "SexyBack" is mentioned in the prompt. The retriever must infer that the number 1 song was "SexyBack", find its chart dates, and locate the number 4 song from that specific week.

---

### Category 2: Entity Distraction and Sub-entity Confusion
Occurs when a distractor document shares significant lexical overlap with the query entity, distracting sparse retrieval.

- Example Query:
  - Question: "Were Halldor Laxness and Timothy Leary from the same country?"
  - Gold Documents: "Halldor Laxness" and "Timothy Leary"
  - BM25 Top-1: "Memoir of Halldor Laxness"
  - Dense Top-1: "Halldor Laxness"
  - Failure Mechanism: BM25 matched on the dense repetition of "Halldor Laxness" in a biographical memoir passage rather than the primary encyclopedic article. Dense embeddings correctly weighted the main entity subject over the modifier.

---

### Category 3: Lexical Mismatch and Paraphrasing
Occurs when the query describes attributes or concepts without using the exact vocabulary present in the document.

- Performance Impact: Dense retrieval won outright at Rank 1 in 66 queries where BM25 failed to place a gold document in the top slot. In queries containing synonyms, implicit temporal references, or descriptive paraphrases, BM25's exact token matching failed to accumulate sufficient term-frequency weight.

---

## 3. Implications for Targeted Improvement (RQ3)

Based on this failure diagnosis, several concrete avenues exist for designing the targeted improvement in the next experimental phase:

1. Dynamic Fusion Weighting:
   Currently, RRF applies equal weights to BM25 and Dense (1:1). Because Dense dominates top-rank precision (0.8333 vs. 0.6933), a weighted reciprocal rank fusion or score-level convex combination favoring Dense at top ranks could prevent BM25 distractors from displacing true top-1 hits.

2. Document Title Exact Match Boosting:
   In encyclopedic retrieval, queries often contain the exact title of at least one target document. Augmenting retriever scoring with a specific title exact-match multiplier would resolve many entity confusion failures (e.g. promoting "Halldor Laxness" over "Memoir of Halldor Laxness").

3. Iterative / Two-Step Multi-Hop Retrieval:
   Single-step retrieval is structurally incapable of solving second-hop bridging queries when the second entity is unmentioned in the prompt. An iterative pipeline where the top-ranked document from step 1 is analyzed for entity links to expand the query for step 2 would directly target Category 1 bridging failures.

---

## 4. Limitations and Threats to Validity

1. Sample Size: The evaluation set contains 300 questions (46 comparison questions). While overall metrics have adequate sample support, subgroup conclusions on comparison questions have wider confidence intervals.
2. Distractor Setting: HotpotQA's distractor setting provides a pool of 2,988 candidate paragraphs rather than full Wikipedia (millions of documents). While this accurately reflects a scoped domain RAG corpus, absolute recall numbers are higher than would be expected at web scale.
3. Fixed Default Hyperparameters: As pre-registered in RESEARCH_PLAN.md, BM25 (k1=1.5, b=0.75) and RRF (k=60) were not tuned on validation data. Parameter tuning could adjust the relative margin between methods.
