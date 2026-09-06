# Pre-Registration of Retrieval Improvement Design

This document pre-registers the targeted retrieval improvement design for Research Question 3 (RQ3) prior to implementing or evaluating the code. In accordance with the scientific methodology established in RESEARCH_PLAN.md, this plan is committed before running any improvement experiments or observing any results.

## 1. Targeted Failure Type

This intervention targets Multi-Hop Bridging Failures (Second-Hop Invisibility), as defined in docs/FAILURE_ANALYSIS.md Section 2.1.

In the systematic baseline failure analysis, multi-hop bridging failure was identified as the single largest failure category, representing 14 of the 42 total failure instances (33.33%) across the union of failing queries at Recall@5 = 0. It affected all three baseline retrieval methods:
- BM25: 10 failure cases
- Hybrid (RRF): 3 failure cases
- Dense: 1 failure case

In these queries, the question requires information spanning two distinct documents to reach an answer, but the second gold document is structurally unmentioned in the prompt text (e.g. an unstated actor name, a song title, or an author). Single-step retrieval systems score candidate documents in isolation against the unexpanded prompt, making second-hop target documents invisible to lexical and semantic matching.

## 2. Proposed Improvement: Two-Step (Iterative) Hybrid Retrieval with Query Expansion

We propose a two-step iterative retrieval architecture that uses hop-1 context to bridge to hop-2 documents without introducing new external models or training:

1. Hop 1 Retrieval:
   - Reuse the existing baseline Hybrid (RRF) ranked list for each query.
   - Hop 1 operates strictly on the original question text, exactly identical to the evaluated baseline.

2. Hop 1 Query Expansion:
   - Extract the top hop1_top_n documents (configured in config/config.yaml as hop1_top_n: 3) from the Hop 1 Hybrid ranking.
   - Retrieve their raw, uncleaned text from artifacts/corpus.jsonl.
   - Construct an expanded query by concatenating the original question with the full text of these top documents:
     expanded_query = original_question + " " + doc_1_text + " " + doc_2_text + " " + doc_3_text

3. Hop 2 Retrieval:
   - BM25 Retrieval: Tokenize the expanded query using the baseline lowercase regex tokenizer and score all corpus documents using the existing BM25Okapi index (k1 = 1.5, b = 0.75).
   - Dense Retrieval: Embed the expanded query using the baseline sentence-transformers/all-MiniLM-L6-v2 model with L2 normalization, and query the existing FAISS IndexFlatIP index.
   - Hop 2 Fusion: Combine the hop-2 BM25 ranking and hop-2 Dense ranking using baseline Reciprocal Rank Fusion (RRF with k = 60) to produce a Hop 2 Hybrid ranking.

4. Final Ranking Fusion:
   - Combine the Hop 1 Hybrid ranking and the Hop 2 Hybrid ranking using RRF (k = 60):
     score(d) = (1 / (60 + rank_hop1(d))) + (1 / (60 + rank_hop2(d)))
   - Sort descending by fused score to produce the final "Improved Method" ranking per query.

Architectural Constraints:
- No new models, embeddings, or external libraries (such as LLM generation, Named Entity Recognition, or fine-tuned re-rankers) are introduced.
- The pipeline reuses the exact BM25, Dense, and RRF components already built, validated, and frozen.

## 3. Theoretical Justification (Why It Is Expected to Work)

In HotpotQA multi-hop questions, while the second gold document title is absent from the question, it is frequently referenced inside the text of the first gold document. For example:
- In query 5abc19705542993a06baf86e ("Black Book starred the actress and writer of what heritage?"), the prompt does not mention "Halina Reijn". However, the Wikipedia article for "Black Book (film)" lists Halina Reijn prominently in its opening paragraph.
- By retrieving the first document at Hop 1 and appending its text to the query, the Hop 2 retrieval pass gains direct lexical overlap and semantic relatedness with the second gold document.
- Fusing Hop 1 and Hop 2 ensures that documents highly ranked in both hops remain at the top, preventing the first gold document from being dropped while lifting the second gold document into the top 10.

## 4. Expected Performance and Failure Behavior

### Primary Prediction:
- A measurable reduction in the count of Multi-Hop Bridging Failures among the 14 baseline failure cases identified in results/failure_taxonomy_mapping.json.
- Specifically, we anticipate that queries where Hop 1 successfully ranked the primary bridge document in the top 3 will successfully surface the second gold document during Hop 2.

### Secondary Prediction:
- Overall Recall@5 and Recall@10 are expected to improve or remain highly competitive relative to the baseline Hybrid model (0.9733 Recall@5, 0.9933 Recall@10).

### Explicit Failure Trade-Offs and Risks:
- Incomplete Resolution: We explicitly predict that this approach will NOT resolve all 14 bridging failure cases. In cases where Hop 1 fails to place the first gold document in the top 3 (such as query 5a7344e95542991f9a20c6ce where both gold songs were missing from the top 10), Hop 1 will expand the query with distractor text, preventing Hop 2 from finding the true bridge.
- Risk of Query Drift and Compounding Errors: If Hop 1 retrieves an irrelevant distractor document (particularly in cases vulnerable to Entity Confusion or Topical Distraction), concatenating several paragraphs of distractor text may dilute the original question keywords and cause new retrieval failures on queries that previously succeeded.

## 5. Experimental Controls and Evaluation Protocol

To ensure rigorous and unbiased evaluation, the Improved Method will be evaluated under the identical protocol as the baselines:
- Dataset: The identical frozen 300 questions from artifacts/sampled_questions.json.
- Corpus: The identical 2,988 documents from artifacts/corpus.jsonl.
- Gold Labels: artifacts/qrels.json.
- Metrics: Recall@1, Recall@5, Recall@10, and nDCG@10 (identical calculation logic via src/evaluate.py).
- Group Breakdown: Identical evaluation across Bridge (n = 254) and Comparison (n = 46) question subsets.
- Parameter Fixation: hop1_top_n is pre-registered at 3 and will not be tuned against evaluation results.
