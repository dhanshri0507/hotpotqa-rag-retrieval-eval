# Research Plan

**Topic:** Failure-Aware Evaluation and Improvement of a Small RAG Retrieval Pipeline

This document is written and committed before running the main experiments, as required. It defines the research questions, hypotheses, dataset, retrieval baselines, evaluation metrics, expected risks, and the planned experimental sequence. 

---

## 1. Research Questions

**RQ1:** How do BM25, Dense Retrieval, and Hybrid Retrieval differ in retrieval performance?

**RQ2:** What major types of retrieval failures occur, and how do the failure patterns differ across retrieval methods?

**RQ3:** Can a targeted improvement derived from the observed failure analysis improve the relevant failure type and retrieval performance?

**RQ4:** How does retrieval performance vary across HotpotQA question types (Bridge vs. Comparison)?

---

## 2. Hypotheses

Each hypothesis is stated in a form that can be tested against the metrics defined in Section 5, and is written before the main experiments are run.

**H1 (RQ1):**
Hybrid retrieval will outperform both BM25 and Dense retrieval individually across Recall@1, Recall@5, Recall@10, and nDCG@10, since it combines the lexical exact-match strength of BM25 with the semantic matching ability of Dense retrieval. The advantage is expected to be smaller at Recall@1 (fusion has less room to help when there's only one slot) and larger at Recall@5/Recall@10 (fusion has more room to surface a document that either single method ranked a bit lower).

**H2 (RQ2):**
BM25, Dense, and Hybrid retrieval will show different failure patterns, and these patterns will fall into at least four distinct categories, each hitting the methods differently:

1. **Lexical mismatch** (paraphrased or synonym-heavy questions, where the question wording and the gold paragraph's wording barely overlap). Expected to hurt BM25 the most, since it has no way to see word-level similarity.
2. **Multi-hop bridging failure** (the second gold document only becomes relevant once you already know a fact from the first one; on its own it looks weakly related to the original question). Expected to hurt both BM25 and Dense, since neither method reasons across hops, it just scores each document against the raw question text. Hybrid may partially recover some of these through rank fusion, but is not expected to fully solve them.
3. **Semantic drift / topical distraction** (a distractor paragraph about the same general topic or entity as the gold one outranks it, because it "looks similar" in meaning even though it doesn't answer the question). Expected to hurt Dense the most, since embedding similarity is more sensitive to topical closeness than to exact factual relevance.
4. **Entity confusion** (two different documents mention similar or identically-named entities, e.g. two people or places with the same name, and the wrong one gets retrieved). Expected to affect both BM25 (matches the name string regardless of which entity it refers to) and Dense (embeds both mentions similarly), so this is a case where Hybrid may not help much either, since both underlying signals are confused the same way.

Additional failure categories may be added once the failure analysis (RQ2 proper) is underway, if the data suggests a pattern not covered above.

**H3 (RQ3):**
A targeted retrieval improvement designed from the observed failure analysis will improve performance specifically on the targeted failure type, and will improve overall Hybrid retrieval performance relative to the original, untuned Hybrid baseline, without meaningfully hurting other failure types.

**H4 (RQ4):**
Comparison questions will show a smaller performance gap between BM25 and Dense retrieval than Bridge questions do. The reasoning: Comparison questions tend to rely more on direct entity or keyword overlap, which favors BM25, while Bridge questions need the model to connect two facts that are related in meaning but not in wording, which favors Dense (and by extension, Hybrid).

---

## 3. Dataset Construction (summary)

Full detail and reproducible code live in `build_dataset.py`. This is a short summary for context.

- **Source:** HotpotQA, distractor setting, validation (dev) split, `hotpot_dev_distractor_v1.json`.
- **Sampling:** 300 questions sampled deterministically with `random_seed = 42`.
- **Corpus:** built only from the context paragraphs attached to the 300 sampled questions, deduplicated by paragraph title (`document_id = title`). No paragraphs outside the sampled questions' contexts are included.
- **Gold labels (qrels):** derived from each question's `supporting_facts`, at the paragraph-title level (not sentence level).
- **Note on gold-document counts:** HotpotQA is a multi-hop dataset by design, and every question in the full 7,405-question dev set has exactly 2 gold supporting documents, no question has just 1. Because of this, an earlier planned sanity check ("some questions have 1 gold document, some have 2") can never be satisfied and was adjusted. The check now reports the actual gold-count distribution for the sampled questions and only fails if any question ends up with zero gold documents, which would signal something broke earlier in the pipeline (e.g. a paragraph missing from the corpus).
- **Frozen artifacts:** `artifacts/sampled_questions.json`, `artifacts/corpus.jsonl`, `artifacts/qrels.json`, `artifacts/dataset_manifest.json`. These are fixed inputs for every retrieval method below, so no method sees a different dataset.
- **Validated by:** the sanity-check suite in `build_dataset.py` (unique question IDs, corpus scope, no orphan qrels, no duplicate or conflicting titles, and the gold-count check described above).

---

## 4. Retrieval Baselines

All three methods run over the same frozen corpus (Section 3) and the same 300 queries, so results are directly comparable.

### Baseline A: Sparse Retrieval (BM25)
- **Library:** `rank_bm25` (`BM25Okapi`)
- **Tokenizer:** lowercase the text, then split on word characters only (regex `\w+`). No stemming, no stopword removal, kept simple on purpose so it's easy to explain and debug.
- **Parameters:** `k1 = 1.5`, `b = 0.75` (standard defaults from the BM25 literature; not tuned, since there's no validation set to tune against).
- **Chunking:** each HotpotQA paragraph in the corpus is treated as one retrieval unit, no further splitting. This matches the level at which `supporting_facts` (the gold labels) are annotated, so evaluation stays unambiguous.

### Baseline B: Dense Retrieval
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Why this model:** it's a widely used, well-benchmarked general-purpose sentence embedding model, a common baseline choice in RAG work, so results are easy to compare against what others report. It's also small (384 dimensions, about 80MB) and fast enough to run on CPU at this corpus size, no GPU needed. It's symmetric too, meaning no separate query/passage prefix is required, which keeps the pipeline simpler for a first implementation.
- **Similarity:** cosine similarity, done as inner product over L2-normalized embeddings.
- **Vector index:** FAISS `IndexFlatIP`, exact (non-approximate) search. Chosen because the corpus is small enough that exact search runs fast, which keeps approximation error out of the comparison entirely.
- **Chunking:** same paragraph-level units as BM25 above, embedded once and cached.

### Baseline C: Hybrid Retrieval
- **Fusion method:** Reciprocal Rank Fusion (RRF).
- **Formula:** for each document *d*, `score(d) = sum over methods of 1 / (k + rank_method(d))`, added up across the ranked lists where *d* shows up (BM25 and Dense).
- **Parameter:** `k = 60`, the standard constant used in the RRF literature; not tuned, for the same reason as BM25's parameters.
- **Inputs:** top-ranked results (e.g. top-100) from BM25 and Dense per query, enough since the final metrics only look at top-1/5/10. RRF is used here specifically because BM25 scores and cosine similarity scores aren't on the same scale, but rank positions are, so fusing by rank sidesteps that problem.

### Summary Table

| Component | Choice | Key parameter(s) |
|---|---|---|
| Sparse (BM25) | `rank_bm25` BM25Okapi | k1=1.5, b=0.75, lowercase/regex tokenizer |
| Chunking | Paragraph-level, no splitting | matches `supporting_facts` granularity |
| Dense embedding | `all-MiniLM-L6-v2` | 384-dim, cosine similarity (normalized dot product) |
| Vector index | FAISS `IndexFlatIP` | exact search, no approximation |
| Fusion | Reciprocal Rank Fusion | k = 60 |

---

## 5. Evaluation Metrics

Computed for every method (BM25, Dense, Hybrid), against `artifacts/qrels.json`:

- **Recall@1**, whether at least one gold document is retrieved in the top-1 result. Since every question has exactly 2 gold documents (see Section 3), Recall@1 is a strict metric across the whole dataset, not just some subset: a single top-1 slot can never hold both gold documents, so a perfect Recall@1 score is not achievable even in principle. This is worth keeping in mind when comparing Recall@1 against Recall@5/@10.
- **Recall@5**
- **Recall@10**
- **nDCG@10**

Metrics will also be broken down by question type (Bridge vs. Comparison) to support RQ4/H4, reported as:

```
             BM25    Dense    Hybrid
Bridge        ...      ...       ...
Comparison    ...      ...       ...
```

---

## 6. Expected Risks or Limitations

- **Sample size:** 300 questions limits statistical power. Differences between methods may fall within noise, especially once broken down by question type, where subgroup sizes get even smaller.
- **Embedding model generality:** `all-MiniLM-L6-v2` is general-purpose, not fine-tuned for retrieval or for this domain. Dense performance here is a lower bound on what a dedicated retrieval-specialized embedding model could achieve.
- **Untuned fusion and BM25 parameters:** RRF's `k=60` and BM25's `k1`/`b` are literature defaults, not tuned to this dataset, since no held-out validation set exists for tuning. Results reflect reasonable defaults, not optimized settings.
- **Simple tokenizer:** the BM25 tokenizer does no stemming or stopword removal, so it may miss valid lexical matches that differ only in small ways (plurals, verb tense, etc.).
- **Exact search only:** FAISS runs with a flat (exact) index, which avoids approximation noise but doesn't reflect the performance/latency trade-offs of the approximate indexes used in larger production systems.
- **Recall@1 is structurally hard here:** as noted in Section 5, every question has 2 gold documents, so Recall@1 can never reach a perfect score, even for a very strong retriever. This is a property of the dataset, not a flaw in the retrieval methods.
- **Distractor setting:** the HotpotQA distractor split gives each question a constrained, pre-selected paragraph pool rather than the full open-Wikipedia retrieval problem. Results should be read as retrieval quality within that pool, not as full-Wikipedia-scale retrieval performance.

---

## 7. Planned Experimental Sequence

1. Load the frozen corpus and treat it as-is (chunking: one paragraph = one retrieval unit, no splitting).
2. Implement and sanity-check BM25 retrieval (Baseline A) over the corpus (tokenize corpus + queries, build `BM25Okapi` index, get ranked list per query).
3. Implement and sanity-check Dense retrieval (Baseline B): embed the corpus once with `all-MiniLM-L6-v2` and cache it (normalize, save embeddings + doc_id mapping), embed queries the same way, build FAISS `IndexFlatIP`, run search to get ranked list per query.
4. Implement Hybrid retrieval (Baseline C) via RRF (`k=60`), combining the BM25 and Dense ranked lists from steps 2 to 3.
5. Run the evaluation script to compute Recall@1, Recall@5, Recall@10, and nDCG@10 for all three methods against `qrels.json`, including the Bridge vs. Comparison breakdown for RQ4. Save results to a results file.
6. Analyze retrieval failures per method and categorize them (lexical mismatch, multi-hop bridging, semantic drift, entity confusion, or new categories as needed) to feed RQ2.
7. Design and implement one targeted improvement based on the failure analysis (RQ3), then re-run evaluation to compare against the original Hybrid baseline.
8. Write up findings against H1 to H4, noting which hypotheses were supported, partially supported, or rejected, tying back to the metrics and failure analysis above.

---

*This plan is committed before running the main experiments and before examining results, per the project requirements. Any deviation from this plan during implementation will be documented explicitly, along with the reason for the change.*
