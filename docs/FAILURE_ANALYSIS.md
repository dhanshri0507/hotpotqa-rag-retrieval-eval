# Systematic Failure Analysis

This document provides a systematic, reproducible failure analysis of the baseline retrieval methods (BM25, Dense, and Hybrid) on the HotpotQA evaluation dataset, fulfilling the requirements for Research Question 2 (RQ2). All failure cases analyzed here were extracted using a predefined, deterministic algorithmic rule implemented in src/failure_analysis.py.

## 1. Methodology and Predefined Selection Rule

To avoid cherry-picking and ensure full reproducibility, failure instances were selected using a fixed code-defined criterion based on retrieval ranking files (results/bm25_rankings.jsonl, results/dense_rankings.jsonl, results/hybrid_rankings.jsonl) evaluated against gold labels (artifacts/qrels.json).

### Selection Criterion and Threshold Widening:
1. Primary Definition: A query is identified as a failure for a given retrieval method if that method fails to retrieve any gold document in its top ranks (Recall@k = 0).
2. Floor Rule: The analysis protocol mandates an evaluation floor of at least 20 unique failing queries across the union of all three methods.
3. Threshold Evaluation:
   - Recall@10 = 0: The union of Recall@10 failures across BM25, Dense, and Hybrid yielded only 11 unique queries (8 for BM25, 3 for Dense, 2 for Hybrid; 13 total query-method instances). Because 11 < 20, the threshold was deterministically widened.
   - Recall@5 = 0: Widening the criterion to Recall@5 = 0 produced 33 unique failing queries across 42 (query, method) failure instances (28 for BM25, 6 for Dense, 8 for Hybrid).
   - Final Decision: Because 33 >= 20, the widening process stopped at Recall@5 = 0.
4. No Arbitrary Capping: Rather than truncating at exactly 20 cases, all 42 failure instances across all 33 unique queries were extracted into results/failure_cases_raw.jsonl and analyzed.

Every step of this selection is automated and verifiable by executing:
```bash
python -m src.failure_analysis
```

---

## 2. Derived Failure Taxonomy

Qualitative analysis of all 42 failure records revealed four distinct recurring failure mechanisms. Every failure case has been mapped to exactly one category in results/failure_taxonomy_mapping.json.

### 2.1 Multi-Hop Bridging Failures (Second-Hop Invisibility)

- Definition: The query requires multi-hop reasoning across two documents, but the second gold document is completely unmentioned in the query text and cannot be matched directly without first identifying and reading the primary bridge document.
- Frequency: 14 cases (33.33% of all failure instances).
- Methods Affected: BM25 (10 cases), Hybrid (3 cases), Dense (1 case).
- Representative Examples:
  - Query ID: 5abc19705542993a06baf86e (BM25, Hybrid)
    - Question: "Black Book starred the actress and writer of what heritage?"
    - Gold Documents: "Black Book (film)" and "Halina Reijn"
    - Evidence: The actress "Halina Reijn" is never named in the query. While "Black Book (film)" can be retrieved from keyword overlap, single-step retrievers cannot bridge to "Halina Reijn" without first reading the cast list of "Black Book". BM25 placed "Black Book (film)" at rank 238 and "Halina Reijn" at rank 85.
  - Query ID: 5a7344e95542991f9a20c6ce (Dense, BM25, Hybrid)
    - Question: "What song was number 4 on the charts when a song from FutureSex/LoveSounds was number 1?"
    - Gold Documents: "Rudebox (song)" and "SexyBack"
    - Evidence: Neither song title appears in the prompt. The retriever must infer that the number 1 song from the album was "SexyBack", check chart history, and locate "Rudebox". Both Dense and BM25 failed to place either document in the top 5, causing Hybrid to fail as well.
  - Query ID: 5a8f503c5542992414482a34 (BM25)
    - Question: "The Prodigal Daughter, though a book about an American, is by a novelist of what nationality?"
    - Gold Documents: "The Prodigal Daughter" and "Jeffrey Archer"
    - Evidence: The author "Jeffrey Archer" is omitted from the prompt. BM25 placed "The Prodigal Daughter" at rank 6 and "Jeffrey Archer" at rank 1599.
- Analytical Reasoning: This is the defining challenge of multi-hop question answering. Standard dense and sparse retrievers score individual documents in isolation against the raw query. When a target document has zero lexical overlap and minimal semantic affinity with the ungrounded query, single-step retrieval fundamentally breaks down.

---

### 2.2 Entity Confusion and Partial Name Overlap

- Definition: A distractor document contains partial, substring, or homonym overlap with entity names in the query (such as shared media titles, namesake locations, or memoirs), causing the distractor to outrank the true encyclopedic document.
- Frequency: 12 cases (28.57% of all failure instances).
- Methods Affected: BM25 (9 cases), Hybrid (3 cases).
- Representative Examples:
  - Query ID: 5adde73f5542992200553b94 (BM25, Hybrid)
    - Question: "What do both Spider-Man and Spider-Man in film have in comman"
    - Gold Documents: "Spider-Man (pinball)" and "Spider-Man in film"
    - Evidence: BM25 retrieved "Spider-Man (2002 video game)" (rank 1), "Lego Spider-Man" (rank 2), and "Spider-Man 2" (rank 3) before either gold document.
  - Query ID: 5ac3e0f7554299194317388b (BM25)
    - Question: "Which actor does American Beauty and American Beauty have in common?"
    - Gold Documents: "American Beauty (soundtrack)" and "American Beauty (1999 film)"
    - Evidence: BM25 retrieved the 1927 silent film "American Beauty (1927 film)" at rank 2, pushing the 1999 film to rank 8 and soundtrack to rank 10.
  - Query ID: 5ae32e125542991a06ce9946 (BM25, Hybrid)
    - Question: "According to the 2001 census, what was the population of the city in which Kirton End is located?"
    - Gold Documents: "Kirton End" and "Boston, Lincolnshire"
    - Evidence: The query keyword "Kirton" matched distractor documents "Kirton, Nottinghamshire" (rank 1) and "Kirton, Suffolk" (rank 3), while "Kirton End" was pushed to rank 30 in BM25.
  - Query ID: 5a8c493e554299653c1aa020 (BM25, Hybrid)
    - Question: "John ruskin named his album due to a removal of what?"
    - Gold Documents: "I Gotta Rash/We Are Thee Goblins from Canada" and "Nardwuar the Human Serviette"
    - Evidence: The query refers to musician Nardwuar (legal name John Ruskin), but BM25 matched on the historical Victorian art critic, ranking "John Ruskin (painting)" at rank 1 and "The Passion of John Ruskin" at rank 2.
- Analytical Reasoning: Sparse retrieval relies on term frequency and document length normalization. When multiple documents share prominent proper nouns, BM25 cannot distinguish the specific entity intended by the context, allowing high-frequency name matches to crowd the top ranks.

---

### 2.3 Topical Distraction and Semantic Drift

- Definition: Candidate distractor documents in the corpus share the same broad subject matter, league, franchise, or entity domain as the query, clustering at high similarity scores and crowding out the specific target passage.
- Frequency: 10 cases (23.81% of all failure instances).
- Methods Affected: BM25 (5 cases), Dense (4 cases), Hybrid (1 case).
- Representative Examples:
  - Query ID: 5a7f54665542992097ad2f1a (Dense)
    - Question: "The city that hosted the Olympics where Taiwan competed as Chinese Taipei is situated at the confluence of what two rivers?"
    - Gold Documents: "Chinese Taipei at the 1988 Winter Olympics" and "Calgary"
    - Evidence: Dense retrieval returned "Chinese Taipei at the 2008 Summer Olympics" (rank 1), "Chinese Taipei at the 2006 Winter Olympics" (rank 2), "Chinese Taipei at the 2006 Asian Games" (rank 3), and "Chinese Taipei at the 2010 Winter Olympics" (rank 4), crowding out the 1988 Winter Olympics to rank 6.
  - Query ID: 5a809e9f5542996402f6a5b1 (Dense)
    - Question: "Ricky Martin's concert tour in 1999 featured an American heavy metal band formed in what year?"
    - Gold Documents: "Livin la Vida Loco" and "Machine Head (band)"
    - Evidence: Dense retrieval populated top ranks with alternative Ricky Martin tours: "Ricky Martin Live" (rank 1), "Livin' la Vida Loca Tour" (rank 2), and "Musica + Alma + Sexo World Tour" (rank 3), pushing the gold tour document down.
  - Query ID: 5a87c13f5542996e4f30890c (Dense)
    - Question: "In what city did the 'Prince of tenors' star in a film based on an opera by Giacomo Puccini?"
    - Gold Documents: "Tosca (1956 film)" and "Franco Corelli"
    - Evidence: Dense retrieval matched on the composer Giacomo Puccini, retrieving "Festival Puccini" (rank 1), "Cavalleria rusticana" (rank 2), and "Gianni Schicchi" (rank 3).
  - Query ID: 5ab5c263554299488d4d9a18 (BM25)
    - Question: "Which country refrained from participating in the 1991 Baltic Cup though it had participated in previous Baltic Cup competitions?"
    - Gold Documents: "Estonia national football team 1991" and "Baltic Cup (football)"
    - Evidence: BM25 retrieved competing annual tournament pages: "2001 Baltic Cup" (rank 1), "1995 Baltic Cup" (rank 2), and "1992 Baltic Cup" (rank 3).
- Analytical Reasoning: In a closed candidate pool (such as HotpotQA's distractor setting), negative documents are chosen specifically to be topical distractors. Dense sentence embeddings capture general semantic relatedness strongly; consequently, passages describing sibling events in the same series receive virtually identical cosine similarities.

---

### 2.4 Lexical and Syntactic Mismatch

- Definition: The query describes a concept using paraphrasing, informal aliases, typographical spelling errors, or phonetic transcriptions that diverge from the formal vocabulary in the target document, preventing exact lexical matching.
- Frequency: 6 cases (14.29% of all failure instances).
- Methods Affected: BM25 (4 cases), Dense (1 case), Hybrid (1 case).
- Representative Examples:
  - Query ID: 5ae7cea355429952e35ea9c1 (Dense, Hybrid)
    - Question: "What is the name of the series who's first season was released on Netflix on September 22, 2016 and stars actor pronounced as John-a-kite?"
    - Gold Documents: "Evan Jonigkeit" and "Easy (TV series)"
    - Evidence: The query provides a phonetic transcription ("John-a-kite") for actor "Evan Jonigkeit". Neither dense nor sparse models can reliably connect this phonetic description to the spelling "Jonigkeit", causing Dense to rank Evan Jonigkeit at rank 37 and Easy at rank 21.
  - Query ID: 5adbcc085542996e6852523c (BM25)
    - Question: "What are some foods that may have been served at the Hawaiin Cottage?"
    - Gold Documents: "Hawaiian Cottage" and "Luau"
    - Evidence: The query misspelled "Hawaiian" as "Hawaiin". BM25 exact matching completely failed on the entity title, relegating "Hawaiian Cottage" to rank 12 and "Luau" to rank 259.
  - Query ID: 5ab31864554299233954ff06 (BM25)
    - Question: "What class of instrument does Apatim Majumdar play?"
    - Gold Documents: "Apratim Majumdar" and "Sarod"
    - Evidence: Typo in the query ("Apatim" instead of "Apratim"). BM25 ranked "Apratim Majumdar" at rank 9 and "Sarod" at rank 15.
- Analytical Reasoning: Exact token matching algorithms have zero tolerance for character-level deviations or phonetic descriptions. Without subword n-gram fuzzy matching or phonetic normalization, any minor typo in a primary keyword degrades BM25 scoring dramatically.

---

## 3. Comparison to Pre-Registered Hypothesis H2

RESEARCH_PLAN.md formulated Hypothesis 2 prior to running experiments, predicting four failure categories:
1. Lexical mismatch
2. Multi-hop bridging failure
3. Semantic drift / topical distraction
4. Entity confusion

### Hypothesis Verification:
- All four hypothesized failure categories were confirmed by the empirical evidence.
- Distribution of Observed Failures:
  - Multi-hop bridging failure proved to be the largest category (33.33%), confirming the hypothesis that neither BM25 nor Dense can effectively reason across hops from the query text alone.
  - Entity confusion (28.57%) and topical distraction (23.81%) accounted for more than half of all failure cases combined.
  - Lexical mismatch accounted for 14.29% of cases.

### Unanticipated Emergent Findings:
1. Complete Second-Hop Invisibility: While H2 anticipated that second-hop documents would look "weakly related," the empirical data showed that in multiple instances (e.g. 5abc19705542993a06baf86e), the second gold document title and text share literally zero substantive tokens with the query.
2. User Typographical and Phonetic Noise: H2 assumed lexical mismatch would stem from synonymy or abstract paraphrasing. In practice, typographical misspellings ("Hawaiin", "Apatim") and intentional phonetic spelling ("John-a-kite") were primary drivers of severe sparse retrieval breakdown.
3. Dense Vulnerability to Series Distractors: Dense retrieval's failures were almost exclusively concentrated in Topical Distraction (e.g. sibling Olympic competitions or concert tours) where contextual embeddings could not discriminate between fine-grained temporal qualifiers (1988 vs. 2008).

---

## 4. Summary Table

The table below summarizes the failure categories, instance counts, percentages, and affected retrieval methods across all 42 failure cases:

| Category | Instance Count | Percentage | Methods Affected | Primary Mechanism |
| :--- | :---: | :---: | :--- | :--- |
| Multi-Hop Bridging Failure | 14 | 33.33% | BM25, Dense, Hybrid | Second-hop gold document is unmentioned in prompt and invisible to single-step retrieval |
| Entity Confusion & Partial Overlap | 12 | 28.57% | BM25, Hybrid | Substrings, media adaptations, and namesake locations outrank target entity |
| Topical Distraction & Semantic Drift | 10 | 23.81% | BM25, Dense, Hybrid | Clustered sibling entries (tour dates, Olympic years, genres) crowd top ranks |
| Lexical & Syntactic Mismatch | 6 | 14.29% | BM25, Dense, Hybrid | Typos, phonetic hints, and descriptive paraphrasing evade exact token matching |
| **Total** | **42** | **100.0%** | All | 42 failure instances across 33 unique queries |
