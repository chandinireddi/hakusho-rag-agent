# Evaluation report

> `gpt-4o-mini` · `text-embedding-3-small` · 345 chunks · 17 questions · 2026-11-16

**Retrieval** (correct document and page)

| Retriever | DocHit@1 | Hit@1 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| v1 (whole-document BM25) | 0.87 | – | – | – |
| BM25 (page chunks) | 1.00 | 0.93 | 1.00 | 0.96 |

**Cross-lingual retrieval** (the same 15 questions typed in English, searched as-is, without the agent's translation step)

| Retriever | DocHit@1 | Hit@1 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| BM25 (page chunks) | 0.53 | 0.13 | 0.27 | 0.18 |

**Number-error detection** — easy: 21 sentences with one digit changed · hard: 13 sentences where the number is swapped for another number from the same page

| Verifier | Precision | Recall | F1 | False alarm | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|
| Rule-based number check only | 1.00 | 1.00 | 1.00 | 0.00 | 21 / 0 / 0 / 21 |
| Rule-based number check only · hard | 0.00 | 0.00 | 0.00 | 0.00 | 0 / 0 / 13 / 13 |

_Hybrid search, LLM judge and full-agent results appear after running with an API key._
