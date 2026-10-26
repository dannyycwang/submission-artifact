# Data and provenance

## Sources and licenses

| Dataset | Upstream snapshot | Evaluated scope |
|---|---|---|
| [WANDS](https://github.com/wayfair/WANDS/tree/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5) | `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5` | 42,994 products; full original 480-query input; fixed held-out evaluation support |
| [Amazon Shopping Queries / ESCI](https://github.com/amazon-science/esci-data/tree/7916cdf6ab75a462e77f20ab40428a10923998d5) | `7916cdf6ab75a462e77f20ab40428a10923998d5` | US, small-version test split; 500 hash-selected queries; union of their 10,076 judged products |

Upstream dataset licenses and notices are included in `licenses/`. No new license is assigned to the research code by this packaging operation.

WANDS holds out 384 queries after a 96-query development split. The primary inclusion/Recall support contains the 308 held-out queries with an Exact label, totaling 21,299 pairs. ESCI has 499 eligible queries and 4,434 E-labeled pairs. Exact query IDs are in `configs/evaluation.json`.

WANDS preprocessing removes every query-product pair with contradictory labels and removes duplicate annotations. ESCI sampling sorts query IDs by `SHA256("20260904:" + query_id)` and retains the first 500. Catalog order is fixed and is the tie-breaking order. The product JSONL records preserve attribute duplicates and raw entry contents.

## Schemas

- `data/catalogs/*_products.jsonl.gz`: one product record per line; `product_id`, `title`, `class`, `category`, `description`, `attributes`, `attribute_separator`, and `section_order`. List order is the fixed catalog index.
- `data/catalogs/*_queries.csv`: query ID and query text. The input includes queries outside the primary metric support; filtering uses the frozen IDs in configuration.
- `data/catalogs/*_judgments.csv`: query ID, product ID, and original categorical relevance label.
- `data/conditions.json`: method, dataset, model, schedule, and relative rank file. `original_source` is provenance, not an input dependency.
- `data/ranks/`: one row per judged query-product pair, with one-based full-catalog rank and saved FP32 score where available. All graded calculations map the categorical label through the paper's gains; obsolete grade or query-taxonomy columns are removed.
- `data/target_only/`: highest-label pairs with columns `C0`, `C1`, `C2s1`-`C2s5` and corresponding `whole_*` catalog-wide ranks. Only identities and the fourteen target/catalog rank columns are required for analysis.
- `data/token_lengths.csv`: complete-input lengths and the all-seven fully-fitting indicator for five encoders. Additional-model lengths include native prompts and special tokens; their overhead is not represented as a fixed additive count.
- `data/boundary/`: selected boundary witnesses, three repeated decisions, and numerical audit summaries for EmbeddingGemma 2 and Harrier.
- `data/serialization/`: unique seven-schedule serialization counts used by the additional-model evaluator.
- `data/expanded/`: long-form harmonized target ranks, order plan, and fixed support. `fully_fitting` requires every expanded-family target input to fit.
- `data/adjacent/`: outcome-independent frozen cohort/serialized plan, one row per distinct swap, and repeated numerical checks. All original observed outcomes are retained.
- `data/reference_*.parquet`: exact per-query reference records. Reproduction independently reconstructs their metrics, inclusion states, and transition counts.

## Metrics and units

Highest relevance is `Exact` for WANDS and `E` for ESCI. Direct gains are WANDS `3/1/0` for Exact/Partial/Irrelevant and ESCI `1/0.1/0.01/0` for E/S/C/I. Standard nDCG assigns zero gain to unjudged catalog items; condensed nDCG removes unjudged items before discounting. These conventions are evaluated separately.

Core estimates are decimal fractions. Reader-facing table columns explicitly ending in `_percent` multiply fractions by 100. Differences plotted in percentage points likewise multiply by 100. Raw dense/hybrid effectiveness averages seven schedules within each query before query-macro aggregation. VI measures whether a highest-label pair crosses a cutoff within the tested family. Zero VI includes both persistent inclusion and persistent omission.

Inclusion-state and transition denominators are all eligible highest-label pairs within each query, followed by equal averaging across queries. Canonical transition cells are not conditional loss probabilities. Full and fully-fitting populations differ for MiniLM/BGE and cannot identify a causal truncation effect.

## Case evidence

- **Teal chair:** WANDS query 409, product 24318, GTE; source rank 11, C2s5 rank 939. The original title and every attribute occurrence remain available.
- **Wooden chair outdoor:** WANDS query 367, BGE, C0 versus C1. `all_products.csv` contains all 24 union products and the frame-material classification. MATCH counts are 8 and 6; the same nine unknown/ambiguous products remain in both lists. `matched_scoring.json` supplies exact union ranks from cached FP32 scoring. This case uses the native truncated pipeline.

All supplied English-benchmark product text is preserved without translation or normalization. Formatting, Unicode characters, or unusual spelling inside raw records are part of the scientific inputs.

## Integrity

`MANIFEST.sha256` covers the distributed files. `provenance/source_files.json` maps curated files to original repository paths, source hashes, and transformations. Original paths in provenance identify historical sources and are not required at runtime. Execution profiles preserve numerical settings; machine-specific paths are omitted where they are not scientific inputs.

`reference/recent_models.csv` retains 976 manuscript-related estimates/intervals from the completed additional-model experiments. `reference/gte_canonical.csv` retains the twelve GTE Recall comparisons. `reference/manuscript_values.csv` records rounded cells transcribed from the final manuscript for an independent presentation-level check. Original-model reference files retain their original support and are compared as a strict subset of the final five-model outputs.
