# Validation performed

The final five-model artifact was checked against the manuscript and completed experiment evidence.

- Recomputed 62,139 query-effectiveness records and 67,788 membership records from 154 retained rank conditions.
- Compared original-model estimates and paired 95% intervals with full-precision references at an absolute tolerance of `1e-12`.
- Independently verified 976 additional-model estimates/intervals, twelve GTE canonical comparisons, four boundary-audit cells, and 332 values printed in the final manuscript.
- Reconciled all 40 additional-model score matrices with archived hashes, all-judged metrics, and highest-label ranks before packaging.
- Reconstructed inclusion states, canonical transitions, the expanded-order analysis, and adjacent-swap statistics.
- Verified every additional-model raw/canonical serialized input against original encoding descriptors, in addition to the original seven-schedule and frozen diagnostic input checks.
- Matched preprocessing to the pinned upstream catalogs, queries, and cleaned labels.
- Passed eight CPU regression tests, including cache corruption recovery and independent brute-force catalog/target ranking.
- Passed actual GPU native/reference smoke checks for EmbeddingGemma 2 and Harrier; both explicit pooling paths matched the official encoder exactly at fixed batch sizes. Three repeated forwards produced identical embeddings, scores, and decisions. Batch-size sensitivity is recorded separately.
- Retained the earlier successful MiniLM/GTE native and MiniLM/BGE diagnostic forward checks.
- Ran the complete statistical reproduction in a separate directory containing only distribution files; reviewed the generated plots.

`statistical_reproduction.json`, `final_data_assembly.json`, `input_equivalence.json`, `preprocessing.json`, `inference_checks.json`, `regression_tests.json`, and `diagnostic_analysis.json` record these checks. The diagnostic summarizers also match all 120 expanded-order and 72 adjacent-swap reference estimates.

Full-catalog GPU inference and the complete diagnostic encoding runs were not repeated during packaging. Their completed original evidence is supplied and the statistical analysis was rerun. Smoke tests do not replace benchmark results.

`content_anonymity.json` records a scan of text, compressed catalog contents, and Parquet strings/metadata. Benchmark text is unchanged; archive contents exclude Git history.
