# Validation performed

The package was checked against the supplied manuscript and original local research artifacts.

- Recomputed manuscript statistics from saved rank records, including 43,578 method/schedule/query effectiveness records, 48,420 membership records, and complete paired control comparisons.
- Compared means and 95% confidence limits with full-precision references at an absolute tolerance of `1e-12`.
- Reconstructed inclusion states and canonical transitions from product identities and checked every retained reference cell.
- Verified the primary serialization for every product in both catalogs under all seven schedules against the original implementation.
- Verified every frozen expanded-order and adjacent-swap serialization hash.
- Rebuilt WANDS and ESCI preprocessing from the pinned local upstream data and matched all retained product records, query text, and cleaned labels.
- Ran six regression tests covering exact target insertion with ties, canonical invariance and duplicates, direct-gain metrics, real query records, Recall cancellation, and lexical fusion.
- Ran actual MiniLM/GTE native-encoder smoke tests and MiniLM/BGE fixed-bucket diagnostic forward tests.
- Reran both PAD-reference French-molding attributions; completeness and boundary guards passed.
- Reviewed the generated plots and checked English source/documentation text, file integrity, and command-line entry points.

Full-catalog GPU inference, the complete expanded-order encoding run, and the complete adjacent-swap inference run were not repeated during packaging. Their completed original evidence is supplied, their analysis was recomputed, and their inference code is included separately.

`statistical_reproduction.json`, `input_equivalence.json`, `preprocessing.json`, `inference_checks.json`, `diagnostic_analysis.json`, and `attribution.json` record the validation results. The fresh-run diagnostic summarizers were also checked against all 120 expanded-order and 72 adjacent-swap reference estimates. No saved reference evidence was replaced by a new model run.
