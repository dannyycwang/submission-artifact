# Scope and adaptations

The final manuscript defines this distribution's scope: five frozen encoders; seven attribute-order schedules; catalog-wide and target-only retrieval; full and fully-fitting supports; exact Recall cancellation; graded effectiveness; three canonical dense rules; MiniLM/BGE Set-Mean, two-view centroid/max, BM25 and RRF controls; expanded ordering families; adjacent swaps; and the two illustrated product cases.

Only these experiments and their supporting data are distributed. Earlier presentation perturbations, unpublished model trials, attribution studies, fine-tuning, extra control sweeps, manuscript drafts, response history, author prompts, progress logs, local environments, model weights, and regenerable embedding/score caches are excluded.

The portable code retains the original serialization, pooling, batching, scoring, rank insertion, and bootstrap calculations. New orchestration reads the supplied catalogs and writes to separate output directories. Original machine paths and workspace bookkeeping are removed. Additional-model score files were checked against archived SHA-256 receipts before extracting compact all-judged ranks; all query metrics and highest-label ranks were reconciled with the completed experiment snapshot. GTE canonical evidence is included for the final five-encoder comparison.

For EmbeddingGemma 2 and Harrier, native preprocessing, prompt text, projection/pooling, BF16 operation order, FP32 normalization/scoring, deterministic batching, and full dimensionality are preserved. The driver checks model-file hashes and software versions, rejects changed profiles in existing caches, and compares its pooling path with the official end-to-end encoder before inference. Harrier was selected after the EmbeddingGemma 2 ESCI results; these additions are not whole-study preregistration.

Fixed canonical representations imply invariance under deterministic retrieval. The recorded repeatability audit establishes cutoff agreement only for sampled raw boundary pairs. Fresh inference can differ across hardware and execution profiles; saved-rank statistical reproduction is distinguished from running the models again.

Author-identifying repository links, source commit identifiers, local user paths, and unrelated workspace records are omitted. Dataset/model attribution and scientific provenance remain. The standalone archive excludes Git history.
