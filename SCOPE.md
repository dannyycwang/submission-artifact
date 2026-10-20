# Package scope

The supplied PDF is authoritative for this package. Its title is **Same Product, Different Visibility: Attribute-Order Sensitivity in Dense Retrieval**, even though the uploaded filename contains an earlier title. The original repository's README and manuscript sources describe older experiments and are not used as the current scope.

Retained: primary attribute-order interventions; full/fully-fitting supports; exact Recall cancellation; graded effectiveness; pure canonical rules; Set-Mean; two-view centroid and max retrieval; BM25 and RRF; expanded ordering-family coverage; adjacent swaps; and the three manuscript case studies.

Excluded: Phase-I presentation variants, sentence/section/JSON perturbations outside the primary family, query taxonomy experiments, generative rewriting, fine-tuning/PI-FT, learned set architectures, unrelated reranker studies, extra query samples, view-count sweeps beyond M=2, unused attribution cases/baselines, drafts, reviewer-response history, author prompts, Chinese commentary, presentation assets, failed export copies, logs, local environments, and regenerable embedding caches.

## Code adaptations

- Native serialization, encoding, lexical scoring, rank insertion, bootstrap estimators, and diagnostic inference are retained from the relevant research implementations.
- Unused sentence/section/JSON perturbations and chunk-encoding branches are removed.
- A new portable driver connects the retained representation and retrieval functions to the supplied catalogs. It provides separate fresh-inference outputs and records model/software settings.
- The statistical runner independently reconstructs all-judged effectiveness and identity transitions, then compares them with retained full-precision references.
- Expanded-order and adjacent-swap runners use relative paths and a new output root. Obsolete editorial-checkpoint dependencies are removed. The adjacent runner records the current GPU name instead of requiring the author's GPU name; pinned software and model checks remain enforced.
- The French-molding runner retains the paper's PAD-reference computation and reads the supplied fixed query embedding/native boundary records. Zero-reference and other unpublished attribution experiments are excluded.
- Obsolete numeric grade and query-taxonomy columns are removed from curated catalog/rank tables. Original text, categorical labels, IDs, rankings, scores, and evaluation supports are preserved.

Fresh model runs do not overwrite reported evidence. Their floating-point execution may differ from historical caches, so the package distinguishes exact statistical reproduction from a new inference run. No claims about bitwise reproduction of all original GPU experiments are made.

Author-identifying repository metadata, local user paths, and unrelated workspace records are omitted for double-anonymous review. Reproducibility is checked using the package manifest, frozen data and model revisions, and recorded numerical execution settings.
