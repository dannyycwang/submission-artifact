# Same Product, Different Visibility

Reproducibility package for **Same Product, Different Visibility: Attribute-Order Sensitivity in Dense Retrieval**.

This package follows the supplied 12-page manuscript, including Appendices A-E. It contains the evaluation catalogs, relevance judgments, frozen query supports, saved ranks, experiment code, and case evidence needed for that manuscript.

## Reproduce the reported results

Use Python 3.10. Run the following commands from this directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
python scripts/verify_package.py
python -m unittest discover -s tests -v
python scripts/reproduce_paper.py
```

The last command recomputes graded effectiveness from all-judged ranks, inclusion states from rank families, gains and losses from product identities, canonical transitions, paired confidence intervals, expanded-order estimates, and adjacent-swap estimates. It checks the results against the supplied reference evidence with an absolute tolerance of `1e-12`.

Results are written to `outputs/reproduced/`. Start with `verification.json`, `tables/table3.csv`, and the plots in `figures/`. The plots use the paper's numerical evidence with a simplified presentation. No GPU, model download, training, or paid service is needed for this route.

## Run the models again

[REPRODUCE.md](REPRODUCE.md) provides the separate environments and commands for fresh catalog retrieval, target-only retrieval, the expanded ordering family, adjacent swaps, and the French-molding attribution diagnostic.

Fresh inference writes to separate output directories. The historical experiments used several execution profiles, so freshly encoded ranks must be compared with the supplied ranks rather than assumed to be bitwise identical. The saved-rank route above reproduces the reported statistics directly.

## Contents

| Directory | Purpose |
|---|---|
| `configs/` | Model revisions, query supports, seeds, gains, and adjacent-swap design |
| `data/catalogs/` | Complete evaluated catalogs, queries, and deduplicated judgments |
| `data/ranks/` | All-judged catalog ranks for 108 retained method/schedule conditions |
| `data/target_only/` | Seven-order target ranks with source-order competitors fixed |
| `data/expanded/` | Harmonized follow-up ranks and frozen order-construction plan |
| `data/adjacent/` | Frozen cohort, serialized swap plan, outcomes, and numerical repeats |
| `data/cases/` | Teal-chair, wooden-chair, and French-molding evidence |
| `src/`, `scripts/`, `experiments/` | Analysis and inference implementations |
| `reference/` | Full-precision published-result references for numerical comparison |
| `provenance/`, `licenses/` | Source mappings, execution profiles, and dataset notices |
| `validation/` | Checks performed when preparing this package |

[PAPER_MAP.md](PAPER_MAP.md) links each paper result to its inputs and outputs. [DATA.md](DATA.md) describes schemas, units, denominators, and dataset provenance. [SCOPE.md](SCOPE.md) explains the retained scope and code adaptations.

## Evaluation scope

- WANDS: 42,994 products; 308 eligible held-out queries; 21,299 highest-label query-product pairs.
- ESCI: the frozen US test sample's 10,076-product union; 499 eligible queries; 4,434 highest-label pairs.
- Primary orders: source, reversal, and five deterministic attribute shuffles.
- Encoders: frozen MiniLM, BGE, and GTE; controls use MiniLM and BGE.
- Confidence intervals: 10,000 paired query-cluster bootstrap resamples; unadjusted 95% percentile limits.

Product text and benchmark relevance labels remain unchanged. Only manuscript-related methods are included. Model weights and regenerable embedding caches are downloaded or generated separately.
