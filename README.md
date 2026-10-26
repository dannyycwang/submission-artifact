# Same Product, Different Visibility

Reproducibility artifact for **Same Product, Different Visibility: Attribute-Order Sensitivity in Dense Retrieval**. This distribution follows the final 12-page manuscript, including Appendices A-D.

## Reproduce the paper

Use Python 3.10 and run from this directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
python scripts/verify_package.py
python -m unittest discover -s tests -v
python scripts/reproduce_paper.py
```

This route recomputes the statistics from retained ranks and product identities. It needs no GPU, model download, training, or paid service. It checks the original full-precision references, 976 additional-model estimates and intervals, GTE canonical comparisons, and the values printed in the final manuscript. Reference comparisons use an absolute tolerance of `1e-12`; printed values use their stated rounding precision.

Start with `outputs/reproduced/verification.json`, `tables/table3.csv`, and `figures/`. Figures use the paper's evidence with a simplified layout. Figure 2 and Table 1 are definitions, not additional numerical experiments.

## Evaluated models

| Encoder | Checkpoint | Input limit | Native pooling |
|---|---|---:|---|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 256 | Masked mean |
| BGE | `BAAI/bge-base-en-v1.5` | 512 | CLS |
| GTE | `Alibaba-NLP/gte-modernbert-base` | 8,192 | CLS |
| EmbeddingGemma 2 | `google/embeddinggemma-2` | 8,192 | Native projection, then masked mean |
| Harrier 0.6B | `microsoft/harrier-oss-v1-0.6b` | 32,768 | Last nonpadding token |

All five cover seven-order catalog-wide and target-only retrieval, full/fully-fitting analyses, and three canonical dense rules. The broader controls, expanded-order study, and adjacent swaps use MiniLM/BGE. Exact model revisions are in `configs/models.json`; native prompts, BF16 execution, dimensions, batching, and software for the two additional models are in `configs/recent_models.json`.

## Contents

| Path | Purpose |
|---|---|
| `configs/` | Frozen supports, model revisions, seeds, gains, and execution settings |
| `data/catalogs/` | Complete evaluated catalogs, queries, and cleaned relevance judgments |
| `data/ranks/` | All-judged catalog ranks for 154 retained conditions |
| `data/target_only/` | Ten seven-order target/catalog rank files, one per dataset and encoder |
| `data/expanded/`, `data/adjacent/` | Frozen designs, complete outcomes, and numerical checks |
| `data/boundary/` | Three-pass boundary checks for the two additional encoders |
| `data/cases/` | Teal-chair and wooden-chair evidence |
| `src/`, `scripts/`, `experiments/` | Statistical analysis and portable inference code |
| `reference/` | Full-precision result references and printed manuscript values |
| `provenance/`, `licenses/`, `validation/` | Source transformations, dataset notices, and completed checks |

[PAPER_MAP.md](PAPER_MAP.md) maps each figure and table to its evidence. [DATA.md](DATA.md) documents schemas and denominators. [REPRODUCE.md](REPRODUCE.md) gives fresh inference commands and environment requirements. [SCOPE.md](SCOPE.md) describes the retained scope and adaptations.

WANDS uses 42,994 products, 308 eligible held-out queries, and 21,299 highest-label pairs. ESCI uses the frozen 10,076-product union, 499 eligible queries, and 4,434 highest-label pairs. Raw product text and benchmark labels are unchanged. Model weights and regenerable embeddings are obtained separately.
