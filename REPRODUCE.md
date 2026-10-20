# Reproduction instructions

Run commands from the package root. The supplied catalogs and rank evidence are complete for the evaluated populations; restoring the original repository's backup releases is unnecessary.

## 1. Recompute the manuscript statistics

Install `requirements-analysis.txt`, then run:

```bash
python scripts/verify_package.py
python -m unittest discover -s tests -v
python scripts/reproduce_paper.py
```

`outputs/reproduced/verification.json` reports each reference comparison. Computation starts from the retained ranks, not rounded manuscript values. The analysis also reconstructs product-identity transitions and checks the saved per-query reference records. Figure 2 and Table 1 are conceptual definitions, not additional model runs.

## 2. Fresh catalog and target-only retrieval

Create a separate Python 3.10 environment. The GTE implementation needs the newer Transformers version in this environment.

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
python scripts/download_models.py
python scripts/run_retrieval.py --dataset wands --model minilm --smoke --output outputs/smoke
python scripts/run_retrieval.py --dataset wands --model minilm --output outputs/wands_minilm
python scripts/run_retrieval.py --dataset wands --model bge_base --output outputs/wands_bge
python scripts/run_retrieval.py --dataset wands --model gte_modernbert --output outputs/wands_gte
python scripts/run_retrieval.py --dataset esci --model minilm --output outputs/esci_minilm
python scripts/run_retrieval.py --dataset esci --model bge_base --output outputs/esci_bge
python scripts/run_retrieval.py --dataset esci --model gte_modernbert --output outputs/esci_gte
```

Use a new output directory for each run. The smoke option encodes four actual inputs; it does not estimate benchmark performance. Run GPU jobs sequentially. `--batch-size` can lower memory use; the selected value is recorded in `run.json`.

Each full run saves all-judged catalog ranks, source-order and alternative target ranks, token-length eligibility, model revisions, and execution metadata. MiniLM/BGE runs also evaluate the three canonical rules, BM25, full-list RRF, Set-Mean, and two-view centroid/max controls. To inspect fresh effectiveness and compare it with the paper reference:

```bash
python scripts/evaluate_rerun.py --run outputs/wands_minilm
```

The native encoder implementation is retained from the original code. Historical results reused different caches and execution profiles. Hardware, software, query batching, and padding can change floating-point results near a cutoff. The supplied reference ranks remain the authoritative source for exact manuscript-statistic reproduction.

## 3. Expanded order families

Use a separate Python 3.10 environment with the diagnostic requirements:

```bash
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-diagnostics.txt
python scripts/download_models.py --models minilm bge_base
python experiments/expanded/run.py --model minilm
python experiments/expanded/run.py --model bge_base
python scripts/analyze_diagnostics.py expanded --run outputs/expanded/harmonized
```

The runners read the frozen 128-query WANDS cohort, all eligible ESCI pairs, order plan, and tokenizer-input hashes. They jointly encode source-order competitors, queries, and all target orders under one profile. Outputs are resumable under `outputs/expanded/harmonized/`. A changed profile is rejected in an existing run directory. Use a separate package copy for another profile.

The supplied reference results use WANDS families of 7, 16, and 32 schedules and exhaustive distinct target orders for ESCI. Fully-fitting supports are fixed across family sizes. Historical and harmonized ranks are compared explicitly, never pooled.

## 4. Adjacent-swap inference

Use the same diagnostic environment. The code verifies pinned software and model snapshots. The original and current GPU names are recorded separately; a different GPU is not treated as the original execution.

```bash
python experiments/adjacent/rebuild_reference.py --model minilm
python experiments/adjacent/infer.py --model minilm --reference-dir outputs/adjacent/cache/rebuilt_reference_minilm
python experiments/adjacent/rebuild_reference.py --model bge_base
python experiments/adjacent/infer.py --model bge_base --reference-dir outputs/adjacent/cache/rebuilt_reference_bge_base
python scripts/analyze_diagnostics.py adjacent --run outputs/adjacent
```

These commands reconstruct C0 competitors and queries from the supplied catalogs, then evaluate the frozen serialized swap plan. All observed cutoff crossings receive three additional forward checks. Original and new results are kept separately. The saved-rank reproduction in step 1 uses the original complete outcomes and confirmation records.

## 5. French-molding Integrated Gradients

In the diagnostic environment:

```bash
python scripts/run_attribution.py --output outputs/french_molding
```

This reruns the paper's PAD-reference C0/C1 diagnostic using the supplied fixed query embedding and native boundary records. It retains FP16-rounded model parameters converted to FP32 for gradients, unchanged special-token embeddings, Gauss-Legendre integration, completeness checks, and inclusion-boundary guards. Failures produce a nonzero exit code. This reference-dependent case diagnostic is not a causal attribution estimate or a population prevalence result.

## 6. Optional reconstruction from upstream raw data

The primary package already includes the exact processed product records and labels. To audit preprocessing independently, obtain the pinned upstream snapshots listed in `DATA.md`, then run:

```bash
python scripts/prepare_data.py --dataset wands --raw /path/to/WANDS/dataset --output outputs/prepared_wands
python scripts/prepare_data.py --dataset esci --raw /path/to/esci-data/shopping_queries_dataset --output outputs/prepared_esci
```

The preparation script compares product records, query text, and cleaned labels with the supplied dataset. Gzip container timestamps and historical auxiliary columns are not compared. It writes to a new directory and leaves package data intact.
