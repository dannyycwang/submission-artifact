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

Each full run saves all-judged catalog ranks, source-order and alternative target ranks, token-length eligibility, model revisions, and execution metadata. All three encoders evaluate the three canonical rules. MiniLM/BGE additionally evaluate BM25, full-list RRF, Set-Mean, and two-view centroid/max controls. To inspect fresh effectiveness and compare it with the paper reference:

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

## 5. EmbeddingGemma 2 and Harrier

Create a separate Python 3.10 environment for these models. Their native implementation requires the pinned Transformers 5 / SentenceTransformers 6 environment, independently of the original-model and diagnostic environments.

```bash
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-recent-models.txt
python experiments/recent_models/run.py --model embeddinggemma2 --stage download
python experiments/recent_models/run.py --model harrier06b --stage download
python experiments/recent_models/run.py --model embeddinggemma2 --stage all
python experiments/recent_models/run.py --model harrier06b --stage all
```

Obtain access to each upstream model under its terms when required. No account token is included in this artifact. A CUDA GPU with BF16 support is required for the frozen inference profile. The original GPU was an RTX 4060 Laptop GPU with 8 GB memory. Full-catalog inference can take substantial time; CPU statistical reproduction is the quick route.

Run GPU jobs sequentially. The default covers both datasets. `--dataset wands` or `--dataset esci` selects one. `--stage all` validates official preprocessing/pooling, encodes shared queries and seven raw catalogs, evaluates both interventions, runs all three canonical rules, repeats selected boundary batches three times, and writes analysis tables. EmbeddingGemma 2 uses batch size 1. Harrier uses the frozen length tiers 8/4/2/1. No silent truncation, quantization, dimension reduction, or OOM batch change is applied.

Outputs are under `outputs/recent_models/<model>/`: `runs/<model>/<dataset>/evaluation/` contains per-query metrics, all-judged ranks, target ranks, and boundary results; `tables/summary.csv` and `tables/reference_comparison.csv` contain fresh estimates and their differences from the original results. Full score matrices, embeddings, and resumable checkpoints are generated locally. They are not part of the distribution.

Individual stages are `smoke`, `encode`, `evaluate`, `canonical`, `boundary`, and `analyze`. Raw evaluation must precede canonical evaluation and boundary checks. The supplied model snapshot can be selected with `--model-dir /path/to/snapshot`; its required files are checked against `configs/recent_model_files.json`. Use `--output outputs/new_profile` for a separate run; changed runtime/profile settings are rejected in an existing output directory.

## 6. Optional reconstruction from upstream raw data

The primary package already includes the exact processed product records and labels. To audit preprocessing independently, obtain the pinned upstream snapshots listed in `DATA.md`, then run:

```bash
python scripts/prepare_data.py --dataset wands --raw /path/to/WANDS/dataset --output outputs/prepared_wands
python scripts/prepare_data.py --dataset esci --raw /path/to/esci-data/shopping_queries_dataset --output outputs/prepared_esci
```

The preparation script compares product records, query text, and cleaned labels with the supplied dataset. Gzip container timestamps and historical auxiliary columns are not compared. It writes to a new directory and leaves package data intact.
