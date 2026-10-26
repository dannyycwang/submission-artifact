"""Run the frozen EmbeddingGemma 2 or Harrier retrieval profile."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["embeddinggemma2", "harrier06b"], required=True
    )
    parser.add_argument("--dataset", choices=["wands", "esci", "both"], default="both")
    parser.add_argument(
        "--stage",
        choices=[
            "download",
            "smoke",
            "encode",
            "evaluate",
            "canonical",
            "boundary",
            "analyze",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/recent_models"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Optional local snapshot; all pinned input-file hashes are checked.",
    )
    args = parser.parse_args()
    os.environ["ARTIFACT_RECENT_MODEL"] = args.model
    os.environ["ARTIFACT_RECENT_OUTPUT"] = str(args.output.resolve())
    from common import ROOT, HERE, protocol, initialize, read, sha

    spec = protocol()["models"][args.model]
    needs_model = args.stage in [
        "download",
        "smoke",
        "encode",
        "canonical",
        "boundary",
        "all",
    ]
    if needs_model:
        if args.model_dir is None:
            from huggingface_hub import snapshot_download

            model_dir = Path(
                snapshot_download(repo_id=spec["name"], revision=spec["revision"])
            )
        else:
            model_dir = args.model_dir.resolve()
        for name, expected in read(ROOT / "configs/recent_model_files.json")[
            args.model
        ].items():
            path = model_dir / name
            assert (
                path.is_file() and sha(path) == expected
            ), f"Pinned model file mismatch: {name}"
        os.environ["ARTIFACT_MODEL_DIR"] = str(model_dir)
        print(
            "Verified pinned model/tokenizer files:",
            args.model,
            spec["revision"],
            flush=True,
        )
    if args.stage == "download":
        return
    initialize()
    datasets = ["wands", "esci"] if args.dataset == "both" else [args.dataset]
    if args.stage in ["smoke", "all"]:
        import smoke

        assert smoke.run(args.model) == 0, "Native/reference validation failed."
    if args.stage in ["encode", "all"]:
        import encode

        for ds in datasets:
            encode.run(args.model, ds)
    if args.stage in ["evaluate", "all"]:
        import evaluate

        for ds in datasets:
            evaluate.run(args.model, ds)
    if args.stage in ["canonical", "all"]:
        import encode, evaluate

        for ds in datasets:
            encode.run(args.model, ds, canonical=True)
            evaluate.run(args.model, ds, canonical=True)
    if args.stage in ["boundary", "all"]:
        import boundary

        for ds in datasets:
            boundary.run(args.model, ds)
    if args.stage in ["analyze", "all"]:
        import analyze

        analyze.run()
        import pandas as pd

        columns = [
            "dataset",
            "model",
            "kind",
            "schedule",
            "K",
            "support",
            "metric",
            "raw_state",
            "canonical_state",
        ]
        actual = pd.read_csv(HERE / "tables/summary.csv")
        reference = pd.read_csv(ROOT / "reference/recent_models.csv")
        for column in columns:
            if column not in actual:
                actual[column] = float("nan")
        result = actual.merge(
            reference[columns + ["estimate", "ci_low", "ci_high"]],
            on=columns,
            suffixes=("_rerun", "_reference"),
            validate="one_to_one",
        )
        for column in ["estimate", "ci_low", "ci_high"]:
            result[column + "_difference"] = (
                result[column + "_rerun"] - result[column + "_reference"]
            )
        result.to_csv(HERE / "tables/reference_comparison.csv", index=False)
        print(
            "Saved fresh estimates and differences from archived evidence.", flush=True
        )


if __name__ == "__main__":
    main()
