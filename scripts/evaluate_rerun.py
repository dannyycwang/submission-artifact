"""Evaluate fresh rank files and compare them with the manuscript reference."""

from pathlib import Path
import argparse, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.common import ROOT, read
from src.metrics import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = read(ROOT / "configs/evaluation.json")
    rows = []
    for condition in read(run / "conditions.json"):
        ds = condition["dataset"]
        frame = pd.read_parquet(run / condition["rank_source"])
        for qid, g in frame.groupby("query_id"):
            values = {}
            for k in [20, 100]:
                values.update(
                    evaluate(g, cfg["gains"][ds], "Exact" if ds == "wands" else "E", k)
                )
            rows.append(
                {k: condition[k] for k in ["dataset", "model", "method", "schedule"]}
                | dict(query_id=qid, **values)
            )
    actual = pd.DataFrame(rows)
    actual.to_csv(run / "effectiveness_per_query.csv", index=False)
    keys = ["dataset", "model", "method", "schedule", "query_id"]
    columns = [c for c in actual if c not in keys]
    ref = pd.read_parquet(ROOT / "data/reference_per_query.parquet")
    paired = actual.merge(
        ref[keys + columns],
        on=keys,
        suffixes=("_new", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(actual):
        print(
            f"{len(actual)-len(paired)} fresh rows have no manuscript comparator; retained separately."
        )
    for c in columns:
        paired[c + "_delta"] = paired[c + "_new"] - paired[c + "_reference"]
    paired.to_csv(run / "comparison_per_query.csv", index=False)
    summary = (
        paired.groupby(keys[:4])[[c + "_delta" for c in columns]].mean().reset_index()
    )
    summary.to_csv(run / "comparison_summary.csv", index=False)
    targets = pd.read_parquet(run / "target_only.parquet")
    lengths = pd.read_csv(run / "token_lengths.csv", dtype={"product_id": str})
    targets["product_id"] = targets.product_id.astype(str)
    targets = targets.merge(lengths, on="product_id", validate="many_to_one")
    state_rows = []
    for scope in ["full", "fully_fitting"]:
        selected = targets if scope == "full" else targets[targets.fully_fitting]
        for k in [20, 100]:
            inc = selected[cfg["schedules"]].to_numpy() <= k
            values = selected[["query_id", "product_id"]].copy()
            values["persistent_inclusion"] = inc.all(1)
            values["VI"] = inc.any(1) & ~inc.all(1)
            values["persistent_omission"] = ~inc.any(1)
            q = values.groupby("query_id")[
                ["persistent_inclusion", "VI", "persistent_omission"]
            ].mean()
            state_rows.append(
                dict(
                    scope=scope,
                    K=k,
                    queries=len(q),
                    pairs=len(selected),
                    **q.mean().to_dict(),
                )
            )
    pd.DataFrame(state_rows).to_csv(run / "target_inclusion_summary.csv", index=False)
    print("Fresh evaluation and explicit reference differences saved to", run)
