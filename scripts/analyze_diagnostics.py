"""Summarize fresh expanded-order or adjacent-swap outputs on their frozen support."""

from pathlib import Path
import argparse, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.common import ROOT, STEMS, summarize
from src.expanded_analysis import query_states
from src.adjacent_analysis import summarize_queries, query_means


def expanded(folder):
    rows = []
    for model in ["minilm", "bge_base"]:
        for ds in ["wands", "esci"]:
            frame = pd.read_parquet(folder / model / f"{ds}_ranks.parquet")
            families = (
                [
                    ("7", frame.schedule_index < 7),
                    ("16", frame.schedule_index < 16),
                    ("32", frame.schedule_index < 32),
                ]
                if ds == "wands"
                else [
                    ("7", frame.schedule.isin(STEMS)),
                    ("exhaustive_target", frame.schedule.str.startswith("E")),
                ]
            )
            for scope in ["full", "common_all_fitting"]:
                support = (
                    np.ones(len(frame), dtype=bool)
                    if scope == "full"
                    else frame.fully_fitting
                )
                for family, mask in families:
                    for k in [20, 100]:
                        _, query = query_states(frame[mask & support], k)
                        rows.extend(
                            summarize(
                                query,
                                ["persistent_inclusion", "VI", "persistent_omission"],
                                dict(
                                    dataset=ds,
                                    model=model,
                                    support=scope,
                                    family=family,
                                    K=k,
                                ),
                            )
                        )
    return (
        pd.DataFrame(rows),
        ["dataset", "model", "support", "family", "K", "metric"],
        "harmonized_summary.csv",
    )


def adjacent(folder):
    rows = []
    for model in ["minilm", "bge_base"]:
        frame = pd.read_parquet(folder / "data" / f"{model}_variants.parquet")
        for (ds, qid, pid), group in frame.groupby(
            ["dataset", "query_id", "product_id"]
        ):
            for k in [20, 100]:
                flips = group[f"flip_{k}"]
                confirmed = group[f"confirmed_{k}"]
                inside = bool(group[f"c0_included_{k}"].iloc[0])
                affected = bool(flips.any())
                confirmed_any = bool(confirmed.any())
                rows.append(
                    dict(
                        dataset=ds,
                        model=model,
                        query_id=qid,
                        product_id=pid,
                        K=k,
                        distinct_swaps=len(group),
                        A=float(affected),
                        F=float(flips.mean()),
                        loss=float(affected and inside),
                        gain=float(affected and not inside),
                        baseline_inclusion=float(inside),
                        A_confirmed=float(confirmed_any),
                        F_confirmed=float(confirmed.mean()),
                        loss_confirmed=float(confirmed_any and inside),
                        gain_confirmed=float(confirmed_any and not inside),
                    )
                )
    estimates = []
    for (ds, model, k), g in pd.DataFrame(rows).groupby(["dataset", "model", "K"]):
        estimates.extend(
            summarize_queries(query_means(g), dict(dataset=ds, model=model, K=k))
        )
    return (
        pd.DataFrame(estimates),
        ["dataset", "model", "K", "metric"],
        "adjacent_estimates.csv",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["expanded", "adjacent"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run.resolve()
    actual, keys, reference = (expanded if args.experiment == "expanded" else adjacent)(
        folder
    )
    actual.to_csv(folder / "summary.csv", index=False)
    expected = pd.read_csv(ROOT / "reference" / reference)
    comparison = actual.merge(
        expected[keys + ["estimate", "ci_low", "ci_high"]],
        on=keys,
        suffixes=("_new", "_reference"),
        validate="one_to_one",
    )
    for c in ["estimate", "ci_low", "ci_high"]:
        comparison[c + "_delta"] = comparison[c + "_new"] - comparison[c + "_reference"]
    comparison.to_csv(folder / "reference_comparison.csv", index=False)
    print("Fresh estimates and reference differences saved to", folder)
