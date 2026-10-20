"""Recompute all numerical evidence without model downloads or GPU inference."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import *
from src import analysis, states, adjacent_analysis, expanded_analysis


def compare(name, keys, columns):
    actual = pd.read_csv(HERE / "tables" / f"{name}.csv").set_index(keys).sort_index()
    expected = (
        pd.read_csv(ROOT / "reference" / f"{name}.csv").set_index(keys).sort_index()
    )
    assert actual.index.equals(expected.index), (name, "row keys")
    error = float(
        np.max(abs(actual[columns].to_numpy() - expected[columns].to_numpy()))
    )
    assert np.allclose(actual[columns], expected[columns], atol=1e-12, rtol=0), (
        name,
        error,
    )
    return dict(table=name, rows=len(actual), max_absolute_error=error, passed=True)


def main():
    for folder in ["data", "tables", "figures", "qa"]:
        (HERE / folder).mkdir(parents=True, exist_ok=True)
    analysis.inclusion()
    pq = analysis.graded()
    states.recompute(pq)
    analysis.membership(pq)
    analysis.controls(pq)
    analysis.transitions()
    expanded_analysis.main()
    adjacent_analysis.main()
    keys = ["dataset", "model"]
    checks = []
    for name, extra, columns in [
        (
            "inclusion_complete",
            ["intervention", "target_support", "K", "metric"],
            ["estimate", "ci_low", "ci_high"],
        ),
        (
            "membership_all_contrasts",
            ["method", "schedule", "K"],
            [
                "gain_share",
                "loss_share",
                "turnover",
                "delta_recall",
                "equal_positive_share_all",
            ],
        ),
        (
            "controls_complete",
            ["method", "schedule", "K", "metric"],
            ["estimate", "ci_low", "ci_high"],
        ),
        (
            "controls_paired_comparisons",
            ["method", "schedule", "K", "metric", "comparator"],
            ["estimate", "ci_low", "ci_high"],
        ),
        (
            "transitions_complete",
            ["family", "rule", "K", "reference_state", "control_state"],
            ["estimate", "ci_low", "ci_high"],
        ),
        (
            "harmonized_summary",
            ["support", "family", "K", "metric"],
            ["estimate", "ci_low", "ci_high"],
        ),
        (
            "harmonized_increments",
            ["support", "K", "metric", "contrast"],
            ["estimate", "ci_low", "ci_high"],
        ),
    ]:
        checks.append(compare(name, keys + extra, columns))
    a = (
        pd.read_csv(HERE / "data/aggregate_estimates.csv")
        .set_index(keys + ["K", "metric"])
        .sort_index()
    )
    b = (
        pd.read_csv(ROOT / "reference/adjacent_estimates.csv")
        .set_index(keys + ["K", "metric"])
        .sort_index()
    )
    assert a.index.equals(b.index)
    assert np.allclose(
        a[["estimate", "ci_low", "ci_high"]],
        b[["estimate", "ci_low", "ci_high"]],
        atol=1e-12,
        rtol=0,
    )
    checks.append(dict(table="adjacent_estimates", rows=len(a), passed=True))
    from src.paper_outputs import build

    checks.extend(build())
    dump(
        HERE / "verification.json",
        dict(
            passed=True,
            checks=checks,
            mode="saved-rank statistical reproduction",
            new_model_inference=False,
        ),
    )
    print("PASS: all numerical comparisons agree with the retained reference evidence.")


if __name__ == "__main__":
    main()
