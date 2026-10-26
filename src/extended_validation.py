"""Compare final-manuscript additions with independent archived references."""

from .common import *


def validate(pq):
    reference = pd.read_csv(ROOT / "reference/recent_models.csv")
    inc = pd.read_parquet(HERE / "data/inclusion_per_query.parquet")
    membership = pd.read_parquet(HERE / "data/membership_per_query.parquet")
    transitions = pd.read_parquet(HERE / "data/control_transitions_per_query.parquet")
    errors = []
    for r in reference.itertuples(index=False):
        common = dict(dataset=r.dataset, model=r.model)
        if r.kind in [
            "effectiveness",
            "canonical_effectiveness",
            "canonical_minus_raw_seven_mean",
        ]:
            group = pq[pq.dataset.eq(r.dataset) & pq.model.eq(r.model)]
            metric = (
                "JudgedCoverage" if r.metric == "judged_coverage" else r.metric
            ) + f"@{r.K}"
            raw = group[group.method.eq("raw")]
            mean = raw.groupby("query_id")[metric].mean().sort_index()
            if r.kind == "effectiveness":
                values = (
                    mean
                    if r.schedule == "seven_mean"
                    else raw[raw.schedule.eq(r.schedule)]
                    .set_index("query_id")[metric]
                    .sort_index()
                )
            else:
                values = (
                    group[group.method.eq(r.schedule)]
                    .set_index("query_id")[metric]
                    .sort_index()
                )
                if r.kind == "canonical_minus_raw_seven_mean":
                    assert values.index.equals(mean.index)
                    values = values - mean
        elif r.kind in ["catalog_states", "target_states"]:
            intervention = (
                "catalog_wide" if r.kind == "catalog_states" else "target_only"
            )
            scope = "full" if r.support == "full" else "fully_fitting"
            metric = "VI" if r.metric == "crossing" else r.metric
            group = inc[
                inc.dataset.eq(r.dataset)
                & inc.model.eq(r.model)
                & inc.intervention.eq(intervention)
                & inc.target_support.eq(scope)
                & inc.K.eq(r.K)
            ]
            values = group.set_index("query_id")[metric].sort_index()
        elif r.kind == "paired_C0_contrast":
            mapping = {
                "delta_recall": "delta_recall",
                "delta_ndcg": "delta_ndcg",
                "gain_fraction": "gain_share",
                "loss_fraction": "loss_share",
                "turnover_fraction": "turnover",
                "same_recall_changed_relevant_membership": "exact_replacement",
            }
            group = membership[
                membership.dataset.eq(r.dataset)
                & membership.model.eq(r.model)
                & membership.method.eq("raw")
                & membership.schedule.eq(r.schedule)
                & membership.K.eq(r.K)
            ]
            values = group.set_index("query_id")[mapping[r.metric]].sort_index()
        elif r.kind == "transition":
            group = transitions[
                transitions.dataset.eq(r.dataset)
                & transitions.model.eq(r.model)
                & transitions.family.eq("dense")
                & transitions.rule.eq(r.schedule)
                & transitions.K.eq(r.K)
                & transitions.reference_state.eq(r.raw_state)
                & transitions.control_state.eq(r.canonical_state)
            ]
            values = group.set_index("query_id").cell_fraction.sort_index()
        else:
            raise ValueError("Unexpected archived metric kind: " + r.kind)
        assert len(values) == r.queries and not values.index.duplicated().any()
        actual = np.array(ci(values.to_numpy(float), values.index))
        expected = np.array([r.estimate, r.ci_low, r.ci_high])
        error = float(abs(actual - expected).max())
        assert np.allclose(actual, expected, atol=1e-12, rtol=0), (r, actual, error)
        errors.append(error)

    gte = pd.read_csv(ROOT / "reference/gte_canonical.csv")
    controls = pd.read_csv(HERE / "tables/controls_complete.csv")
    differences = pd.read_csv(HERE / "tables/controls_paired_comparisons.csv")
    for r in gte.itertuples(index=False):
        actual = controls[
            controls.dataset.eq(r.dataset)
            & controls.model.eq(r.model)
            & controls.method.eq(r.alternative)
            & controls.metric.eq("Recall")
            & controls.K.eq(r.K)
        ].iloc[0]
        contrast = differences[
            differences.dataset.eq(r.dataset)
            & differences.model.eq(r.model)
            & differences.method.eq(r.alternative)
            & differences.metric.eq("Recall")
            & differences.K.eq(r.K)
            & differences.comparator.eq("raw:seven_mean")
        ].iloc[0]
        assert np.allclose(
            actual[["estimate", "ci_low", "ci_high"]].to_numpy(float),
            [
                r.alternative_recall,
                r.alternative_recall_ci_low,
                r.alternative_recall_ci_high,
            ],
            atol=1e-12,
            rtol=0,
        )
        assert np.allclose(
            contrast[["estimate", "ci_low", "ci_high"]].to_numpy(float),
            [
                r.delta_vs_mean_schedules,
                r.delta_vs_mean_schedules_ci_low,
                r.delta_vs_mean_schedules_ci_high,
            ],
            atol=1e-12,
            rtol=0,
        )

    boundary_checks = []
    for model in ["embeddinggemma2", "harrier06b"]:
        for dataset in ["wands", "esci"]:
            folder = ROOT / "data/boundary" / model / dataset
            audit = read(folder / "boundary_audit.json")
            selected = pd.read_parquet(folder / "boundary_selected.parquet")
            repeats = pd.read_parquet(folder / "boundary_repeats.parquet")
            keys = ["query_id", "product_id", "intervention", "K", "schedule", "side"]
            assert audit["selected_pairs"] == 32 and audit["repeat_count"] == 3
            assert len(selected) == 64 and len(repeats) == 192
            assert not selected.duplicated(keys).any()
            for _, g in repeats.groupby("repeat"):
                checked = g.merge(
                    selected[keys + ["rank"]], on=keys, validate="one_to_one"
                )
                assert len(checked) == len(selected)
                assert (checked.original_rank == checked["rank"]).all()
                assert (
                    (checked.repeated_rank <= checked.K)
                    == (checked.original_rank <= checked.K)
                ).all()
            assert (
                not repeats.numerical_ambiguity.any()
                and audit["numerical_ambiguities"] == 0
            )
            boundary_checks.append(
                dict(
                    model=model,
                    dataset=dataset,
                    pairs=32,
                    repeats=3,
                    all_cutoff_decisions_reproduced=True,
                )
            )
    dump(HERE / "qa/recent_model_boundary_checks.json", boundary_checks)
    return [
        dict(
            table="recent_models",
            rows=len(reference),
            max_absolute_error=max(errors),
            passed=True,
        ),
        dict(table="gte_canonical", rows=len(gte), passed=True),
        dict(table="recent_model_boundary", cells=4, passed=True),
    ]
