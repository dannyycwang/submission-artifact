"""Tables and comparisons using the final manuscript's numbering and scope."""

from .common import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_LABELS = ["MiniLM", "BGE", "GTE", "EmbeddingGemma 2", "Harrier 0.6B"]


def save(fig, name):
    fig.savefig(HERE / "figures" / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(HERE / "figures" / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def build():
    controls = pd.read_csv(HERE / "tables/controls_complete.csv")
    chosen = controls[~controls.schedule.eq("C0")].copy()
    rows = []
    for (ds, model, method), g in chosen.groupby(["dataset", "model", "method"]):
        row = dict(
            dataset=ds,
            model=model,
            method=method,
            panel="A" if method in ["raw"] + RULES else "B",
        )
        for k, metric, key, scale in [
            (100, "Recall", "Recall@100_percent", 100),
            (20, "nDCG", "nDCG@20", 1),
            (100, "VI", "VI@100_percent", 100),
        ]:
            row[key] = (
                float(g[g.K.eq(k) & g.metric.eq(metric)].estimate.iloc[0]) * scale
            )
        rows.append(row)
    table3 = pd.DataFrame(rows)
    assert len(table3[table3.panel.eq("A")]) == 40
    assert len(table3[table3.panel.eq("B")]) == 32
    csv(table3, HERE / "tables/table3.csv")

    inc = pd.read_csv(HERE / "tables/inclusion_complete.csv")
    table4 = inc[
        inc.K.eq(20)
        & inc.metric.eq("VI")
        & (inc.intervention.eq("target_only") | inc.target_support.eq("full"))
    ].copy()
    table4["percent"] = 100 * table4.estimate
    csv(table4, HERE / "tables/table4.csv")
    expanded = pd.read_csv(HERE / "tables/harmonized_summary.csv")
    table5 = expanded[
        expanded.K.eq(20)
        & expanded.metric.eq("VI")
        & ~expanded.family.astype(str).eq("16")
    ].copy()
    table5["percent"] = 100 * table5.estimate
    csv(table5, HERE / "tables/table5.csv")

    membership = pd.read_csv(HERE / "tables/membership_all_contrasts.csv")
    membership = membership[membership.method.eq("raw") & membership.K.eq(20)].copy()
    csv(membership, HERE / "tables/figure6.csv")
    table6 = membership[
        membership.schedule.eq("C1")
        & membership.model.isin(["embeddinggemma2", "harrier06b"])
    ].copy()
    for col in [
        "turnover",
        "delta_recall",
        "delta_recall_ci_low",
        "delta_recall_ci_high",
    ]:
        table6[col + "_percent_or_pp"] = table6[col] * 100
    csv(table6, HERE / "tables/table6.csv")

    transitions = pd.read_csv(HERE / "tables/transitions_complete.csv")
    table7 = transitions[
        transitions.K.eq(20)
        & transitions.family.eq("dense")
        & transitions.reference_state.eq("crossing")
        & ~transitions.model.eq("gte_modernbert")
    ].copy()
    table7["percent"] = 100 * table7.estimate
    csv(table7, HERE / "tables/table7.csv")
    paired = pd.read_csv(HERE / "tables/controls_paired_comparisons.csv")
    table8 = paired[
        paired.method.eq("raw_hybrid")
        & paired.schedule.eq("seven_mean")
        & paired.comparator.eq("raw:seven_mean")
        & paired.K.eq(100)
        & paired.metric.isin(["Recall", "VI"])
    ].copy()
    for col in ["estimate", "ci_low", "ci_high"]:
        table8[col + "_pp"] = 100 * table8[col]
    csv(table8, HERE / "tables/table8.csv")
    figure7 = paired[
        paired.method.isin(RULES)
        & paired.comparator.eq("raw:seven_mean")
        & paired.K.eq(20)
        & paired.metric.eq("Recall")
    ].copy()
    assert len(figure7) == 30
    zero_intervals = int(((figure7.ci_low <= 0) & (figure7.ci_high >= 0)).sum())
    assert zero_intervals == 28
    csv(figure7, HERE / "tables/figure7.csv")

    # Figure 6: one common source-order contrast per column and encoder.
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharey=True)
    for i, ds in enumerate(["wands", "esci"]):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            g = (
                membership[membership.dataset.eq(ds) & membership.model.eq(model)]
                .set_index("schedule")
                .loc[STEMS[1:]]
            )
            y = np.arange(6)
            ax.barh(y, -100 * g.loss_share, color="#da9443", label="Loss")
            ax.barh(y, 100 * g.gain_share, color="#287c86", label="Gain")
            v = 100 * g.delta_recall.to_numpy()
            ax.errorbar(
                v,
                y,
                xerr=np.vstack(
                    [
                        v - 100 * g.delta_recall_ci_low.to_numpy(),
                        100 * g.delta_recall_ci_high.to_numpy() - v,
                    ]
                ),
                fmt="o",
                color="black",
                ms=3,
                capsize=2,
            )
            ax.axvline(0, color="gray", lw=0.6)
            ax.set_yticks(y, STEMS[1:])
            ax.set_title(MODEL_LABELS[j])
            ax.set_xlabel("Gain/loss (%) and Recall change (pp)")
            if j == 0:
                ax.set_ylabel(ds.upper())
    axes[0, 0].invert_yaxis()
    axes[1, 0].legend(loc="lower left", bbox_to_anchor=(0, -0.72), ncol=2)
    fig.tight_layout()
    save(fig, "figure6_membership_changes")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, ds in zip(axes, ["wands", "esci"]):
        for j, (rule, label, color, marker) in enumerate(
            zip(
                RULES,
                ["Ascending", "Descending", "Type priority"],
                ["#226788", "#c77529", "#368864"],
                ["o", "s", "^"],
            )
        ):
            g = (
                figure7[figure7.dataset.eq(ds) & figure7.method.eq(rule)]
                .set_index("model")
                .loc[MODELS]
            )
            v = g.estimate.to_numpy() * 100
            ax.errorbar(
                v,
                np.arange(5) + (j - 1) * 0.2,
                xerr=np.vstack(
                    [v - g.ci_low.to_numpy() * 100, g.ci_high.to_numpy() * 100 - v]
                ),
                fmt=marker,
                color=color,
                label=label,
                capsize=3,
            )
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_yticks(np.arange(5), MODEL_LABELS)
        ax.set_xlabel("Recall@20 change (percentage points)")
        ax.set_title(ds.upper())
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.3), ncol=3)
    fig.tight_layout()
    save(fig, "figure7_canonical_recall")

    # Unadjusted paired graded-effectiveness claim in Section 5.2.
    assert len(membership) == 60
    trend = dict(
        lower=int((membership.delta_ndcg_ci_high < 0).sum()),
        higher=int((membership.delta_ndcg_ci_low > 0).sum()),
        contains_zero=int(
            (
                (membership.delta_ndcg_ci_low <= 0)
                & (membership.delta_ndcg_ci_high >= 0)
            ).sum()
        ),
    )
    assert trend == dict(lower=8, higher=2, contains_zero=50), trend
    checks = [
        dict(
            claim="canonical Recall@20 intervals containing zero",
            comparisons=30,
            contains_zero=zero_intervals,
            passed=True,
        ),
        dict(
            claim="source-order nDCG@20 contrasts", comparisons=60, **trend, passed=True
        ),
    ]
    checks.extend(
        check_printed_values(controls, inc, expanded, transitions, membership, paired)
    )
    return checks


def check_printed_values(
    controls, inclusion, expanded, transitions, membership, paired
):
    expected = pd.read_csv(ROOT / "reference/manuscript_values.csv")
    for r in expected.itertuples(index=False):
        value_column = "estimate"
        if r.table == 3:
            f = controls[
                controls.dataset.eq(r.dataset)
                & controls.model.eq(r.model)
                & controls.method.eq(r.condition)
                & ~controls.schedule.eq("C0")
                & controls.K.eq(r.K)
                & controls.metric.eq(r.metric)
            ]
        elif r.table == 4:
            intervention, scope = r.condition.split(":")
            f = inclusion[
                inclusion.dataset.eq(r.dataset)
                & inclusion.model.eq(r.model)
                & inclusion.intervention.eq(intervention)
                & inclusion.target_support.eq(scope)
                & inclusion.K.eq(r.K)
                & inclusion.metric.eq(r.metric)
            ]
        elif r.table == 5:
            family, scope = r.condition.split(":")
            f = expanded[
                expanded.dataset.eq(r.dataset)
                & expanded.model.eq(r.model)
                & expanded.family.astype(str).eq(family)
                & expanded.support.eq(scope)
                & expanded.K.eq(r.K)
                & expanded.metric.eq(r.metric)
            ]
        elif r.table == 6:
            f = membership[
                membership.dataset.eq(r.dataset)
                & membership.model.eq(r.model)
                & membership.schedule.eq(r.condition)
                & membership.K.eq(r.K)
            ]
            value_column = r.metric
        elif r.table == 7:
            rule, outcome = r.condition.split(":")
            f = transitions[
                transitions.dataset.eq(r.dataset)
                & transitions.model.eq(r.model)
                & transitions.family.eq("dense")
                & transitions.rule.eq(rule)
                & transitions.reference_state.eq("crossing")
                & transitions.control_state.eq(outcome)
                & transitions.K.eq(r.K)
            ]
        elif r.table == 8:
            metric, value_column = r.metric.split(":")
            f = paired[
                paired.dataset.eq(r.dataset)
                & paired.model.eq(r.model)
                & paired.method.eq(r.condition)
                & paired.schedule.eq("seven_mean")
                & paired.comparator.eq("raw:seven_mean")
                & paired.K.eq(r.K)
                & paired.metric.eq(metric)
            ]
        else:
            raise ValueError(r.table)
        assert len(f) == 1, r
        actual = float(f[value_column].iloc[0]) * r.scale
        assert abs(actual - r.value) <= 0.5 * 10 ** (-r.decimals) + 1e-10, (r, actual)
    return [dict(table="printed_manuscript_values", cells=len(expected), passed=True)]
