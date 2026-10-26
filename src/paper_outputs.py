"""Readable paper tables and plots built from recomputed evidence."""

from .common import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "raw": "Raw dense",
    "raw_hybrid": "Raw hybrid",
    "lexical_ascending": "Canonical: ascending",
    "lexical_descending": "Canonical: descending",
    "field_priority_type": "Canonical: type priority",
    "set_mean": "Set-Mean",
    "centroid_2": "Centroid (M=2)",
    "max_2": "Multi-vector (M=2)",
    "BM25": "BM25",
    "canonical_hybrid_lexical_ascending": "Canonical hybrid: ascending",
    "canonical_hybrid_lexical_descending": "Canonical hybrid: descending",
    "canonical_hybrid_field_priority_type": "Canonical hybrid: type priority",
}
ORDER = [
    "raw",
    "lexical_ascending",
    "lexical_descending",
    "field_priority_type",
    "set_mean",
    "centroid_2",
    "max_2",
    "BM25",
    "raw_hybrid",
    "canonical_hybrid_lexical_ascending",
    "canonical_hybrid_lexical_descending",
    "canonical_hybrid_field_priority_type",
]


def save(fig, name):
    fig.savefig(HERE / "figures" / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(HERE / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def build():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    checks = []
    from .final_paper import build as final_tables, MODEL_LABELS

    checks.extend(final_tables())

    # Figure 1: preserve all seven target ranks, including the two illustrated orders.
    teal = pd.read_csv(ROOT / "data/cases/teal_chair/native_records.csv")
    assert teal.saved_rank.tolist() == [11, 468, 123, 301, 677, 435, 939]
    target = pairread(ROOT / "data/target_only/wands_gte_modernbert.parquet")
    target = target[target.query_id.eq(409) & target.product_id.eq("24318")].iloc[0]
    assert [int(target[s]) for s in STEMS] == teal.saved_rank.tolist()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(teal.schedule, teal.saved_rank, "o-", color="#236b8e")
    ax.axhline(20, color="#b02b2c", linestyle="--", label="Top-20 cutoff")
    ax.set_yscale("log")
    ax.set_ylabel("Target rank (log scale)")
    ax.set_title("Figure 1 evidence: teal chair, WANDS / GTE")
    ax.legend()
    save(fig, "figure1_target_ranks")
    csv(teal, HERE / "tables/figure1.csv")
    checks.append(
        dict(case="teal chair", source_rank=11, alternative_rank=939, passed=True)
    )

    inc = pd.read_csv(HERE / "tables/inclusion_complete.csv")
    plot = inc[inc.intervention.eq("target_only") & inc.K.eq(20) & inc.metric.eq("VI")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, ds in zip(axes, ["wands", "esci"]):
        for i, m in enumerate(MODELS):
            for offset, scope, color, marker in [
                (-0.11, "full", "#236b8e", "o"),
                (0.11, "fully_fitting", "#db7a2b", "s"),
            ]:
                if scope == "fully_fitting" and m in [
                    "gte_modernbert",
                    "embeddinggemma2",
                    "harrier06b",
                ]:
                    continue
                row = plot[
                    plot.dataset.eq(ds)
                    & plot.model.eq(m)
                    & plot.target_support.eq(scope)
                ].iloc[0]
                ax.errorbar(
                    i + offset,
                    row.estimate * 100,
                    yerr=[
                        [100 * (row.estimate - row.ci_low)],
                        [100 * (row.ci_high - row.estimate)],
                    ],
                    fmt=marker,
                    color=color,
                    capsize=3,
                    label=scope.replace("_", " ") if i == 0 else None,
                )
        ax.set_xticks(
            range(5),
            ["MiniLM", "BGE", "GTE", "EmbeddingGemma 2", "Harrier 0.6B"],
            rotation=15,
        )
        ax.set_title(ds.upper())
        ax.set_ylabel("Target-only VI@20 (%)")
        ax.legend()
    fig.tight_layout()
    save(fig, "figure3_target_sensitivity")
    csv(plot, HERE / "tables/figure3.csv")

    adj = pd.read_csv(HERE / "data/aggregate_estimates.csv")
    adj = adj[adj.K.eq(20) & adj.metric.eq("A")]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(adj))
    v = adj.estimate.to_numpy() * 100
    ax.errorbar(
        x,
        v,
        yerr=np.vstack(
            (v - adj.ci_low.to_numpy() * 100, adj.ci_high.to_numpy() * 100 - v)
        ),
        fmt="o",
        capsize=4,
        color="#236b8e",
    )
    ax.set_xticks(x, [f"{r.dataset.upper()}\n{r.model}" for r in adj.itertuples()])
    ax.set_ylabel("Targets affected by an adjacent swap (%)")
    ax.set_title("Figure 4: fully-fitting target-only intervention")
    save(fig, "figure4_adjacent_swaps")
    csv(adj, HERE / "tables/figure4.csv")

    mem = pd.read_parquet(HERE / "data/membership_per_query.parquet")
    mem = mem[mem.method.eq("raw") & mem.schedule.eq("C1") & mem.K.eq(20)]
    mat = mem[mem.dataset.eq("wands") & mem.model.eq("bge_base") & mem.query_id.eq(231)]
    r = mat.iloc[0]
    assert (
        r.gain_count,
        r.loss_count,
        r.source_count,
        r.alternative_count,
        r.n_highest,
    ) == (5, 5, 20, 20, 307)
    csv(mat, HERE / "tables/anti_fatigue_mat.csv")
    checks.append(
        dict(
            case="anti fatigue mat",
            gained=5,
            lost=5,
            recall="20/307 in both schedules",
            passed=True,
        )
    )
    bars = []
    for (ds, m), g in mem.groupby(["dataset", "model"], sort=False):
        bars.append(
            dict(
                dataset=ds,
                model=m,
                queries=len(g),
                exact_cancellation=int(g.exact_replacement.sum()),
                unchanged=int((~g.changed).sum()),
                net_change=int((g.changed & ~g.exact_replacement).sum()),
            )
        )
    b = (
        pd.DataFrame(bars)
        .set_index(["dataset", "model"])
        .reindex(
            pd.MultiIndex.from_product(
                [["wands", "esci"], MODELS], names=["dataset", "model"]
            )
        )
        .reset_index()
    )
    csv(b, HERE / "tables/figure5.csv")
    row = b[b.dataset.eq("wands") & b.model.eq("bge_base")].iloc[0]
    assert (row.exact_cancellation, row.queries) == (73, 308)
    fig, ax = plt.subplots(figsize=(10, 6))
    left = np.zeros(len(b))
    for col, color, label in [
        ("unchanged", "#c4cbd1", "Unchanged relevant membership"),
        ("exact_cancellation", "#df902b", "Equal positive gains and losses"),
        ("net_change", "#346c8f", "Net Recall change"),
    ]:
        values = b[col].to_numpy() / b.queries.to_numpy() * 100
        ax.barh(np.arange(len(b)), values, left=left, color=color, label=label)
        left += values
    names = dict(zip(MODELS, MODEL_LABELS))
    ax.set_yticks(
        range(len(b)),
        [f"{r.dataset.upper()} / {names[r.model]}" for r in b.itertuples()],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Eligible queries (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    save(fig, "figure5_recall_cancellation")
    checks.append(
        dict(
            claim="WANDS/BGE exact Recall cancellation",
            numerator=73,
            denominator=308,
            passed=True,
        )
    )

    wooden = pd.read_csv(
        ROOT / "data/cases/wooden_chair/all_products.csv", dtype={"product_id": str}
    )
    before = wooden[wooden.membership.isin(["retained", "lost"])]
    after = wooden[wooden.membership.isin(["retained", "gained"])]
    assert len(before) == len(after) == 20
    assert (
        before.condition_status.eq("MATCH").sum(),
        after.condition_status.eq("MATCH").sum(),
    ) == (8, 6)
    assert (
        before.condition_status.isin(["UNKNOWN", "AMBIGUOUS"]).sum()
        == after.condition_status.isin(["UNKNOWN", "AMBIGUOUS"]).sum()
        == 9
    )
    matched = read(ROOT / "data/cases/wooden_chair/matched_scoring.json")["results"]
    for schedule in ["C0", "C1"]:
        mapping = {
            str(r["product_id"]): r["rescored_rank"]
            for r in matched[schedule]["products"]
        }
        wooden["cached_score_rank_" + schedule] = wooden.product_id.map(mapping)
    csv(wooden[~wooden.membership.eq("retained")], HERE / "tables/table2_products.csv")
    summary = pd.read_parquet(ROOT / "data/reference_per_query.parquet")
    summary = summary[
        summary.dataset.eq("wands")
        & summary.model.eq("bge_base")
        & summary.method.eq("raw")
        & summary.query_id.eq(367)
        & summary.schedule.isin(["C0", "C1"])
    ]
    assert np.allclose(summary["Recall@20"], 7 / 42, atol=1e-14, rtol=0)
    csv(
        summary[["schedule", "n_highest", "Recall@20", "nDCG@20"]],
        HERE / "tables/table2_metrics.csv",
    )
    checks.append(
        dict(
            case="wooden chair outdoor",
            confirmed_wood_before=8,
            confirmed_wood_after=6,
            recall="7/42 in both schedules",
            passed=True,
        )
    )
    return checks
