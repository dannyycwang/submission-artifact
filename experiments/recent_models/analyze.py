"""Query-level evidence, paired CIs, manuscript candidate tables; no partial full summary."""

from common import *
from metrics import interval, bootstrap_matrix, macro_states


def summarize(frame, columns, meta):
    frame = frame.sort_values("query_id")
    assert not frame.query_id.duplicated().any()
    w = bootstrap_matrix(len(frame))
    rows = []
    for column in columns:
        mean, lo, hi = interval(frame[column].to_numpy(float), w)
        rows.append(
            dict(
                **meta,
                metric=column,
                estimate=float(mean),
                ci_low=float(lo),
                ci_high=float(hi),
                queries=len(frame),
                bootstrap_seed=SEED,
                bootstrap_draws=DRAWS
            )
        )
    return rows


def run():
    design = protocol()
    tables = HERE / "tables"
    tables.mkdir(exist_ok=True)
    summary = []
    main = []
    appendix = []
    missing = []
    for model, spec in design["models"].items():
        for ds in ["wands", "esci"]:
            base = HERE / "runs" / model / ds
            out = base / "evaluation/raw"
            flag = out / "complete.json"
            if not flag.exists():
                reason = "No complete evaluated full-catalog checkpoint"
                missing.append(
                    dict(model=model, dataset=ds, status="not_run", reason=reason)
                )
                continue
            check = read(flag)
            assert all(sha(HERE / x["path"]) == x["sha256"] for x in check["outputs"])
            metrics = pd.read_parquet(out / "query_metrics.parquet")
            members = pd.read_parquet(out / "query_membership.parquet")
            states = pd.read_parquet(out / "query_states.parquet")
            pairs = pd.read_parquet(out / "pair_ranks.parquet")
            rawmean = metrics.groupby(["query_id", "K"], as_index=False)[
                ["Recall", "nDCG", "cNDCG", "judged_coverage"]
            ].mean()
            for (c, k), g in metrics.groupby(["schedule", "K"]):
                summary += summarize(
                    g,
                    ["Recall", "nDCG", "cNDCG", "judged_coverage"],
                    dict(
                        model=model,
                        dataset=ds,
                        kind="effectiveness",
                        schedule=c,
                        K=int(k),
                        support="full",
                    ),
                )
            for k, g in rawmean.groupby("K"):
                summary += summarize(
                    g,
                    ["Recall", "nDCG", "cNDCG", "judged_coverage"],
                    dict(
                        model=model,
                        dataset=ds,
                        kind="effectiveness",
                        schedule="seven_mean",
                        K=int(k),
                        support="full",
                    ),
                )
            for (inter, support_name, k), g in states.groupby(
                ["intervention", "support", "K"]
            ):
                summary += summarize(
                    g,
                    ["persistent_inclusion", "crossing", "persistent_omission"],
                    dict(
                        model=model,
                        dataset=ds,
                        kind=inter + "_states",
                        schedule="seven",
                        K=int(k),
                        support=support_name,
                        pairs=int(g.n_highest.sum()),
                    ),
                )
            for (c, k), g in members.groupby(["schedule", "K"]):
                g = g.copy()
                g["gain_fraction"] = g.gains / g.n_highest
                g["loss_fraction"] = g.losses / g.n_highest
                g["turnover_fraction"] = g.turnover / g.n_highest
                summary += summarize(
                    g,
                    [
                        "delta_recall",
                        "delta_ndcg",
                        "gain_fraction",
                        "loss_fraction",
                        "turnover_fraction",
                        "same_recall_changed_relevant_membership",
                        "all_topk_turnover",
                    ],
                    dict(
                        model=model,
                        dataset=ds,
                        kind="paired_C0_contrast",
                        schedule=c,
                        K=int(k),
                        support="full",
                        same_recall_changed_queries=int(
                            g.same_recall_changed_relevant_membership.sum()
                        ),
                    ),
                )
            for k in [20, 100]:
                row = dict(
                    dataset=ds,
                    model=model,
                    K=k,
                    status="completed",
                    queries=int(pairs.query_id.nunique()),
                    pairs=len(pairs),
                    C0_Recall=float(
                        metrics[
                            (metrics.schedule == "C0") & (metrics.K == k)
                        ].Recall.mean()
                    ),
                    raw_mean_Recall=float(rawmean[rawmean.K == k].Recall.mean()),
                    same_Recall_changed_relevant_membership=float(
                        members[
                            (members.schedule == "C1") & (members.K == k)
                        ].same_recall_changed_relevant_membership.mean()
                    ),
                    fitting_pair_coverage=float(pairs.all_seven_fitting.mean()),
                    fitting_queries=int(
                        pairs[pairs.all_seven_fitting].query_id.nunique()
                    ),
                )
                for inter in ["target", "catalog"]:
                    g = states[
                        (states.intervention == inter)
                        & (states.support == "full")
                        & (states.K == k)
                    ].sort_values("query_id")
                    m, l, h = interval(g.crossing)
                    row.update(
                        {
                            inter + "_VI": float(m),
                            inter + "_VI_ci_low": float(l),
                            inter + "_VI_ci_high": float(h),
                        }
                    )
                (main if k == 20 else appendix).append(row)
            can = base / "evaluation/canonical"
            if (can / "complete.json").exists():
                for entry in read(can / "complete.json")["outputs"]:
                    assert sha(HERE / entry["path"]) == entry["sha256"]
                cm = pd.read_parquet(can / "query_metrics.parquet")
                tr = pd.read_parquet(can / "query_transitions.parquet")
                for (rule, k), g in cm.groupby(["schedule", "K"]):
                    ref = rawmean[rawmean.K == k].sort_values("query_id")
                    g = g.sort_values("query_id")
                    assert np.array_equal(ref.query_id, g.query_id)
                    cols = ["Recall", "nDCG", "cNDCG", "judged_coverage"]
                    diff = g[["query_id"]].copy()
                    diff[cols] = g[cols].to_numpy() - ref[cols].to_numpy()
                    summary += summarize(
                        g,
                        cols,
                        dict(
                            model=model,
                            dataset=ds,
                            kind="canonical_effectiveness",
                            schedule=rule,
                            K=int(k),
                            support="full",
                        ),
                    )
                    summary += summarize(
                        diff,
                        cols,
                        dict(
                            model=model,
                            dataset=ds,
                            kind="canonical_minus_raw_seven_mean",
                            schedule=rule,
                            K=int(k),
                            support="full",
                        ),
                    )
                for (rule, k, rs, cs), g in tr.groupby(
                    ["rule", "K", "raw_state", "canonical_state"]
                ):
                    summary += summarize(
                        g,
                        ["fraction"],
                        dict(
                            model=model,
                            dataset=ds,
                            kind="transition",
                            schedule=rule,
                            K=int(k),
                            support="full",
                            raw_state=rs,
                            canonical_state=cs,
                        ),
                    )
    pd.DataFrame(
        summary,
        columns=list(
            dict.fromkeys(
                [
                    "dataset",
                    "model",
                    "kind",
                    "schedule",
                    "K",
                    "support",
                    "metric",
                    "estimate",
                    "ci_low",
                    "ci_high",
                    "queries",
                ]
                + [k for r in summary for k in r]
            )
        ),
    ).to_csv(tables / "summary.csv", index=False)
    pd.DataFrame(
        main, columns=list(main[0]) if main else ["dataset", "model", "K", "status"]
    ).to_csv(tables / "main_candidates.csv", index=False)
    pd.DataFrame(
        appendix,
        columns=list(appendix[0]) if appendix else ["dataset", "model", "K", "status"],
    ).to_csv(tables / "appendix_K100.csv", index=False)
    pd.DataFrame(missing, columns=["dataset", "model", "status", "reason"]).to_csv(
        tables / "missing_cells.csv", index=False
    )
    plot = [
        r
        for r in summary
        if r["kind"] in ["target_states", "catalog_states"]
        and r["support"] == "full"
        and r["metric"] == "crossing"
    ]
    pd.DataFrame(
        plot,
        columns=(
            list(plot[0])
            if plot
            else ["dataset", "model", "kind", "K", "estimate", "ci_low", "ci_high"]
        ),
    ).to_csv(tables / "cross_model_VI.csv", index=False)
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        selected = [r for r in plot if r["K"] == 20]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for i, r in enumerate(selected):
            ax.errorbar(
                r["estimate"] * 100,
                i,
                xerr=[
                    [100 * (r["estimate"] - r["ci_low"])],
                    [100 * (r["ci_high"] - r["estimate"])],
                ],
                fmt="o",
                color="#17677A",
            )
        ax.set_yticks(
            range(len(selected)),
            [
                r["dataset"].upper()
                + " / "
                + r["model"]
                + " / "
                + r["kind"].replace("_states", "")
                for r in selected
            ],
        )
        ax.set_xlabel("Full-support query-macro VI@20 (%)")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(tables / "cross_model_VI.png", dpi=180)
        plt.close(fig)
    active = design.get("active_models", list(design["models"]))
    count = sum(r["model"] in active for r in main)
    status(
        "analyze",
        state="completed" if count == 2 * len(active) else "partial",
        complete_cells=count,
        expected_cells=2 * len(active),
    )
