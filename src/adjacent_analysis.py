"""Adjacent-swap estimates from the completed, numerically confirmed trials."""

from .common import *

METRICS = [
    "A",
    "F",
    "loss",
    "gain",
    "baseline_inclusion",
    "A_confirmed",
    "F_confirmed",
    "loss_confirmed",
    "gain_confirmed",
]


def summarize_queries(frame, meta):
    q = frame.sort_values("query_id")
    values = q[METRICS].to_numpy(dtype=float)
    rng = np.random.default_rng(2026100101)
    indices = rng.integers(0, len(q), size=(10000, len(q)))
    samples = values[indices].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
    point = values.mean(axis=0)
    return [
        dict(
            **meta,
            metric=metric,
            estimate=float(e),
            ci_low=float(l),
            ci_high=float(h),
            queries=len(q),
            targets=int(q.targets.sum()),
            swaps=int(q.swaps.sum()),
            bootstrap_seed=2026100101,
            bootstrap_resamples=10000,
            interval="95% unadjusted query-cluster percentile",
        )
        for metric, e, l, h in zip(METRICS, point, lo, hi)
    ]


def query_means(f):
    g = f.groupby("query_id", sort=True)
    q = g[METRICS].mean()
    q["targets"] = g.size()
    q["swaps"] = g.distinct_swaps.sum()
    return q.reset_index()


def main():
    out = []
    for m in ["minilm", "bge_base"]:
        out.append(pd.read_parquet(ROOT / f"data/adjacent/{m}_variants.parquet"))
    f = pd.concat(out, ignore_index=True)
    f.to_parquet(HERE / "data/per_variant.parquet", index=False)
    rows = []
    for (ds, m, q, p), g in f.groupby(
        ["dataset", "model", "query_id", "product_id"], sort=True
    ):
        r = g.iloc[0]
        assert len(g) == r.distinct_swaps
        for k in [20, 100]:
            flips = g[f"flip_{k}"]
            confirmed = g[f"confirmed_{k}"]
            unresolved = g[f"unresolved_{k}"]
            assert ((confirmed | unresolved) == flips).all() and not (
                confirmed & unresolved
            ).any()
            affected = bool(flips.any())
            confirmed_any = bool(confirmed.any())
            inside = bool(r[f"c0_included_{k}"])
            rows.append(
                dict(
                    dataset=ds,
                    model=m,
                    query_id=int(q),
                    product_id=p,
                    K=k,
                    distinct_swaps=len(g),
                    attribute_count=int(r.attribute_count),
                    c0_rank=int(r.c0_rank),
                    c0_score=r.c0_score,
                    c0_margin=r[f"c0_margin_{k}"],
                    c0_tokens=int(r.c0_tokens),
                    maximum_tokens=int(r.max_tokens),
                    context_limit=int(r.context_limit),
                    A=float(affected),
                    F=float(flips.mean()),
                    loss=float(affected and inside),
                    gain=float(affected and not inside),
                    baseline_inclusion=float(inside),
                    A_confirmed=float(confirmed_any),
                    F_confirmed=float(confirmed.mean()),
                    loss_confirmed=float(confirmed_any and inside),
                    gain_confirmed=float(confirmed_any and not inside),
                    observed_flip_count=int(flips.sum()),
                    confirmed_flip_count=int(confirmed.sum()),
                    unresolved_flip_count=int(unresolved.sum()),
                    minimum_swap_rank=int(g.swap_rank.min()),
                    maximum_swap_rank=int(g.swap_rank.max()),
                    mean_rank_change=float(g.rank_change.mean()),
                    maximum_absolute_rank_change=int(g.rank_change.abs().max()),
                    minimum_swap_margin=float(g[f"swap_margin_{k}"].min()),
                    maximum_swap_margin=float(g[f"swap_margin_{k}"].max()),
                    case_hash=r.case_hash,
                )
            )
    targets = pd.DataFrame(rows)
    targets["swap_count_bin"] = pd.cut(
        targets.distinct_swaps,
        [0, 1, 3, 7, 15, np.inf],
        labels=["1", "2-3", "4-7", "8-15", "16+"],
    ).astype(str)
    targets["c0_rank_bin"] = pd.cut(
        targets.c0_rank,
        [0, 10, 20, 40, np.inf],
        labels=["1-10", "11-20", "21-40", ">40"],
    ).astype(str)
    targets.to_parquet(HERE / "data/per_target.parquet", index=False)
    csv(targets, HERE / "data/per_target.csv")
    qframes = []
    estimates = []
    event_counts = []
    breakdowns = []
    descriptives = []
    for (ds, m, k), g in targets.groupby(["dataset", "model", "K"], sort=True):
        meta = dict(dataset=ds, model=m, K=int(k))
        q = query_means(g)
        qframes.append(q.assign(**meta))
        estimates.extend(summarize_queries(q, meta))
        event_counts.append(
            dict(
                **meta,
                queries=g.query_id.nunique(),
                targets=len(g),
                swaps=int(g.distinct_swaps.sum()),
                affected_targets=int(g.A.sum()),
                confirmed_affected_targets=int(g.A_confirmed.sum()),
                loss_targets=int(g.loss.sum()),
                gain_targets=int(g.gain.sum()),
                confirmed_loss_targets=int(g.loss_confirmed.sum()),
                confirmed_gain_targets=int(g.gain_confirmed.sum()),
                observed_flips=int(g.observed_flip_count.sum()),
                confirmed_flips=int(g.confirmed_flip_count.sum()),
                unresolved_flips=int(g.unresolved_flip_count.sum()),
                unresolved_targets=int(g.unresolved_flip_count.gt(0).sum()),
                affected_queries=int(q.A.gt(0).sum()),
            )
        )
        for field in ["swap_count_bin", "c0_rank_bin"]:
            for value, sub in g.groupby(field, sort=True):
                means = query_means(sub)
                breakdowns.append(
                    dict(
                        **meta,
                        breakdown=field,
                        bin=value,
                        queries=len(means),
                        targets=len(sub),
                        swaps=int(sub.distinct_swaps.sum()),
                        affected_targets=int(sub.A.sum()),
                        confirmed_targets=int(sub.A_confirmed.sum()),
                        **{metric: float(means[metric].mean()) for metric in METRICS},
                    )
                )
        for field in [
            "attribute_count",
            "distinct_swaps",
            "c0_rank",
            "c0_margin",
            "mean_rank_change",
            "maximum_absolute_rank_change",
        ]:
            a = g[field]
            descriptives.append(
                dict(
                    **meta,
                    variable=field,
                    targets=len(g),
                    minimum=float(a.min()),
                    p25=float(a.quantile(0.25)),
                    median=float(a.median()),
                    p75=float(a.quantile(0.75)),
                    maximum=float(a.max()),
                    mean=float(a.mean()),
                )
            )
    queries = pd.concat(qframes, ignore_index=True)
    csv(queries, HERE / "data/per_query.csv")
    queries.to_parquet(HERE / "data/per_query.parquet", index=False)
    csv(pd.DataFrame(estimates), HERE / "data/aggregate_estimates.csv")
    csv(pd.DataFrame(event_counts), HERE / "data/event_counts.csv")
    csv(pd.DataFrame(breakdowns), HERE / "data/descriptive_breakdowns.csv")
    csv(pd.DataFrame(descriptives), HERE / "data/supporting_distributions.csv")
