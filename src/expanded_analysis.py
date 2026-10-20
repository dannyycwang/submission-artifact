"""Nested order-family comparisons on a fixed common support."""

from .common import *


def query_states(frame, k):
    grouped = (
        frame.assign(included=frame["rank"] <= k)
        .groupby(["query_id", "product_id"])
        .included.agg(["all", "any"])
    )
    grouped["persistent_inclusion"] = grouped["all"].astype(float)
    grouped["VI"] = (grouped["any"] & ~grouped["all"]).astype(float)
    grouped["persistent_omission"] = (~grouped["any"]).astype(float)
    g = grouped[["persistent_inclusion", "VI", "persistent_omission"]]
    q = g.groupby(level="query_id").mean()
    q["n_highest"] = g.groupby(level="query_id").size()
    return g, q.reset_index()


def main():
    summary, increments = [], []
    cols = ["persistent_inclusion", "VI", "persistent_omission"]
    for ds in ["wands", "esci"]:
        for m in ["minilm", "bge_base"]:
            f = pairread(ROOT / f"data/expanded/{ds}_{m}.parquet")
            families = (
                [
                    ("7", f.schedule_index < 7),
                    ("16", f.schedule_index < 16),
                    ("32", f.schedule_index < 32),
                ]
                if ds == "wands"
                else [
                    ("7", f.schedule.isin(STEMS)),
                    ("exhaustive_target", f.schedule.str.startswith("E")),
                ]
            )
            for scope in ["full", "common_all_fitting"]:
                support_mask = (
                    np.ones(len(f), dtype=bool) if scope == "full" else f.fully_fitting
                )
                for k in [20, 100]:
                    prev = None
                    origin = None
                    for family, mask in families:
                        g, q = query_states(f[mask & support_mask], k)
                        meta = dict(
                            dataset=ds, model=m, support=scope, family=family, K=k
                        )
                        summary.extend(summarize(q, cols, meta))
                        if origin is None:
                            origin = g
                        if prev is not None:
                            pf, pg = prev
                            assert g.index.equals(pg.index) and (g.VI >= pg.VI).all()
                            comparisons = [(family + " minus " + pf, pg)]
                            if family == "32":
                                comparisons.append(("32 minus 7", origin))
                            for contrast, base in comparisons:
                                delta = (
                                    (g - base)
                                    .groupby(level="query_id")
                                    .mean()
                                    .reset_index()
                                )
                                delta["n_highest"] = q.n_highest
                                increments.extend(
                                    summarize(
                                        delta,
                                        cols,
                                        dict(
                                            dataset=ds,
                                            model=m,
                                            support=scope,
                                            K=k,
                                            contrast=contrast,
                                        ),
                                    )
                                )
                        prev = (family, g)
    csv(pd.DataFrame(summary), HERE / "tables/harmonized_summary.csv")
    csv(pd.DataFrame(increments), HERE / "tables/harmonized_increments.csv")
