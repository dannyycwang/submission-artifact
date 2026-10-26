"""Derive inclusion states and canonical transitions from product identities."""

from .common import *


def recompute(pq):
    rows, transitions = [], []
    states = {}
    for (ds, model, method), group in pq.groupby(
        ["dataset", "model", "method"], sort=False
    ):
        for k in [20, 100]:
            for qid, q in group.groupby("query_id", sort=True):
                highest = set(json.loads(q.highest_support_ids.iloc[0]))
                views = [set(json.loads(v)) for v in q[f"included_ids_K{k}"]]
                included, ever = set.intersection(*views), set.union(*views)
                crossing, omitted = ever - included, highest - ever
                assert len(included) + len(crossing) + len(omitted) == len(highest)
                source = q[q.schedule.eq("C0")]
                source_ids = set(
                    json.loads(
                        (source if len(source) else q)[f"included_ids_K{k}"].iloc[0]
                    )
                )
                n = len(highest)
                row = dict(
                    dataset=ds,
                    model=model,
                    method=method,
                    K=k,
                    query_id=qid,
                    n_highest=n,
                    persistent_inclusion=len(included) / n,
                    persistent_omission=len(omitted) / n,
                    VI=len(crossing) / n,
                    mean_recall=sum(map(len, views)) / len(views) / n,
                    source_recall=len(source_ids) / n,
                    basis=(
                        "observed seven catalog schedules"
                        if len(views) == 7
                        else "fixed representation"
                    ),
                )
                if method == "BM25":
                    rows.extend(dict(row, model=m) for m in ["minilm", "bge_base"])
                else:
                    rows.append(row)
                states[(ds, model, method, k, qid)] = (
                    highest,
                    included,
                    crossing,
                    omitted,
                )
    for ds in ["wands", "esci"]:
        for m in MODELS:
            for family, raw in [("dense", "raw"), ("hybrid", "raw_hybrid")]:
                if family == "hybrid" and m not in ["minilm", "bge_base"]:
                    continue
                for rule in RULES:
                    method = rule if family == "dense" else "canonical_hybrid_" + rule
                    for k in [20, 100]:
                        for qid in support(ds):
                            highest, inc, cross, omit = states[(ds, m, raw, k, qid)]
                            fixed = states[(ds, m, method, k, qid)][1]
                            for name, ids in [
                                ("persistent_inclusion", inc),
                                ("crossing", cross),
                                ("persistent_omission", omit),
                            ]:
                                for target, items in [
                                    ("included", ids & fixed),
                                    ("omitted", ids - fixed),
                                ]:
                                    transitions.append(
                                        dict(
                                            dataset=ds,
                                            model=m,
                                            family=family,
                                            rule=rule,
                                            K=k,
                                            query_id=qid,
                                            reference_state=name,
                                            control_state=target,
                                            cell_pairs=len(items),
                                            Hq=len(highest),
                                            state_pairs=len(ids),
                                            cell_fraction=len(items) / len(highest),
                                            conditional_fraction=(
                                                len(items) / len(ids) if ids else np.nan
                                            ),
                                        )
                                    )
    for name, frame, keys, cols in [
        (
            "states",
            pd.DataFrame(rows),
            ["dataset", "model", "method", "K", "query_id"],
            [
                "n_highest",
                "persistent_inclusion",
                "persistent_omission",
                "VI",
                "mean_recall",
                "source_recall",
            ],
        ),
        (
            "transitions",
            pd.DataFrame(transitions),
            [
                "dataset",
                "model",
                "family",
                "rule",
                "K",
                "query_id",
                "reference_state",
                "control_state",
            ],
            ["cell_pairs", "Hq", "state_pairs", "cell_fraction"],
        ),
    ]:
        reference = (
            pd.read_parquet(ROOT / f"data/reference_{name}.parquet")
            .set_index(keys)
            .sort_index()
        )
        actual = frame.set_index(keys).sort_index()
        assert reference.index.isin(actual.index).all(), name
        assert not actual.index.duplicated().any(), name
        assert np.allclose(
            actual.loc[reference.index, cols], reference[cols], atol=1e-14, rtol=0
        ), name
        path = (
            "states_per_query.parquet"
            if name == "states"
            else "control_transitions_per_query.parquet"
        )
        frame.to_parquet(HERE / "data" / path, index=False)
