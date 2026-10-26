"""RQ1/RQ2 from the same full-catalog FP32 scores; never evaluate incomplete encodings."""

from common import *
from metrics import rank, query_metrics, membership, states, macro_states


def run(model, dataset, canonical=False):
    design = protocol()
    products, queries, judgments = support(dataset)
    spec = design["models"][model]
    directory = HERE / "runs" / model / dataset
    tag = "canonical" if canonical else "raw"
    out = directory / "evaluation" / tag
    out.mkdir(parents=True, exist_ok=True)
    for collection in ["queries", tag]:
        path = directory / collection
        assert valid(path / "complete.json", read(path / "descriptor.json")), (
            "incomplete or corrupt embedding collection",
            path,
        )
    conditions = RULES if canonical else STEMS
    n = len(products)
    doc = np.load(directory / tag / "embeddings.npy", mmap_mode="r")
    qe = np.load(directory / "queries/embeddings.npy")
    assert doc.shape == (len(conditions) * n, spec["dimension"]) and qe.shape == (
        len(queries),
        spec["dimension"],
    )
    ids = np.array([str(p["product_id"]) for p in products])
    pi = {p: i for i, p in enumerate(ids)}
    qi = {int(q): i for i, q in enumerate(queries.query_id)}
    inputs = dict(
        protocol_sha256=sha(HERE / "PROTOCOL.json"),
        query_complete=source(directory / "queries/complete.json"),
        document_complete=source(directory / tag / "complete.json"),
        evaluator_sha256=sha(Path(__file__)),
        metrics_sha256=sha(CODE / "metrics.py"),
    )
    if valid(out / "complete.json", inputs):
        return
    scores = []
    orders = []
    ranks = []
    for si, condition in enumerate(conditions):
        score = (
            np.asarray(qe, dtype=np.float32)
            @ np.asarray(doc[si * n : (si + 1) * n], dtype=np.float32).T
        )
        assert score.dtype == np.float32 and np.isfinite(score).all()
        np.save(out / f"{condition}_scores.npy", score)
        pair = [rank(row) for row in score]
        order = np.stack([p[0] for p in pair])
        rk = np.stack([p[1] for p in pair])
        scores.append(score)
        orders.append(order)
        ranks.append(rk)
        print("RANKED", model, dataset, condition, score.shape, flush=True)
    h = (
        judgments[judgments.label.eq(highest(dataset))]
        .sort_values(["query_id", "product_id"])
        .reset_index(drop=True)
    )
    pair = h[["query_id", "product_id", "label"]].copy()
    ii = np.array([pi[p] for p in pair.product_id])
    qq = np.array([qi[int(q)] for q in pair.query_id])
    unique = pd.read_csv(
        ROOT / f"data/serialization/{dataset}.csv", dtype={"product_id": str}
    ).set_index("product_id")
    pair["unique_serialized_views"] = unique.loc[
        pair.product_id, "unique_serialized_views"
    ].to_numpy()
    lengths = np.load(directory / tag / "lengths.npy").reshape(len(conditions), n)
    pair["all_seven_fitting"] = (
        np.max(np.load(directory / "raw/lengths.npy").reshape(7, n), axis=0)[ii]
        <= spec["max_length"]
    )
    for si, c in enumerate(conditions):
        pair["catalog_" + c] = ranks[si][qq, ii]
        if not canonical:
            target = np.empty(len(pair), dtype=np.int32)
            for qid, g in pair.groupby("query_id", sort=True):
                inds = np.array([pi[p] for p in g.product_id])
                qix = qi[int(qid)]
                target[g.index] = rank_target(
                    scores[0][qix], inds, scores[si][qix, inds]
                )
            pair["target_" + c] = target
    if not canonical:
        assert np.array_equal(
            pair.catalog_C0, pair.target_C0
        ), "C0 target/catalog disagreement"
    # Retain all judged ranks for independent graded-metric reconstruction.
    for si, condition in enumerate(conditions):
        judged = judgments[["query_id", "product_id", "label"]].copy()
        jj = judged.product_id.map(pi).to_numpy()
        qj = judged.query_id.map(qi).to_numpy()
        judged["rank"] = ranks[si][qj, jj]
        judged["score"] = scores[si][qj, jj]
        judged.to_parquet(out / (condition + "_all_judged.parquet"), index=False)
    metrics = []
    members = []
    for qid, g in judgments.groupby("query_id", sort=True):
        qix = qi[int(qid)]
        idx = np.array([pi[p] for p in g.product_id])
        high = set(g.loc[g.label.eq(highest(dataset)), "product_id"])
        for k in [20, 100]:
            base_top = set(ids[orders[0][qix, :k]])
            base_rel = base_top & high
            base_metrics = query_metrics(g.label, ranks[0][qix, idx], dataset, k)
            for si, c in enumerate(conditions):
                top = set(ids[orders[si][qix, :k]])
                rel = top & high
                m = query_metrics(g.label, ranks[si][qix, idx], dataset, k)
                metrics.append(
                    dict(
                        query_id=int(qid),
                        schedule=c,
                        K=k,
                        **m,
                        highest_ids=json.dumps(sorted(rel)),
                        top_ids=json.dumps(ids[orders[si][qix, :k]].tolist()),
                    )
                )
                if not canonical and si:
                    change = membership(base_rel, rel, len(high))
                    assert np.isclose(
                        m["Recall"] - base_metrics["Recall"],
                        change["delta_recall"],
                        atol=1e-15,
                        rtol=0,
                    )
                    members.append(
                        dict(
                            query_id=int(qid),
                            schedule=c,
                            K=k,
                            n_highest=len(high),
                            **change,
                            all_topk_losses=len(base_top - top),
                            all_topk_gains=len(top - base_top),
                            all_topk_turnover=len(base_top ^ top),
                            source_relevant_count=len(base_rel),
                            alternative_relevant_count=len(rel),
                            delta_ndcg=m["nDCG"] - base_metrics["nDCG"],
                            ndcg_exactly_same=m["nDCG"] == base_metrics["nDCG"],
                            ndcg_within_1e12=abs(m["nDCG"] - base_metrics["nDCG"])
                            <= 1e-12,
                        )
                    )
    pd.DataFrame(metrics).to_parquet(out / "query_metrics.parquet", index=False)
    pair.to_parquet(out / "pair_ranks.parquet", index=False)
    if not canonical:
        pd.DataFrame(members).to_parquet(out / "query_membership.parquet", index=False)
        state_rows = []
        witness = []
        for intervention in ["catalog", "target"]:
            cols = [intervention + "_" + s for s in STEMS]
            for k in [20, 100]:
                pair[f"{intervention}_state_K{k}"] = states(pair[cols], k)
                for support_name, frame in [
                    ("full", pair),
                    ("fitting", pair[pair.all_seven_fitting]),
                ]:
                    if not len(frame):
                        continue
                    result = macro_states(frame, cols, k)
                    result["support"] = support_name
                    result["intervention"] = intervention
                    result["K"] = k
                    state_rows.append(result)
                crossing = pair[pair[f"{intervention}_state_K{k}"].eq("crossing")]
                for row in crossing.itertuples():
                    qix = qi[int(row.query_id)]
                    pix = pi[row.product_id]
                    in_s = next(
                        i
                        for i, s in enumerate(STEMS)
                        if getattr(row, intervention + "_" + s) <= k
                    )
                    out_s = next(
                        i
                        for i, s in enumerate(STEMS)
                        if getattr(row, intervention + "_" + s) > k
                    )
                    for side, si in [("in", in_s), ("out", out_s)]:
                        ci = si if intervention == "catalog" else 0
                        order = orders[ci][qix]
                        boundary = int(
                            order[k] if ranks[ci][qix, pix] <= k else order[k - 1]
                        )
                        score = float(scores[si][qix, pix])
                        threshold = float(scores[ci][qix, boundary])
                        witness.append(
                            dict(
                                query_id=int(row.query_id),
                                product_id=row.product_id,
                                intervention=intervention,
                                K=k,
                                side=side,
                                schedule=STEMS[si],
                                rank=int(getattr(row, intervention + "_" + STEMS[si])),
                                target_score=score,
                                competitor_index=boundary,
                                competitor_id=ids[boundary],
                                competitor_schedule=STEMS[ci],
                                competitor_score=threshold,
                                margin=score - threshold,
                                tie=score == threshold,
                                target_original_index=pix,
                                tie_target_precedes_competitor=pix < boundary,
                                tokens=int(lengths[si, pix]),
                                all_seven_fitting=bool(row.all_seven_fitting),
                            )
                        )
        pd.concat(state_rows, ignore_index=True).to_parquet(
            out / "query_states.parquet", index=False
        )
        pair.to_parquet(out / "pair_ranks.parquet", index=False)
        pd.DataFrame(
            witness,
            columns=[
                "query_id",
                "product_id",
                "intervention",
                "K",
                "side",
                "schedule",
                "rank",
                "target_score",
                "competitor_index",
                "competitor_id",
                "competitor_schedule",
                "competitor_score",
                "margin",
                "tie",
                "target_original_index",
                "tie_target_precedes_competitor",
                "tokens",
                "all_seven_fitting",
            ],
        ).to_parquet(out / "crossing_witnesses.parquet", index=False)
    else:
        raw = pd.read_parquet(directory / "evaluation/raw/pair_ranks.parquet")
        assert raw[["query_id", "product_id"]].equals(pair[["query_id", "product_id"]])
        transitions = []
        for c in conditions:
            for k in [20, 100]:
                s = states(raw[["catalog_" + x for x in STEMS]], k)
                inc = pair["catalog_" + c].to_numpy() <= k
                for qid, ix in pair.groupby("query_id", sort=True).groups.items():
                    for rs in [
                        "persistent_inclusion",
                        "crossing",
                        "persistent_omission",
                    ]:
                        for cs in [True, False]:
                            transitions.append(
                                dict(
                                    query_id=int(qid),
                                    rule=c,
                                    K=k,
                                    raw_state=rs,
                                    canonical_state="included" if cs else "omitted",
                                    n_highest=len(ix),
                                    fraction=float(
                                        np.mean((s[ix] == rs) & (inc[ix] == cs))
                                    ),
                                )
                            )
        tr = pd.DataFrame(transitions)
        assert np.allclose(
            tr.groupby(["query_id", "rule", "K"]).fraction.sum(), 1, atol=1e-15, rtol=0
        )
        tr.to_parquet(out / "query_transitions.parquet", index=False)
    outputs = list(out.glob("*.npy")) + list(out.glob("*.parquet"))
    complete(
        out / "complete.json",
        inputs,
        outputs,
        C0_reconciled=not canonical,
        full_catalog=n,
        queries=len(queries),
        highest_pairs=len(pair),
    )
    status(
        "canonical" if canonical else "evaluate",
        model,
        dataset,
        outputs=[source(out / "complete.json")],
    )
