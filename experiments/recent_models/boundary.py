"""Re-run exact original batches; score the full catalog in the same profile."""

from common import *
from adapters import NativeAdapter
from metrics import rank


def repeat_batches(adapter, directory, role, rows):
    plan = read(directory / "plan.json")
    aliases = np.array(plan["aliases"])
    required = set(aliases[list(rows)])
    original = np.load(directory / "unique_embeddings.npy")
    result = original.copy()
    with gzip.open(directory / "unique_inputs.jsonl.gz", "rt", encoding="utf8") as f:
        texts = [json.loads(l)["text"] for l in f]
    count = 0
    for batch in plan["batches"]:
        if required & set(batch):
            result[batch] = adapter.encode([texts[i] for i in batch], role)
            count += 1
    return result[aliases], dict(
        batches=count,
        unique_required=len(required),
        max_embedding_delta=float(np.max(abs(result - original))),
    )


def run(model, dataset):
    design = protocol()
    base = HERE / "runs" / model / dataset
    out = base / "evaluation/raw"
    assert (out / "complete.json").is_file()
    for entry in read(out / "complete.json")["outputs"]:
        assert sha(HERE / entry["path"]) == entry["sha256"]
    witness = pd.read_parquet(out / "crossing_witnesses.parquet")
    smoke = read(HERE / f"smoke/{model}/validation.json")
    if witness.empty:
        dump(
            out / "boundary_audit.json",
            dict(
                status="completed",
                crossing_witnesses=0,
                selected=0,
                numerical_ambiguities=0,
            ),
        )
        status("boundary", model, dataset)
        return
    pairs = ["query_id", "product_id", "intervention", "K"]
    witness["priority"] = [
        digest(
            f"recent-model-boundary|{dataset}|{r.query_id}|{r.product_id}|{r.intervention}|{r.K}"
        )
        for r in witness.itertuples()
    ]
    selected = set()
    for _, g in witness.groupby(["intervention", "K"]):
        sampled = g.drop_duplicates(pairs).sort_values("priority").head(8)
        selected.update(map(tuple, sampled[pairs].values))
    near = witness[
        (witness.margin.abs() <= smoke["max_repeat_score_range"]) | witness.tie
    ]
    selected.update(map(tuple, near[pairs].values))
    chosen = witness[[tuple(x) in selected for x in witness[pairs].values]].copy()
    chosen.to_parquet(out / "boundary_selected.parquet", index=False)
    products, queries, _ = support(dataset)
    n = len(products)
    qi = {int(q): i for i, q in enumerate(queries.query_id)}
    pi = {str(p["product_id"]): i for i, p in enumerate(products)}
    qrows = {qi[int(q)] for q in chosen.query_id}
    drows = set()
    for r in chosen.itertuples():
        drows.add(STEMS.index(r.schedule) * n + pi[r.product_id])
        drows.add(STEMS.index(r.competitor_schedule) * n + int(r.competitor_index))
        drows.add(pi[r.product_id])
    adapter = NativeAdapter(design["models"][model])
    audit = []
    execution = []
    try:
        for repeat in range(3):
            qe, qinfo = repeat_batches(adapter, base / "queries", "query", qrows)
            de, dinfo = repeat_batches(adapter, base / "raw", "document", drows)
            scores = [qe @ de[i * n : (i + 1) * n].T for i in range(7)]
            for r in chosen.itertuples():
                q = qi[int(r.query_id)]
                p = pi[r.product_id]
                s = STEMS.index(r.schedule)
                rr = (
                    int(rank_target(scores[0][q], np.array([p]), scores[s][q, [p]])[0])
                    if r.intervention == "target"
                    else int(rank(scores[s][q])[1][p])
                )
                audit.append(
                    dict(
                        query_id=int(r.query_id),
                        product_id=r.product_id,
                        intervention=r.intervention,
                        K=int(r.K),
                        schedule=r.schedule,
                        side=r.side,
                        repeat=repeat,
                        original_rank=int(r.rank),
                        repeated_rank=rr,
                        original_score=float(r.target_score),
                        repeated_score=float(scores[s][q, p]),
                        numerical_ambiguity=bool((rr <= r.K) != (r.rank <= r.K)),
                    )
                )
            execution.append(dict(repeat=repeat, query=qinfo, document=dinfo))
    finally:
        adapter.close()
    frame = pd.DataFrame(audit)
    frame.to_parquet(out / "boundary_repeats.parquet", index=False)
    dump(
        out / "boundary_audit.json",
        dict(
            status="completed",
            profile=design["boundary_audit"],
            selected_pairs=len(selected),
            witness_records=len(chosen),
            repeat_count=3,
            numerical_ambiguities=int(frame.numerical_ambiguity.sum()),
            execution=execution,
            original_rank_records_unchanged=True,
        ),
    )
    status(
        "boundary",
        model,
        dataset,
        outputs=[
            source(out / "boundary_audit.json"),
            source(out / "boundary_repeats.parquet"),
        ],
    )
