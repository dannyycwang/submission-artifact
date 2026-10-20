"""Exact target-only ranks, matched cache reuse, no-op gate, and crossing repeats.

Run models sequentially. All new vectors and block hashes are resumable; source
arrays remain read-only. Do not consume partial model outputs in analysis.
"""

from common import *
import argparse, time


class Engine:
    def __init__(self, model):
        import torch, transformers, tokenizers
        import importlib.metadata
        from transformers import AutoTokenizer, AutoModel

        self.torch = torch
        self.key = model
        self.spec = MODEL_SPECS[model]
        self.meta = read(HERE / "execution_profile.json")
        self.profile = self.meta["profiles"][model]
        assert torch.cuda.is_available()
        for p, v in [
            ("torch", torch.__version__),
            ("transformers", transformers.__version__),
            ("tokenizers", tokenizers.__version__),
        ]:
            assert v == self.meta["versions"][p], (p, v, self.meta["versions"][p])
        assert torch.version.cuda == self.meta["cuda"]
        dump(
            HERE / f"qa/{model}_rerun_device.json",
            dict(
                gpu=torch.cuda.get_device_name(),
                original_gpu=self.meta["gpu"],
                same_gpu_name=torch.cuda.get_device_name() == self.meta["gpu"],
                cuda=torch.version.cuda,
            ),
        )
        for package, version in self.meta["versions"].items():
            assert importlib.metadata.version(package) == version, (
                "Frozen environment changed",
                package,
            )
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=self.spec["name"],
                revision=self.spec["revision"],
                local_files_only=True,
            )
        )
        for filename, expected in self.profile["model_snapshot_hashes"].items():
            assert sha(snapshot / filename) == expected, (
                "Frozen model snapshot changed",
                filename,
            )
        torch.set_num_threads(4)
        torch.manual_seed(20261001)
        torch.cuda.manual_seed_all(20261001)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec["name"], revision=self.spec["revision"], local_files_only=True
        )
        self.net = (
            AutoModel.from_pretrained(
                self.spec["name"],
                revision=self.spec["revision"],
                local_files_only=True,
                attn_implementation="eager",
            )
            .cuda()
            .half()
            .eval()
        )
        self.bs = self.profile["batch_size"]
        self.cap = self.spec["max_tokens"]

    def forward(self, texts, b):
        torch = self.torch
        n = len(texts)
        assert 0 < n <= self.bs
        texts = list(texts) + [texts[-1]] * (self.bs - n)
        enc = self.tokenizer(
            texts,
            padding="max_length",
            max_length=int(b),
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.inference_mode():
            hidden = self.net(**enc).last_hidden_state
            if self.key == "minilm":
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                pooled = hidden[:, 0].float()
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return vectors[:n].cpu().numpy()

    def frame(self, frame):
        out = {}
        for b, g in frame.groupby("bucket", sort=True):
            g = g.sort_values("input_sha256")
            for start in range(0, len(g), self.bs):
                part = g.iloc[start : start + self.bs]
                vs = self.forward(part.text.tolist(), b)
                out.update(zip(part.input_sha256, vs))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument(
        "--reference-dir",
        type=Path,
        help="Optional newly rebuilt C0 reference from rebuild_reference.py; never silently substitute it.",
    )
    args = ap.parse_args()
    m = args.model
    frozen = verify_freeze()
    t = time.time()
    folder = HERE / "cache" / m
    folder.mkdir(parents=True, exist_ok=True)
    meta = read(HERE / "execution_profile.json")
    prof = meta["profiles"][m]
    reference = args.reference_dir.resolve() if args.reference_dir else HARM / m
    if args.reference_dir:
        rebuilt = read(reference / "complete.json")
        assert (
            rebuilt["freeze_sha256"] == sha(HERE / "freeze_manifest.json")
            and rebuilt["model"] == m
        )
        refplan = rebuilt
        refhash = rebuilt["vectors_sha256"]
        reference_origin = "rebuilt_matched_C0"
    else:
        refplan = prof["reference_plan"]
        refhash = prof["reference_complete"]["vectors_sha256"]
        reference_origin = "section6_harmonized"
    for name, key in [
        ("inputs.parquet", "inputs_sha256"),
        ("aliases.parquet", "aliases_sha256"),
    ]:
        assert sha(reference / name) == refplan[key]
    assert sha(reference / "vectors.npy") == refhash
    old = np.load(reference / "vectors.npy", mmap_mode="r")
    old_inputs = pd.read_parquet(reference / "inputs.parquet")
    old_map = old_inputs.set_index("input_sha256").vector_index.to_dict()
    old_alias = pd.read_parquet(reference / "aliases.parquet")
    engine = Engine(m)
    tok = engine.tokenizer
    plan = pd.read_parquet(HERE / "data/variant_plan.parquet")
    plan = plan[plan.model.eq(m)].copy()
    queries = pd.read_csv(HERE / "data/queries.csv")
    queries = queries[queries.model.eq(m)]
    cohort = pd.read_csv(HERE / "cohort_manifest.csv", dtype={"product_id": str})
    cohort = cohort[cohort.model.eq(m)]
    products = {}
    catalog_indices = {}
    catalog_vectors = {}
    catalog_checks = []
    for ds in DATASETS:
        ps, _, _ = load(ds)
        products[ds] = {str(p["product_id"]): p for p in ps}
        aliases = old_alias[
            old_alias.dataset.eq(ds) & old_alias.role.eq("catalog")
        ].set_index("item_id")
        catalog_indices[ds] = {str(p["product_id"]): i for i, p in enumerate(ps)}
        vix = []
        for p in ps:
            r = aliases.loc[str(p["product_id"])]
            assert r.text_sha256 == digest(REP._plain(p)), (ds, p["product_id"])
            vix.append(int(r.vector_index))
        catalog_vectors[ds] = old[np.array(vix)]
        catalog_checks.append(
            dict(
                dataset=ds,
                catalog_size=len(ps),
                all_c0_serialization_hashes_match=True,
                fixed_catalog_order_sha256=digest(
                    canonical_json(list(catalog_indices[ds]))
                ),
            )
        )
    qrows = []
    for ds, g in queries.groupby("dataset"):
        enc = tok(g["query"].tolist(), truncation=True, max_length=engine.cap)
        full = tok(g["query"].tolist(), truncation=False, verbose=False)
        for i, r in enumerate(g.itertuples()):
            qrows.append(
                dict(
                    dataset=ds,
                    model=m,
                    query_id=int(r.query_id),
                    product_id="",
                    variant="query",
                    positions="[]",
                    text=r.query,
                    text_sha256=digest(r.query),
                    input_sha256=input_hash(enc, i),
                    tokens=len(full["input_ids"][i]),
                    bucket=bucket(len(full["input_ids"][i]), engine.cap),
                )
            )
    qplan = pd.DataFrame(qrows)
    aliases = pd.concat([plan, qplan], ignore_index=True)
    # Re-tokenize every planned input and check aliases, including source C0 identity.
    for start in range(0, len(aliases), 256):
        part = aliases.iloc[start : start + 256]
        enc = tok(part.text.tolist(), truncation=True, max_length=engine.cap)
        assert [
            input_hash(enc, i) for i in range(len(part))
        ] == part.input_sha256.tolist()
    aliases["origin"] = np.where(
        aliases.input_sha256.isin(old_map), reference_origin, "adjacent_new"
    )
    unique = (
        aliases[aliases.origin.eq("adjacent_new")]
        .drop_duplicates("input_sha256")
        .sort_values(["bucket", "input_sha256"])
        .reset_index(drop=True)
    )
    unique["vector_index"] = np.arange(len(unique))
    new_map = unique.set_index("input_sha256").vector_index.to_dict()
    aliases["vector_index"] = [
        old_map[h] if h in old_map else new_map[h] for h in aliases.input_sha256
    ]
    alias_path = HERE / f"data/{m}_input_aliases.parquet"
    aliases.drop(columns=["text"]).to_parquet(alias_path, index=False)
    manifest = dict(
        freeze_sha256=sha(HERE / "freeze_manifest.json"),
        reference_vectors_sha256=refhash,
        reference_origin=reference_origin,
        unique_inputs=len(set(aliases.input_sha256)),
        new_inputs=len(unique),
        new_query_inputs=len(set(qplan.input_sha256) - old_map.keys()),
        aliases_sha256=sha(alias_path),
        new_plan_sha256=digest(
            canonical_json(
                unique[
                    ["input_sha256", "bucket", "text_sha256", "vector_index"]
                ].to_dict("records")
            )
        ),
    )
    if (folder / "plan.json").exists():
        assert read(folder / "plan.json") == manifest
    else:
        dump(folder / "plan.json", manifest)
    shape = (len(unique), old.shape[1])
    path = folder / "vectors.npy"
    new = (
        np.load(path, mmap_mode="r+")
        if path.exists()
        else np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shape)
    )
    assert new.shape == shape

    def vector(h):
        return old[old_map[h]] if h in old_map else new[new_map[h]]

    def encode(part, stage):
        for b, g in part.groupby("bucket", sort=True):
            for start in range(0, len(g), 1024):
                block = g.iloc[start : start + 1024]
                ix = block.vector_index.to_numpy()
                key = digest(canonical_json(ix.tolist()))[:20]
                marker = folder / f"{stage}_{b}_{key}.json"
                if marker.exists():
                    assert (
                        read(marker)["vector_sha256"]
                        == hashlib.sha256(new[ix].tobytes()).hexdigest()
                    )
                    continue
                started = time.time()
                for pos in range(0, len(block), engine.bs):
                    pp = block.iloc[pos : pos + engine.bs]
                    new[pp.vector_index.to_numpy()] = engine.forward(
                        pp.text.tolist(), b
                    )
                new.flush()
                dump(
                    marker,
                    dict(
                        rows=len(block),
                        bucket=int(b),
                        elapsed_seconds=time.time() - started,
                        vector_sha256=hashlib.sha256(new[ix].tobytes()).hexdigest(),
                        completed_utc=now(),
                    ),
                )
                print(
                    m,
                    stage,
                    "bucket",
                    b,
                    "block",
                    start + len(block),
                    "/",
                    len(g),
                    "elapsed",
                    round(time.time() - t),
                    flush=True,
                )

    qhashes = set(qplan.input_sha256)
    encode(unique[unique.input_sha256.isin(qhashes)], "queries")
    qmap = {(r.dataset, int(r.query_id)): r.input_sha256 for r in qplan.itertuples()}
    pmap = {(r.dataset, r.product_id, r.variant): r for r in plan.itertuples()}
    c0 = plan[plan.variant.eq("C0")].drop_duplicates("input_sha256")
    assert all(h in old_map for h in c0.input_sha256)
    # No fresh swap forward or swap score occurs before this complete C0 replay gate.
    replay_path = folder / "c0_replay.npy"
    c0 = c0.sort_values("input_sha256").reset_index(drop=True)
    if replay_path.exists():
        replay = np.load(replay_path)
        rmeta = read(folder / "c0_replay.json")
        assert (
            sha(replay_path) == rmeta["sha256"]
            and rmeta["input_order"] == c0.input_sha256.tolist()
        )
    else:
        fresh = engine.frame(c0)
        replay = np.stack([fresh[h] for h in c0.input_sha256])
        np.save(replay_path, replay)
        dump(
            folder / "c0_replay.json",
            dict(
                sha256=sha(replay_path),
                input_order=c0.input_sha256.tolist(),
                completed_utc=now(),
            ),
        )
    replay_map = dict(zip(c0.input_sha256, replay))
    baseline = []
    replay_results = []
    for (ds, qid), g in cohort.groupby(["dataset", "query_id"], sort=True):
        qvec = vector(qmap[ds, int(qid)])
        base = dot(catalog_vectors[ds], qvec)
        indices = g.catalog_index.to_numpy(dtype=int)
        saved = base[indices]
        fresh = np.stack(
            [
                replay_map[pmap[ds, r.product_id, "C0"].input_sha256]
                for r in g.itertuples()
            ]
        )
        fresh_scores = dot(fresh, qvec)
        ranks = replacement(base, indices, saved)
        repeat_ranks = replacement(base, indices, fresh_scores)
        order = np.argsort(-base, kind="stable")
        positions = np.empty(len(base), dtype=int)
        positions[order] = np.arange(1, len(base) + 1)
        assert np.array_equal(ranks, positions[indices])
        for i, r in enumerate(g.itertuples()):
            row = dict(
                dataset=ds,
                model=m,
                query_id=int(qid),
                product_id=r.product_id,
                catalog_index=int(r.catalog_index),
                c0_score=float(saved[i]),
                c0_rank=int(ranks[i]),
                distinct_swaps=int(r.distinct_swaps),
                attribute_count=int(r.attribute_count),
                c0_tokens=int(getattr(r, "c0_tokens_" + m)),
                max_tokens=int(getattr(r, "max_tokens_" + m)),
                context_limit=engine.cap,
                case_hash=r.case_hash,
            )
            for k in [20, 100]:
                j, s = boundary(base, int(r.catalog_index), k)
                row[f"threshold_index_{k}"] = j
                row[f"threshold_score_{k}"] = float(s)
                row[f"c0_margin_{k}"] = float(np.float64(saved[i]) - np.float64(s))
                row[f"c0_included_{k}"] = int(ranks[i] <= k)
                row[f"c0_tie_{k}"] = bool(saved[i] == s)
                assert (ranks[i] <= k) == (
                    saved[i] > s or saved[i] == s and r.catalog_index < j
                )
            baseline.append(row)
            replay_results.append(
                dict(
                    dataset=ds,
                    model=m,
                    query_id=int(qid),
                    product_id=r.product_id,
                    c0_score=float(saved[i]),
                    replay_score=float(fresh_scores[i]),
                    score_discrepancy=float(
                        np.float64(fresh_scores[i]) - np.float64(saved[i])
                    ),
                    c0_rank=int(ranks[i]),
                    replay_rank=int(repeat_ranks[i]),
                    vector_max_abs_error=float(
                        abs(
                            fresh[i] - vector(pmap[ds, r.product_id, "C0"].input_sha256)
                        ).max()
                    ),
                    decision20_changed=bool(
                        (ranks[i] <= 20) != (repeat_ranks[i] <= 20)
                    ),
                    decision100_changed=bool(
                        (ranks[i] <= 100) != (repeat_ranks[i] <= 100)
                    ),
                )
            )
    baseline = pd.DataFrame(baseline)
    repframe = pd.DataFrame(replay_results)
    baseline.to_parquet(HERE / f"data/{m}_baseline.parquet", index=False)
    csv(repframe, HERE / f"qa/{m}_c0_replay.csv")
    assert (
        not repframe[["decision20_changed", "decision100_changed"]].any().any()
    ), "C0 replay changed inclusion: resolve before sweep."
    dump(
        HERE / f"qa/{m}_pre_sweep_gate.json",
        dict(
            completed_utc=now(),
            passed=True,
            replayed_pairs=len(repframe),
            distinct_replayed_inputs=len(c0),
            maximum_score_discrepancy=float(repframe.score_discrepancy.abs().max()),
            maximum_vector_discrepancy=float(repframe.vector_max_abs_error.max()),
            rank_discrepancies=int(repframe.c0_rank.ne(repframe.replay_rank).sum()),
            catalog_checks=catalog_checks,
            cache_compatibility=prof["compatibility_checks"],
        ),
    )
    print(m, "C0 REPLAY PASSED", len(repframe), "pairs; begin swap sweep", flush=True)
    encode(unique[~unique.input_sha256.isin(qhashes)], "swaps")
    assert np.isfinite(new).all()
    results = []
    bmap = baseline.set_index(["dataset", "query_id", "product_id"])
    for (ds, qid), g in cohort.groupby(["dataset", "query_id"], sort=True):
        qvec = vector(qmap[ds, int(qid)])
        base = dot(catalog_vectors[ds], qvec)
        for target in g.itertuples():
            b = bmap.loc[ds, int(qid), target.product_id].to_dict()
            b.pop("model")
            pp = plan[
                plan.dataset.eq(ds)
                & plan.product_id.eq(target.product_id)
                & plan.variant.ne("C0")
            ]
            vs = np.stack([vector(h) for h in pp.input_sha256])
            scores = dot(vs, qvec)
            ranks = replacement(base, np.full(len(pp), target.catalog_index), scores)
            for i, r in enumerate(pp.itertuples()):
                row = dict(
                    dataset=ds,
                    model=m,
                    query_id=int(qid),
                    product_id=target.product_id,
                    variant=r.variant,
                    positions=r.positions,
                    text_sha256=r.text_sha256,
                    input_sha256=r.input_sha256,
                    tokens=int(r.tokens),
                    **b,
                    swap_score=float(scores[i]),
                    swap_rank=int(ranks[i]),
                    rank_change=int(ranks[i]) - int(b["c0_rank"]),
                    swap_hash=selection_hash(
                        "case_swap",
                        ds,
                        m,
                        qid,
                        target.product_id,
                        r.positions,
                        r.text_sha256,
                    ),
                )
                for k in [20, 100]:
                    row[f"swap_margin_{k}"] = float(
                        np.float64(scores[i]) - np.float64(b[f"threshold_score_{k}"])
                    )
                    row[f"swap_included_{k}"] = int(ranks[i] <= k)
                    row[f"swap_tie_{k}"] = bool(scores[i] == b[f"threshold_score_{k}"])
                    row[f"flip_{k}"] = bool((ranks[i] <= k) != b[f"c0_included_{k}"])
                    row[f"confirmed_{k}"] = False
                    row[f"unresolved_{k}"] = False
                    assert (ranks[i] <= k) == (
                        scores[i] > b[f"threshold_score_{k}"]
                        or scores[i] == b[f"threshold_score_{k}"]
                        and target.catalog_index < b[f"threshold_index_{k}"]
                    )
                results.append(row)
    f = pd.DataFrame(results)
    f.to_parquet(HERE / f"data/{m}_observed_variants.parquet", index=False)
    crossing = f[f.flip_20 | f.flip_100]
    needed = {}
    for r in crossing.itertuples():
        for v in ["C0", r.variant]:
            entry = pmap[r.dataset, r.product_id, v]
            needed[entry.input_sha256] = entry
    repeats = {}
    repeat_rows = []
    for n, (h, r) in enumerate(sorted(needed.items())):
        marker = folder / ("repeat_" + h + ".npz")
        if marker.exists():
            payload = np.load(marker)
            arr = payload["vectors"]
            assert str(payload["input_hash"]) == h and arr.shape == (3, old.shape[1])
        else:
            arr = np.stack([engine.forward([r.text], r.bucket)[0] for _ in range(3)])
            np.savez_compressed(marker, vectors=arr, input_hash=np.array(h))
        repeats[h] = arr
        if n % 100 == 0:
            print(m, "CONFIRM FORWARDS", n, "/", len(needed), flush=True)
    # Rank each repeat against the exact same fixed competitor score vector.
    for (ds, qid), g in crossing.groupby(["dataset", "query_id"], sort=True):
        qvec = vector(qmap[ds, int(qid)])
        base = dot(catalog_vectors[ds], qvec)
        for r in g.itertuples():
            h0 = pmap[ds, r.product_id, "C0"].input_sha256
            hv = r.input_sha256
            s0 = dot(repeats[h0], qvec)
            sv = dot(repeats[hv], qvec)
            r0 = replacement(base, np.full(3, r.catalog_index), s0)
            rv = replacement(base, np.full(3, r.catalog_index), sv)
            for k in [20, 100]:
                if getattr(r, f"flip_{k}"):
                    confirmed = bool(
                        np.all((r0 <= k) == bool(getattr(r, f"c0_included_{k}")))
                        & np.all((rv <= k) == bool(getattr(r, f"swap_included_{k}")))
                    )
                    f.loc[r.Index, f"confirmed_{k}"] = confirmed
                    f.loc[r.Index, f"unresolved_{k}"] = not confirmed
            for j in range(3):
                rr = dict(
                    dataset=ds,
                    model=m,
                    query_id=int(qid),
                    product_id=r.product_id,
                    variant=r.variant,
                    repetition=j + 1,
                    saved_c0_score=r.c0_score,
                    saved_swap_score=r.swap_score,
                    repeated_c0_score=float(s0[j]),
                    repeated_swap_score=float(sv[j]),
                    c0_score_discrepancy=float(np.float64(s0[j]) - r.c0_score),
                    swap_score_discrepancy=float(np.float64(sv[j]) - r.swap_score),
                    repeated_c0_rank=int(r0[j]),
                    repeated_swap_rank=int(rv[j]),
                    c0_vector_max_error=float(abs(repeats[h0][j] - vector(h0)).max()),
                    swap_vector_max_error=float(abs(repeats[hv][j] - vector(hv)).max()),
                )
                for k in [20, 100]:
                    rr[f"c0_margin_{k}"] = float(
                        np.float64(s0[j]) - getattr(r, f"threshold_score_{k}")
                    )
                    rr[f"swap_margin_{k}"] = float(
                        np.float64(sv[j]) - getattr(r, f"threshold_score_{k}")
                    )
                    rr[f"c0_included_{k}"] = bool(r0[j] <= k)
                    rr[f"swap_included_{k}"] = bool(rv[j] <= k)
                repeat_rows.append(rr)
    f.to_parquet(HERE / f"data/{m}_variants.parquet", index=False)
    repeats_df = pd.DataFrame(repeat_rows)
    repeats_df.to_parquet(HERE / f"data/{m}_numerical_repeats.parquet", index=False)
    report = dict(
        completed_utc=now(),
        model=m,
        pair_variants=len(f),
        observed_crossing_variants=len(crossing),
        repeated_distinct_inputs=len(needed),
        new_forward_inputs=len(unique),
        reused_unique_inputs=len(set(aliases.input_sha256) & old_map.keys()),
        numerical_repeat_rows=len(repeat_rows),
        repetitions=3,
        encoding_plan=manifest,
        maximum_replay_score_error=float(repframe.score_discrepancy.abs().max()),
        maximum_crossing_c0_score_error=(
            float(repeats_df.c0_score_discrepancy.abs().max())
            if len(repeats_df)
            else 0.0
        ),
        maximum_crossing_swap_score_error=(
            float(repeats_df.swap_score_discrepancy.abs().max())
            if len(repeats_df)
            else 0.0
        ),
        counts={
            str(k): dict(
                observed=int(f[f"flip_{k}"].sum()),
                confirmed=int(f[f"confirmed_{k}"].sum()),
                unresolved=int(f[f"unresolved_{k}"].sum()),
            )
            for k in [20, 100]
        },
        elapsed_seconds=time.time() - t,
        new_vectors_sha256=sha(path),
        results_sha256=sha(HERE / f"data/{m}_variants.parquet"),
    )
    dump(HERE / f"qa/{m}_numerical_report.json", report)
    dump(folder / "complete.json", report)
    print("COMPLETE", m, report["counts"], flush=True)


if __name__ == "__main__":
    main()
