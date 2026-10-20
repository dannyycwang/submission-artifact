"""Pinned, deterministic full-family target supplement; never mixes historical vectors.

Run --model minilm then --model bge_base. Uses the existing frozen 128-query
WANDS cohort and all ESCI targets, with a newly encoded fixed C0 catalog/queries.
Exact input aliases reuse one vector across target/source roles and datasets.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from common import *
import argparse, time, gc

OUT = HERE / "harmonized"


def input_hash(enc, i):
    return digest(
        json.dumps(
            {
                k: enc[k][i]
                for k in ["input_ids", "token_type_ids", "attention_mask"]
                if k in enc
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def replacement(base, indices, values):
    order = np.argsort(-base, kind="stable")
    keys = np.empty(len(base), dtype=[("negative", "f4"), ("index", "i8")])
    keys["negative"] = -base[order]
    keys["index"] = order
    v = np.empty(len(values), dtype=keys.dtype)
    v["negative"] = -values
    v["index"] = indices
    return 1 + np.searchsorted(keys, v) - (base[indices] > values).astype(int)


def dot(v, q):
    # The identical FP32 elementwise reduction is used for every source and target.
    return np.sum(v * q[None, :], axis=1, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["minilm", "bge_base"], required=True)
    args = ap.parse_args()
    m = args.model
    import torch, transformers
    from transformers import AutoTokenizer, AutoModel

    torch.set_num_threads(4)
    torch.manual_seed(CFG["seed"])
    torch.cuda.manual_seed_all(CFG["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    spec = next(x for x in CFG["models"] if x["key"] == m)
    cap = spec["max_tokens"]
    bs = 64 if m == "minilm" else 16
    folder = OUT / m
    folder.mkdir(parents=True, exist_ok=True)
    frozen = read(ROOT / "data/expanded/COVERAGE_PLAN_FROZEN.json")
    planpath = ROOT / "data/expanded/coverage_plan.parquet"
    assert sha(planpath) == frozen["plan_sha256"]
    profile = dict(
        version="section6-harmonized-v1",
        model=spec,
        torch=torch.__version__,
        transformers=transformers.__version__,
        tokenizers=__import__("tokenizers").__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        cap=cap,
        pooling=spec["native_pooling"],
        model_dtype="float16",
        pooling_normalization_scoring_dtype="float32",
        prefixes="",
        attention="eager",
        batch_size=bs,
        padding="smallest of 32/64/128/256/512 >= capped input length; fixed-size batches, last batch repeats last real input; no OOM fallback",
        input_order="bucket then SHA256 of native truncated input fields; globally deduplicated across queries/catalogs/targets",
        tf32=False,
        deterministic=True,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        tie="descending exact FP32 score then ascending fixed catalog index",
        scoring="numpy elementwise product and FP32 axis reduction, identical for source and targets",
        serializer_sha256=sha(ROOT / "src/representations.py"),
        frozen_plan_sha256=sha(planpath),
        support_sha256=sha(ROOT / "data/expanded/coverage_common_support.parquet"),
    )
    profile["fingerprint"] = digest(json.dumps(profile, sort_keys=True))
    if (folder / "profile.json").exists():
        assert read(folder / "profile.json") == profile
    else:
        dump(folder / "profile.json", profile)
    tok = AutoTokenizer.from_pretrained(
        spec["name"], revision=spec["revision"], local_files_only=True
    )
    allproducts = {}
    allqueries = {}
    catalogs = {}
    qframes = {}
    for ds in ["wands", "esci"]:
        p, q, _ = load(ds)
        catalogs[ds] = p
        qframes[ds] = q
        allproducts[ds] = {str(x["product_id"]): x for x in p}
        allqueries[ds] = q.set_index("query_id")["query"].to_dict()

    def textrow(r):
        return (
            str(allqueries[r.dataset][int(r.item_id)])
            if r.role == "query"
            else text_for(allproducts[r.dataset][str(r.item_id)], r.schedule)
        )

    if not (folder / "inputs.parquet").exists():
        rows = []
        for ds in ["wands", "esci"]:
            payload = [
                dict(
                    dataset=ds,
                    role="catalog",
                    item_id=str(x["product_id"]),
                    schedule="C0",
                )
                for x in catalogs[ds]
            ]
            h = pd.read_csv(
                ROOT
                / f'data/expanded/{ds}_{"coverage_support" if ds=="wands" else "primary_support"}.csv',
                dtype={"product_id": str},
            )
            qids = sorted(h.query_id.unique())
            payload += [
                dict(dataset=ds, role="query", item_id=str(x), schedule="query")
                for x in qids
            ]
            for start in range(0, len(payload), 256):
                part = payload[start : start + 256]
                texts = [
                    (
                        str(allqueries[ds][int(r["item_id"])])
                        if r["role"] == "query"
                        else text_for(allproducts[ds][r["item_id"]], "C0")
                    )
                    for r in part
                ]
                e = tok(texts, truncation=True, max_length=cap)
                full = tok(texts, truncation=False, verbose=False)
                for i, r in enumerate(part):
                    rows.append(
                        dict(
                            **r,
                            input_sha256=input_hash(e, i),
                            text_sha256=digest(texts[i]),
                            tokens=len(full["input_ids"][i]),
                        )
                    )
        plan = pd.read_parquet(planpath)
        for r in plan.itertuples():
            rows.append(
                dict(
                    dataset=r.dataset,
                    role="target",
                    item_id=str(r.product_id),
                    schedule=r.schedule,
                    input_sha256=getattr(r, "input_sha_" + m),
                    text_sha256=r.text_sha256,
                    tokens=int(getattr(r, "tokens_" + m)),
                )
            )
        records = pd.DataFrame(rows)
        unique = records.drop_duplicates("input_sha256").copy()
        unique["bucket"] = [
            next(b for b in [32, 64, 128, 256, 512] if b >= min(n, cap))
            for n in unique.tokens
        ]
        unique = unique.sort_values(["bucket", "input_sha256"]).reset_index(drop=True)
        unique["vector_index"] = np.arange(len(unique))
        records["vector_index"] = records.input_sha256.map(
            unique.set_index("input_sha256").vector_index
        )
        unique.to_parquet(folder / "inputs.parquet", index=False)
        records.to_parquet(folder / "aliases.parquet", index=False)
        dump(
            folder / "plan.json",
            dict(
                time_utc=now(),
                profile=profile["fingerprint"],
                unique_inputs=len(unique),
                alias_rows=len(records),
                inputs_sha256=sha(folder / "inputs.parquet"),
                aliases_sha256=sha(folder / "aliases.parquet"),
                script_sha256=sha(__file__),
                note="Frozen before this supplement inference; original schedules/support reused after prior outcomes.",
            ),
        )
    unique = pd.read_parquet(folder / "inputs.parquet")
    records = pd.read_parquet(folder / "aliases.parquet")
    fr = read(folder / "plan.json")
    assert (
        sha(folder / "inputs.parquet") == fr["inputs_sha256"]
        and sha(folder / "aliases.parquet") == fr["aliases_sha256"]
    )
    model = (
        AutoModel.from_pretrained(
            spec["name"],
            revision=spec["revision"],
            local_files_only=True,
            attn_implementation="eager",
        )
        .cuda()
        .half()
        .eval()
    )
    hidden = int(model.config.hidden_size)
    vectorpath = folder / "vectors.npy"
    vectors = (
        np.load(vectorpath, mmap_mode="r+")
        if vectorpath.exists()
        else np.lib.format.open_memmap(
            vectorpath, mode="w+", dtype="float32", shape=(len(unique), hidden)
        )
    )

    def forward(texts, bucket):
        n = len(texts)
        assert 0 < n <= bs
        texts = list(texts) + [texts[-1]] * (bs - n)
        e = tok(
            texts,
            padding="max_length",
            max_length=bucket,
            truncation=True,
            return_tensors="pt",
        )
        e = {k: v.cuda() for k, v in e.items()}
        with torch.inference_mode():
            hiddenstates = model(**e).last_hidden_state
            if m == "minilm":
                mask = e["attention_mask"].unsqueeze(-1)
                pooled = (hiddenstates.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                pooled = hiddenstates[:, 0].float()
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled[:n].cpu().numpy()

    blocks = []
    started = time.perf_counter()
    for bucket, g in unique.groupby("bucket", sort=True):
        for off in range(0, len(g), 2048):
            part = g.iloc[off : off + 2048]
            ix = part.index.to_numpy()
            meta = folder / f"block_{ix[0]:07d}.json"
            if meta.exists():
                r = read(meta)
                assert (
                    r["fingerprint"] == profile["fingerprint"]
                    and r["sha256"] == hashlib.sha256(vectors[ix].tobytes()).hexdigest()
                )
                blocks.append(r)
                continue
            t = time.perf_counter()
            for pos in range(0, len(part), bs):
                pp = part.iloc[pos : pos + bs]
                texts = [textrow(r) for r in pp.itertuples()]
                assert [digest(t) for t in texts] == pp.text_sha256.tolist()
                e = tok(texts, truncation=True, max_length=cap)
                assert [
                    input_hash(e, i) for i in range(len(pp))
                ] == pp.input_sha256.tolist()
                vectors[pp.index.to_numpy()] = forward(texts, int(bucket))
            vectors.flush()
            r = dict(
                first=int(ix[0]),
                last=int(ix[-1]),
                rows=len(ix),
                bucket=int(bucket),
                seconds=time.perf_counter() - t,
                fingerprint=profile["fingerprint"],
                sha256=hashlib.sha256(vectors[ix].tobytes()).hexdigest(),
            )
            dump(meta, r)
            blocks.append(r)
            print(
                m,
                "ENCODED",
                int(ix[-1]) + 1,
                "/",
                len(unique),
                "elapsed",
                round(time.perf_counter() - started),
                flush=True,
            )
    assert np.isfinite(vectors).all()
    checks = []
    probes = (
        unique.assign(probe=unique.input_sha256.map(lambda x: digest("repeat|" + x)))
        .sort_values("probe")
        .head(24)
    )
    for r in probes.itertuples():
        fresh = forward([textrow(r)], int(r.bucket))[0]
        old = vectors[r.vector_index]
        checks.append(
            dict(
                kind="hash-fixed repeated input",
                vector_index=r.vector_index,
                max_abs_vector_error=float(np.max(abs(fresh - old))),
                bitwise_identical=bool(np.array_equal(fresh, old)),
            )
        )
    print(m, "ENCODING COMPLETE; SCORE", flush=True)
    results = []
    changes = []
    changed_probes = []
    for ds in ["wands", "esci"]:
        p = catalogs[ds]
        pi = {str(x["product_id"]): i for i, x in enumerate(p)}
        a = records[records.dataset.eq(ds)]
        c = a[a.role.eq("catalog")].set_index("item_id")
        cix = c.loc[list(pi), "vector_index"].to_numpy()
        qi = a[a.role.eq("query")].set_index("item_id").vector_index.to_dict()
        plan = pd.read_parquet(planpath)
        plan = plan[plan.dataset.eq(ds)]
        target = a[a.role.eq("target")][["item_id", "schedule", "vector_index"]].rename(
            columns={"item_id": "product_id"}
        )
        h = pairread(ROOT / "data/expanded/coverage_common_support.parquet")
        h = h[h.dataset.eq(ds) & h.model.eq(m)]
        assert h.query_id.nunique() == (128 if ds == "wands" else 499) and len(h) == (
            7125 if ds == "wands" else 4434
        )
        f = h.merge(plan, on=["dataset", "product_id"], validate="many_to_many").merge(
            target, on=["product_id", "schedule"], validate="many_to_one"
        )
        f["catalog_index"] = f.product_id.map(pi)
        f["score"] = np.float32(0)
        f["rank"] = 0
        historical = pairread(ROOT / f"data/target_only/{ds}_{m}.parquet").set_index(
            ["query_id", "product_id"]
        )
        for qid, g in f.groupby("query_id", sort=True):
            qvec = vectors[qi[str(qid)]]
            base = dot(vectors[cix], qvec)
            vals = dot(vectors[g.vector_index.to_numpy()], qvec)
            ranks = replacement(base, g.catalog_index.to_numpy(), vals)
            f.loc[g.index, "score"] = vals
            f.loc[g.index, "rank"] = ranks
            # C0 replacement is the identical catalog vector, and must reproduce its rank.
            order = np.argsort(-base, kind="stable")
            position = np.empty(len(p), dtype=int)
            position[order] = np.arange(1, len(p) + 1)
            c0 = g.schedule.eq("C0").to_numpy()
            assert np.array_equal(ranks[c0], position[g.catalog_index.to_numpy()[c0]])
            for i, r in enumerate(g.itertuples()):
                if r.schedule in STEMS:
                    old = int(historical.loc[(qid, r.product_id), r.schedule])
                    new = int(ranks[i])
                    if old != new:
                        changes.append(
                            dict(
                                dataset=ds,
                                model=m,
                                query_id=int(qid),
                                product_id=r.product_id,
                                schedule=r.schedule,
                                historical_rank=old,
                                harmonized_rank=new,
                                decision20_changed=(old <= 20) != (new <= 20),
                                decision100_changed=(old <= 100) != (new <= 100),
                            )
                        )
                    if (old <= 20) != (new <= 20) or (old <= 100) != (new <= 100):
                        changed_probes.append(
                            (
                                digest(f"{ds}|{qid}|{r.product_id}|{r.schedule}"),
                                ds,
                                int(qid),
                                r.product_id,
                                r.schedule,
                                int(r.vector_index),
                                int(qi[str(qid)]),
                                int(r.catalog_index),
                                new,
                            )
                        )
        f.to_parquet(folder / f"{ds}_ranks.parquet", index=False)
        print(m, ds, "SCORED", len(f), flush=True)
    # Forward checks on up to 16 changed decisions, selected independently by hash.
    for _, ds, qid, pid, s, vi, qvi, idx, newrank in sorted(changed_probes)[:16]:
        r = unique.iloc[vi]
        fresh = forward([textrow(r)], int(r.bucket))[0]
        qvec = vectors[qvi]
        a = records[records.dataset.eq(ds) & records.role.eq("catalog")].set_index(
            "item_id"
        )
        cix = a.loc[
            [str(x["product_id"]) for x in catalogs[ds]], "vector_index"
        ].to_numpy()
        base = dot(vectors[cix], qvec)
        score = dot(fresh[None, :], qvec)
        rank = int(replacement(base, np.array([idx]), score)[0])
        checks.append(
            dict(
                kind="changed-decision repeat",
                dataset=ds,
                query_id=qid,
                product_id=pid,
                schedule=s,
                vector_index=vi,
                max_abs_vector_error=float(np.max(abs(fresh - vectors[vi]))),
                bitwise_identical=bool(np.array_equal(fresh, vectors[vi])),
                saved_rank=newrank,
                repeated_rank=rank,
                decision20_changed=(rank <= 20) != (newrank <= 20),
                decision100_changed=(rank <= 100) != (newrank <= 100),
            )
        )
    csv(pd.DataFrame(changes), folder / "original7_rank_differences.csv")
    dump(folder / "repeated_input_checks.json", checks)
    dump(
        folder / "complete.json",
        dict(
            time_utc=now(),
            profile=profile["fingerprint"],
            unique_vectors=len(vectors),
            alias_rows=len(records),
            encoding_seconds=sum(x["seconds"] for x in blocks),
            vectors_sha256=sha(vectorpath),
            all_repeated_inputs_identical=all(x["bitwise_identical"] for x in checks),
            changed_decision_checks=len(checks) - len(probes),
            new_inference_only_for_harmonized_supplement=True,
        ),
    )
    print(m, "HARMONIZATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
