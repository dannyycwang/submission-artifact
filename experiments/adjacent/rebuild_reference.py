"""Optional reproduction route when large historical vector caches are absent.

Creates a clearly labeled new reference, with just complete C0 catalogs and the
frozen queries. Uses the frozen profile and never writes historical cache paths.
Requires the pinned processed inputs and model snapshots (see REPRODUCE.md).
"""

from common import *
from infer import Engine
import argparse, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    args = ap.parse_args()
    m = args.model
    verify_freeze()
    engine = Engine(m)
    tok = engine.tokenizer
    folder = HERE / "cache" / ("rebuilt_reference_" + m)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    queries = pd.read_csv(HERE / "data/queries.csv")
    queries = queries[queries.model.eq(m)]
    for ds in DATASETS:
        ps, _, _ = load(ds)
        records = [
            dict(
                dataset=ds,
                role="catalog",
                item_id=str(p["product_id"]),
                schedule="C0",
                text=REP._plain(p),
            )
            for p in ps
        ]
        records += [
            dict(
                dataset=ds,
                role="query",
                item_id=str(r.query_id),
                schedule="query",
                text=r.query,
            )
            for r in queries[queries.dataset.eq(ds)].itertuples()
        ]
        for start in range(0, len(records), 256):
            part = records[start : start + 256]
            texts = [r["text"] for r in part]
            capped = tok(texts, truncation=True, max_length=engine.cap)
            full = tok(texts, truncation=False, verbose=False)
            for i, r in enumerate(part):
                n = len(full["input_ids"][i])
                rows.append(
                    dict(
                        **r,
                        text_sha256=digest(r["text"]),
                        input_sha256=input_hash(capped, i),
                        tokens=n,
                        bucket=bucket(n, engine.cap),
                    )
                )
    aliases = pd.DataFrame(rows)
    unique = (
        aliases.drop_duplicates("input_sha256")
        .sort_values(["bucket", "input_sha256"])
        .reset_index(drop=True)
    )
    unique["vector_index"] = np.arange(len(unique))
    aliases["vector_index"] = aliases.input_sha256.map(
        unique.set_index("input_sha256").vector_index
    )
    planhash = digest(
        canonical_json(
            unique[["input_sha256", "vector_index", "bucket"]].to_dict("records")
        )
    )
    profile = dict(
        model=m,
        freeze_sha256=sha(HERE / "freeze_manifest.json"),
        plan_sha256=planhash,
        execution_profile_sha256=sha(HERE / "execution_profile.json"),
        reference_kind="Newly generated matched C0; not an original historical artifact",
    )
    if (folder / "plan.json").exists():
        assert read(folder / "plan.json") == profile
    else:
        dump(folder / "plan.json", profile)
    path = folder / "vectors.npy"
    shape = (len(unique), int(engine.net.config.hidden_size))
    vectors = (
        np.load(path, mmap_mode="r+")
        if path.exists()
        else np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shape)
    )
    for b, g in unique.groupby("bucket", sort=True):
        for start in range(0, len(g), 1024):
            block = g.iloc[start : start + 1024]
            ix = block.index.to_numpy()
            marker = folder / f"block_{ix[0]:07d}.json"
            if marker.exists():
                assert (
                    read(marker)["sha256"]
                    == hashlib.sha256(vectors[ix].tobytes()).hexdigest()
                )
                continue
            t = time.time()
            for pos in range(0, len(block), engine.bs):
                pp = block.iloc[pos : pos + engine.bs]
                vectors[pp.index.to_numpy()] = engine.forward(pp.text.tolist(), b)
            vectors.flush()
            dump(
                marker,
                dict(
                    sha256=hashlib.sha256(vectors[ix].tobytes()).hexdigest(),
                    rows=len(block),
                    elapsed_seconds=time.time() - t,
                ),
            )
            print("REBUILT", m, int(ix[-1]) + 1, "/", len(unique), flush=True)
    unique.drop(columns="text").to_parquet(folder / "inputs.parquet", index=False)
    aliases.drop(columns="text").to_parquet(folder / "aliases.parquet", index=False)
    dump(
        folder / "complete.json",
        dict(
            **profile,
            vectors_sha256=sha(path),
            inputs_sha256=sha(folder / "inputs.parquet"),
            aliases_sha256=sha(folder / "aliases.parquet"),
            completed_utc=now(),
        ),
    )
    print("New reference:", folder)


if __name__ == "__main__":
    main()
