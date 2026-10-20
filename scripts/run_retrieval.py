"""Rerun frozen encoders, catalog retrieval, controls, and target-only ranks."""

from pathlib import Path
import argparse, hashlib, importlib.metadata, json, platform, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.common import ROOT, load, read, dump, support
from src.encoding import DenseEncoder
from src.representations import serialize, _plain, SCHEDULES, RULES
from src.retrieval import bm25, rank, inverse, hybrid


def norm(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def replacement(base, indices, values):
    order = np.argsort(-base, kind="stable")
    keys = np.empty(len(base), dtype=[("negative", "f4"), ("index", "i8")])
    keys["negative"] = -base[order]
    keys["index"] = order
    needles = np.empty(len(values), dtype=keys.dtype)
    needles["negative"] = -values
    needles["index"] = indices
    return 1 + np.searchsorted(keys, needles) - (base[indices] > values).astype(int)


def run(args):
    import torch

    torch.set_num_threads(4)
    cfg = read(ROOT / "configs/models.json")
    spec = next(s for s in cfg["models"] if s["key"] == args.model)
    if args.batch_size:
        cfg["encoding"]["batch_size_" + args.model] = args.batch_size
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Use a new, empty output directory. Saved evidence is immutable."
        )
    if (
        out == ROOT
        or out.is_relative_to(ROOT / "data")
        or out.is_relative_to(ROOT / "reference")
    ):
        raise ValueError("Output must not overwrite package evidence.")
    out.mkdir(parents=True, exist_ok=True)
    env = {
        p: importlib.metadata.version(p)
        for p in ["numpy", "torch", "transformers", "pandas"]
    }
    dump(
        out / "run.json",
        dict(
            model=spec,
            config=cfg,
            versions=env,
            python=platform.python_version(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            mode="fresh inference; numerical agreement with historical results must be measured",
            smoke=args.smoke,
        ),
    )
    encoder = DenseEncoder(spec, cfg, out / "embeddings")
    profile = {
        "key": "native",
        "chunk_tokens": None,
        "overlap": 0,
        "pooling": "model_native",
    }

    def encode(texts, name):
        digest = hashlib.sha256("\0".join(texts).encode()).hexdigest()
        return np.asarray(encoder.encode(texts, name, profile, digest), dtype="float32")

    if args.smoke:
        p, q, _ = load(args.dataset)
        texts = [q["query"].iloc[0]] + [
            serialize(p[0], s) for s in ["C0", "C1", "lexical_ascending"]
        ]
        v = encode(texts, "smoke")
        assert v.shape == (4, encoder.hidden_size) and np.isfinite(v).all()
        dump(
            out / "smoke.json",
            dict(
                passed=True,
                shape=list(v.shape),
                unit_norms=np.linalg.norm(v, axis=1).tolist(),
                benchmark_results=False,
            ),
        )
        encoder.close()
        return
    products, queries, judgments = load(args.dataset)
    queries = queries[queries.query_id.isin(support(args.dataset))].reset_index(
        drop=True
    )
    judgments = judgments[judgments.query_id.isin(queries.query_id)].copy()
    pi = {str(p["product_id"]): i for i, p in enumerate(products)}
    qi = {int(q): i for i, q in enumerate(queries.query_id)}
    a = np.array([qi[int(q)] for q in judgments.query_id])
    b = np.array([pi[str(p)] for p in judgments.product_id])
    qv = encode(queries["query"].fillna("").tolist(), "queries")
    conditions = []

    def save(scores, method, schedule):
        ranks = inverse(rank(scores))
        f = judgments[["query_id", "product_id", "label"]].copy()
        f["rank"] = ranks[a, b]
        f["score"] = scores[a, b]
        path = out / "ranks" / method / (schedule + ".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(path, index=False)
        conditions.append(
            dict(
                dataset=args.dataset,
                model="lexical" if method == "BM25" else args.model,
                method=method,
                schedule=schedule,
                rank_source=path.relative_to(out).as_posix(),
            )
        )
        return ranks

    bm = bm25([serialize(p) for p in products], queries["query"].tolist())
    bmorder = rank(bm)
    save(bm, "BM25", "fixed")
    highest = "Exact" if args.dataset == "wands" else "E"
    targets = judgments[judgments.label.eq(highest)].copy()
    target_ranks = targets[["query_id", "product_id", "label"]].copy()
    base = None
    token_lengths = []
    for schedule in SCHEDULES:
        texts = [serialize(p, schedule) for p in products]
        pv = encode(texts, schedule)
        scores = qv @ pv.T
        full_ranks = save(scores, "raw", schedule)
        if schedule == "C0":
            base = scores.copy()
        save(hybrid(bmorder, rank(scores), 60), "raw_hybrid", schedule)
        target_ranks[schedule] = 0
        target_ranks["whole_" + schedule] = 0
        for qid, g in targets.groupby("query_id"):
            qindex = qi[int(qid)]
            indices = np.array([pi[str(p)] for p in g.product_id])
            target_ranks.loc[g.index, schedule] = replacement(
                base[qindex], indices, scores[qindex, indices]
            )
            target_ranks.loc[g.index, "whole_" + schedule] = full_ranks[qindex, indices]
        lens = []
        for i in range(0, len(texts), 256):
            lens.extend(
                map(
                    len,
                    encoder.tokenizer(
                        texts[i : i + 256], truncation=False, verbose=False
                    )["input_ids"],
                )
            )
        token_lengths.append(np.array(lens))
        print("COMPLETED", args.dataset, args.model, schedule, flush=True)
    target_ranks.to_parquet(out / "target_only.parquet", index=False)
    pd.DataFrame(
        dict(
            product_id=list(pi),
            max_tokens=np.max(token_lengths, axis=0),
            fully_fitting=np.max(token_lengths, axis=0) <= spec["max_tokens"],
        )
    ).to_csv(out / "token_lengths.csv", index=False)
    if args.model != "gte_modernbert":
        for rule in RULES:
            v = encode([serialize(p, rule) for p in products], rule)
            s = qv @ v.T
            save(s, rule, "fixed")
            save(hybrid(bmorder, rank(s), 60), "canonical_hybrid_" + rule, "fixed")
        unique = sorted({a for p in products for a in p["attributes"]})
        lookup = {a: i for i, a in enumerate(unique)}
        rows, cols, weights = [], [], []
        for i, p in enumerate(products):
            for attribute in p["attributes"]:
                rows.append(i)
                cols.append(lookup[attribute])
                weights.append(1 / len(p["attributes"]))
        incidence = csr_matrix(
            (np.asarray(weights, dtype="float32"), (rows, cols)),
            shape=(len(products), len(unique)),
        )
        nonattrs = [
            "\n".join(
                t
                for t in [
                    p["title"],
                    p.get("class", ""),
                    p.get("category", ""),
                    p.get("description", ""),
                ]
                if t
            )
            for p in products
        ]
        tv = encode(nonattrs, "non_attributes")
        av = encode(unique, "attributes")
        vectors = norm(0.5 * tv + 0.5 * norm(incidence @ av))
        save(qv @ vectors.T, "set_mean", "fixed")
        views = [
            encode([serialize(p, "view" + str(i)) for p in products], "view" + str(i))
            for i in [1, 2]
        ]
        save(qv @ norm(np.mean(views, axis=0)).T, "centroid_2", "fixed")
        save(np.maximum.reduce([qv @ v.T for v in views]), "max_2", "fixed")
    dump(out / "conditions.json", conditions)
    encoder.close()
    print("Fresh ranks saved to", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["wands", "esci"], required=True)
    p.add_argument(
        "--model", choices=["minilm", "bge_base", "gte_modernbert"], required=True
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Four actual inputs; validates encoding only, not paper results.",
    )
    run(p.parse_args())
