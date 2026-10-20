"""Data access and paired query-cluster bootstrap for the reviewer package."""

from pathlib import Path
import datetime, gzip, hashlib, json, sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

threadpool_limits(4)
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "outputs" / "reproduced"
STEMS = ["C0", "C1", "C2s1", "C2s2", "C2s3", "C2s4", "C2s5"]
MODELS = ["minilm", "bge_base", "gte_modernbert"]
RULES = ["lexical_ascending", "lexical_descending", "field_priority_type"]
SEED, DRAWS = 2026091701, 10000


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n",
        encoding="utf-8",
    )


def csv(frame, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(p, index=False, float_format="%.17g")


def support(ds):
    return read(ROOT / "configs/evaluation.json")["eligible_queries"][ds]


def load(ds):
    with gzip.open(
        ROOT / f"data/catalogs/{ds}_products.jsonl.gz", "rt", encoding="utf-8"
    ) as f:
        p = list(map(json.loads, f))
    q = pd.read_csv(ROOT / f"data/catalogs/{ds}_queries.csv")
    j = pd.read_csv(
        ROOT / f"data/catalogs/{ds}_judgments.csv", dtype={"product_id": str}
    )
    return p, q, j


def pairread(p):
    f = pd.read_parquet(p)
    f["product_id"] = f.product_id.astype(str)
    f["query_id"] = f.query_id.astype(int)
    return f


_weights = {}


def weights(ids):
    ids = tuple(map(int, ids))
    assert list(ids) == sorted(ids)
    if ids not in _weights:
        rng = np.random.default_rng(SEED)
        w = np.zeros((DRAWS, len(ids)), dtype="float64")
        for start in range(0, DRAWS, 250):
            ix = rng.integers(0, len(ids), size=(min(250, DRAWS - start), len(ids)))
            np.add.at(w, (np.arange(start, start + len(ix))[:, None], ix), 1)
        _weights[ids] = w
    return _weights[ids]


def ci(v, ids):
    v = np.asarray(v, dtype=float)
    assert len(v) == len(ids) and np.isfinite(v).all()
    b = weights(ids) @ v / len(ids)
    return np.mean(v, axis=0), *np.quantile(b, [0.025, 0.975], axis=0)


def summarize(q, cols, meta):
    q = q.sort_values("query_id")
    e, l, h = ci(q[cols], q.query_id)
    return [
        dict(
            **meta,
            metric=c,
            estimate=float(a),
            ci_low=float(b),
            ci_high=float(d),
            queries=len(q),
            pairs=int(q.n_highest.sum()),
            weighting="query_macro",
            bootstrap_seed=SEED,
            bootstrap_resamples=DRAWS,
            interval="95% unadjusted paired query-cluster percentile",
        )
        for c, a, b, d in zip(cols, e, l, h)
    ]
