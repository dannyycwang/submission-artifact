"""Exact full-catalog ranking; direct-gain graded metrics and paired query bootstrap."""

import json
import numpy as np
import pandas as pd
from common import GAINS, SEED, DRAWS, rank_target


def rank(scores):
    scores = np.asarray(scores, dtype=np.float32)
    assert scores.ndim == 1 and np.isfinite(scores).all()
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1)
    return order, ranks


def query_metrics(labels, positions, dataset, k):
    labels = np.asarray(labels)
    positions = np.asarray(positions)
    assert len(set(positions)) == len(positions) and (positions >= 1).all()
    gains = np.array([GAINS[dataset][x] for x in labels], dtype=np.float64)
    high = labels == ("Exact" if dataset == "wands" else "E")
    assert high.any()
    ideal = np.sort(gains)[::-1][:k]
    den = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    inc = positions <= k
    cg = gains[np.argsort(positions, kind="stable")][:k]
    return dict(
        Recall=float(np.sum(high & inc) / high.sum()),
        nDCG=float(np.sum(gains[inc] / np.log2(positions[inc] + 1)) / den),
        cNDCG=float(np.sum(cg / np.log2(np.arange(2, len(cg) + 2))) / den),
        judged_coverage=float(inc.sum() / k),
        n_highest=int(high.sum()),
        n_judged=len(labels),
    )


def membership(a, b, n):
    a, b = set(a), set(b)
    losses = a - b
    gains = b - a
    return dict(
        losses=len(losses),
        gains=len(gains),
        net=len(gains) - len(losses),
        turnover=len(gains) + len(losses),
        delta_recall=(len(gains) - len(losses)) / n,
        same_recall_changed_relevant_membership=bool(len(gains) == len(losses) > 0),
        gained_ids=json.dumps(sorted(gains)),
        lost_ids=json.dumps(sorted(losses)),
    )


def states(ranks, k):
    inc = np.asarray(ranks) <= k
    return np.where(
        inc.all(1),
        "persistent_inclusion",
        np.where(inc.any(1), "crossing", "persistent_omission"),
    )


def macro_states(frame, cols, k):
    f = frame[["query_id"]].copy()
    s = states(frame[cols].to_numpy(), k)
    for name in ["persistent_inclusion", "crossing", "persistent_omission"]:
        f[name] = (s == name).astype(float)
    g = f.groupby("query_id", sort=True).mean()
    g["n_highest"] = frame.groupby("query_id", sort=True).size()
    assert np.allclose(
        g[["persistent_inclusion", "crossing", "persistent_omission"]].sum(1),
        1,
        rtol=0,
        atol=1e-15,
    )
    return g.reset_index()


def bootstrap_matrix(n, seed=SEED, draws=DRAWS):
    assert n > 0
    rng = np.random.default_rng(seed)
    w = np.zeros((draws, n), dtype=np.uint16)
    for start in range(0, draws, 250):
        ix = rng.integers(0, n, size=(min(250, draws - start), n))
        np.add.at(w, (np.arange(start, start + len(ix))[:, None], ix), 1)
    return w


def interval(values, w=None):
    v = np.asarray(values, dtype=np.float64)
    assert len(v) and np.isfinite(v).all()
    if w is None:
        w = bootstrap_matrix(len(v))
    assert w.shape[1] == len(v)
    boot = w.astype(np.float64) @ v / len(v)
    return np.mean(v, axis=0), *np.quantile(boot, [0.025, 0.975], axis=0)


def competitor_boundary(scores, target, k):
    order, _ = rank(scores)
    competitors = order[order != target]
    idx = int(competitors[k - 1])
    return idx, float(scores[idx])
