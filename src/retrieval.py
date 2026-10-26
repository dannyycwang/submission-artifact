"""Exact ranking and full-catalog lexical fusion."""

from collections import Counter
import numpy as np
from scipy import sparse
from .representations import tokens


def bm25(texts, queries, k1=1.2, b=0.75):
    vocab = {}
    rows, cols, counts, lengths = [], [], [], []
    for i, text in enumerate(texts):
        counter = Counter(tokens(text))
        lengths.append(sum(counter.values()))
        for term, count in counter.items():
            j = vocab.setdefault(term, len(vocab))
            rows.append(i)
            cols.append(j)
            counts.append(count)
    tf = sparse.csr_matrix(
        (np.asarray(counts, dtype=np.float32), (rows, cols)),
        shape=(len(texts), len(vocab)),
    )
    dl = np.asarray(lengths, dtype=np.float32)
    df = np.bincount(tf.indices, minlength=len(vocab))
    idf = np.log(1 + (len(texts) - df + 0.5) / (df + 0.5))
    norm = k1 * (1 - b + b * dl / max(dl.mean(), 1))
    weighted = tf.copy()
    weighted.data = (
        (k1 + 1)
        * tf.data
        / (tf.data + np.repeat(norm, np.diff(tf.indptr)))
        * idf[tf.indices]
    ).astype(np.float32)
    matrix = weighted.tocsc()
    scores = np.zeros((len(queries), len(texts)), dtype=np.float32)
    for i, query in enumerate(queries):
        indices = [vocab[t] for t in sorted(set(tokens(query))) if t in vocab]
        if indices:
            scores[i] = np.asarray(matrix[:, indices].sum(axis=1)).ravel()
    return scores


def rank(scores):
    # Preserve the supplied catalog order; ties use ascending catalog index.
    return np.argsort(-scores, axis=1, kind="stable").astype(np.int32)


def inverse(order):
    result = np.empty_like(order)
    np.put_along_axis(
        result,
        order,
        np.broadcast_to(np.arange(1, order.shape[1] + 1), order.shape),
        axis=1,
    )
    return result


def hybrid(bm_order, dense_order, constant):
    # Full-list RRF avoids censoring exact-pair rank movements at an arbitrary depth.
    return 1 / (constant + inverse(bm_order).astype(np.float32)) + 1 / (
        constant + inverse(dense_order).astype(np.float32)
    )
