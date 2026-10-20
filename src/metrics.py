"""Direct-gain graded metrics and Recall from all-judged catalog ranks."""

import numpy as np


def evaluate(frame, gains, highest, k):
    ranks = frame["rank"].to_numpy()
    gain = frame.label.map(gains).to_numpy()
    if np.isnan(gain).any():
        raise ValueError("Unknown relevance label")
    if len(set(ranks)) != len(ranks):
        raise ValueError("Duplicate product ranks within query")
    ideal = np.sort(gain)[::-1][:k]
    denominator = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    if denominator <= 0:
        raise ValueError("Positive-gain support is required")
    condensed = gain[np.argsort(ranks, kind="stable")][:k]
    relevant = frame.label.eq(highest).to_numpy()
    if not relevant.any():
        raise ValueError("Highest-label support is required")
    return {
        f"Recall@{k}": float(np.mean(ranks[relevant] <= k)),
        f"nDCG@{k}": float(
            np.sum(gain[ranks <= k] / np.log2(ranks[ranks <= k] + 1)) / denominator
        ),
        f"cNDCG@{k}": float(
            np.sum(condensed / np.log2(np.arange(2, len(condensed) + 2))) / denominator
        ),
        f"JudgedCoverage@{k}": float(np.sum(ranks <= k) / k),
    }
