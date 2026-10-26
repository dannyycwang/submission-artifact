"""CPU regressions for resumable encoding and both retrieval interventions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/recent_models"))
import common as recent
import encode
import evaluate


class FakeAdapter:
    """Deterministic embeddings; no model download or GPU required."""

    def __init__(self):
        self.calls = 0
        self.spec = dict(
            dimension=3,
            max_length=100,
            prefixes={"document": ""},
            batching={"base": 2, "limits": [[100, 2]]},
        )

    def inputs(self, texts, role, truncate):
        return {
            "attention_mask": np.array(
                [[1] * len(t) + [0] * (max(map(len, texts)) - len(t)) for t in texts]
            )
        }

    def encode(self, texts, role):
        self.calls += 1
        return np.array(
            [[len(t), sum(map(ord, t)), 1] for t in texts], dtype=np.float32
        )


class RecentPipelineTests(unittest.TestCase):
    def test_deduplication_resume_and_corruption_recovery(self):
        with TemporaryDirectory(prefix="retrieval-test-") as folder, patch.object(
            recent, "HERE", Path(folder)
        ):
            path = Path(folder) / "collection"
            adapter = FakeAdapter()
            texts = ["ccc", "a", "ccc", "bb"]
            actual, lengths = encode.encode_collection(adapter, texts, "document", path)
            expected = np.array(
                [[3, 297, 1], [1, 97, 1], [3, 297, 1], [2, 196, 1]], dtype=np.float32
            )
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(lengths, [3, 1, 3, 2])
            del actual
            calls = adapter.calls
            cached, _ = encode.encode_collection(adapter, texts, "document", path)
            np.testing.assert_array_equal(cached, expected)
            del cached
            self.assertEqual(adapter.calls, calls)
            # A corrupt final file must be reconstructed from verified blocks.
            (path / "embeddings.npy").write_bytes(b"corrupt")
            recovered, _ = encode.encode_collection(adapter, texts, "document", path)
            np.testing.assert_array_equal(recovered, expected)
            del recovered
            self.assertEqual(adapter.calls, calls)
            with self.assertRaisesRegex(AssertionError, "Profile/input changed"):
                encode.encode_collection(adapter, ["changed"], "document", path)

    def test_raw_and_canonical_ranks_against_brute_force(self):
        with TemporaryDirectory(prefix="retrieval-test-") as folder:
            root = Path(folder)
            spec = dict(dimension=3, max_length=100)
            model, dataset, n = "embeddinggemma2", "esci", 120
            products = [dict(product_id=f"p{i:03}") for i in range(n)]
            queries = pd.DataFrame({"query_id": [10, 20], "query": ["first", "second"]})
            judgments = pd.DataFrame(
                [
                    (q, f"p{i:03}", label)
                    for q in [10, 20]
                    for i, label in [(0, "E"), (40, "E"), (100, "S"), (119, "I")]
                ],
                columns=["query_id", "product_id", "label"],
            )
            serial = root / "data/serialization"
            serial.mkdir(parents=True)
            pd.DataFrame(
                {
                    "product_id": [p["product_id"] for p in products],
                    "unique_serialized_views": [7] * n,
                }
            ).to_csv(serial / "esci.csv", index=False)
            query_vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            base = np.array(
                [[i // 2, (n - i) // 3, 1] for i in range(n)], dtype=np.float32
            )
            raw = [np.roll(base, s * 17, axis=0) for s in range(7)]
            canonical = [base[::-1].copy(), base.copy(), np.roll(base, 29, axis=0)]
            directory = root / "runs" / model / dataset
            design = {"models": {model: spec}}
            with patch.object(recent, "HERE", root), patch.object(
                evaluate, "HERE", root
            ), patch.object(evaluate, "ROOT", root), patch.object(
                evaluate, "protocol", return_value=design
            ), patch.object(
                evaluate, "support", return_value=(products, queries, judgments)
            ):
                recent.dump(root / "PROTOCOL.json", design)
                for name, array in [
                    ("queries", query_vectors),
                    ("raw", np.vstack(raw)),
                    ("canonical", np.vstack(canonical)),
                ]:
                    path = directory / name
                    path.mkdir(parents=True)
                    descriptor = {"collection": name}
                    recent.dump(path / "descriptor.json", descriptor)
                    np.save(path / "embeddings.npy", array)
                    np.save(
                        path / "lengths.npy", np.full(len(array), 10, dtype=np.int32)
                    )
                    recent.complete(
                        path / "complete.json",
                        descriptor,
                        [path / "embeddings.npy", path / "lengths.npy"],
                    )
                evaluate.run(model, dataset)
                evaluate.run(model, dataset, canonical=True)
                pair = pd.read_parquet(directory / "evaluation/raw/pair_ranks.parquet")
                for s, condition in enumerate(recent.STEMS):
                    scores = query_vectors @ raw[s].T
                    judged = pd.read_parquet(
                        directory / f"evaluation/raw/{condition}_all_judged.parquet"
                    )
                    for row in judged.itertuples():
                        q = [10, 20].index(row.query_id)
                        p = int(row.product_id[1:])
                        order = sorted(
                            range(n), key=lambda i: (-float(scores[q, i]), i)
                        )
                        self.assertEqual(row.rank, order.index(p) + 1)
                    for row in pair.itertuples():
                        q = [10, 20].index(row.query_id)
                        p = int(row.product_id[1:])
                        replaced = (query_vectors @ raw[0].T)[q].copy()
                        replaced[p] = scores[q, p]
                        order = sorted(range(n), key=lambda i: (-float(replaced[i]), i))
                        self.assertEqual(
                            getattr(row, "target_" + condition), order.index(p) + 1
                        )
                transitions = pd.read_parquet(
                    directory / "evaluation/canonical/query_transitions.parquet"
                )
                np.testing.assert_allclose(
                    transitions.groupby(["query_id", "rule", "K"]).fraction.sum(), 1
                )
                for s, condition in enumerate(recent.RULES):
                    judged = pd.read_parquet(
                        directory
                        / f"evaluation/canonical/{condition}_all_judged.parquet"
                    )
                    for row in judged.itertuples():
                        q = [10, 20].index(row.query_id)
                        p = int(row.product_id[1:])
                        scores = query_vectors[q] @ canonical[s].T
                        order = sorted(range(n), key=lambda i: (-float(scores[i]), i))
                        self.assertEqual(row.rank, order.index(p) + 1)


if __name__ == "__main__":
    unittest.main()
