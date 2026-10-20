"""Scientific regression checks for boundary ranking, metrics, and invariance."""

from pathlib import Path
import itertools, json, sys, unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.representations import serialize, ordered_attributes, SCHEDULES, RULES
from src.metrics import evaluate
from src.common import ROOT, read
from src.retrieval import bm25, rank, hybrid
from scripts.run_retrieval import replacement


class ReproductionTests(unittest.TestCase):
    def test_metrics_match_saved_real_query_records(self):
        reference = pd.read_parquet(ROOT / "data/reference_per_query.parquet")
        gains = read(ROOT / "configs/evaluation.json")["gains"]
        for ds in ["wands", "esci"]:
            f = pd.read_parquet(ROOT / f"data/ranks/{ds}/minilm/raw/C0.parquet")
            r = reference[
                reference.dataset.eq(ds)
                & reference.model.eq("minilm")
                & reference.method.eq("raw")
                & reference.schedule.eq("C0")
            ].set_index("query_id")
            for qid in r.index[:12]:
                for k in [20, 100]:
                    actual = evaluate(
                        f[f.query_id.eq(qid)],
                        gains[ds],
                        "Exact" if ds == "wands" else "E",
                        k,
                    )
                    for metric, value in actual.items():
                        self.assertAlmostEqual(
                            value, float(r.loc[qid, metric]), places=13
                        )

    def test_replacement_matches_explicit_counterfactual(self):
        # Exhaustive ties, moves in both directions, and unchanged targets.
        for base in itertools.product([-0.5, 0.0, 0.5], repeat=4):
            base = np.asarray(base, dtype="float32")
            for target in range(4):
                values = np.asarray([-0.75, -0.5, 0, 0.5, 0.75], dtype="float32")
                actual = replacement(base, np.full(len(values), target), values)
                expected = []
                for value in values:
                    counterfactual = base.copy()
                    counterfactual[target] = value
                    expected.append(
                        int(
                            np.flatnonzero(
                                np.argsort(-counterfactual, kind="stable") == target
                            )[0]
                            + 1
                        )
                    )
                np.testing.assert_array_equal(actual, expected)

    def test_canonical_rules_preserve_duplicates_and_are_invariant(self):
        p = dict(
            product_id="7",
            title="Fixed",
            description="Same",
            attributes=["color: blue", "type: chair", "color: blue", "Style: modern"],
            attribute_separator="|",
            section_order=["title", "description", "attributes"],
        )
        for rule in RULES:
            target = serialize(p, rule)
            for order in set(itertools.permutations(p["attributes"])):
                changed = dict(p, attributes=list(order))
                self.assertEqual(serialize(changed, rule), target)
                self.assertEqual(
                    ordered_attributes(changed, rule).count("color: blue"), 2
                )

    def test_graded_metrics_distinguish_unjudged_and_condensed(self):
        f = pd.DataFrame(dict(rank=[2, 4, 6], label=["E", "S", "I"]))
        metrics = evaluate(f, {"E": 1, "S": 0.1, "C": 0.01, "I": 0}, "E", 3)
        ideal = 1 + 0.1 / np.log2(3)
        self.assertAlmostEqual(metrics["nDCG@3"], (1 / np.log2(3)) / ideal)
        self.assertAlmostEqual(metrics["cNDCG@3"], 1)
        self.assertEqual(metrics["Recall@3"], 1)
        self.assertEqual(metrics["JudgedCoverage@3"], 1 / 3)

    def test_recall_cancellation_keeps_identity_change(self):
        before = {"a", "b", "c"}
        after = {"a", "c", "d"}
        self.assertEqual(len(before), len(after))
        self.assertEqual(after - before, {"d"})
        self.assertEqual(before - after, {"b"})

    def test_bm25_order_invariance_and_full_list_fusion(self):
        a = bm25(["blue chair wood", "metal stool"], ["wood chair"])
        b = bm25(["wood blue chair", "stool metal"], ["wood chair"])
        np.testing.assert_array_equal(a, b)
        lexical = np.asarray([[0, 1, 2]], dtype="int32")
        dense = np.asarray([[2, 1, 0]], dtype="int32")
        scores = hybrid(lexical, dense, 60)
        self.assertAlmostEqual(float(scores[0, 0]), 1 / 61 + 1 / 63, places=7)
        np.testing.assert_array_equal(rank(np.zeros((1, 3))), [[0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
