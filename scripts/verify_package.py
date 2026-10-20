"""Verify distributed files and essential experiment supports before analysis."""

from pathlib import Path
import argparse, gzip, hashlib, json, re

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    count = 0
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Manifest path outside package")
        if not path.is_file():
            raise FileNotFoundError(name)
        if sha(path) != expected:
            raise ValueError("Checksum mismatch: " + name)
        count += 1
    import pandas as pd

    cfg = json.loads((ROOT / "configs/evaluation.json").read_text())
    for ds, n, q, pairs in [("wands", 42994, 308, 21299), ("esci", 10076, 499, 4434)]:
        with gzip.open(
            ROOT / f"data/catalogs/{ds}_products.jsonl.gz", "rt", encoding="utf-8"
        ) as f:
            products = list(map(json.loads, f))
        assert len(products) == n and len({str(x["product_id"]) for x in products}) == n
        labels = pd.read_csv(
            ROOT / f"data/catalogs/{ds}_judgments.csv", dtype={"product_id": str}
        )
        assert not labels.duplicated(["query_id", "product_id"]).any()
        selected = labels[
            labels.query_id.isin(cfg["eligible_queries"][ds])
            & labels.label.eq("Exact" if ds == "wands" else "E")
        ]
        assert len(selected) == pairs and selected.query_id.nunique() == q
    conditions = json.loads((ROOT / "data/conditions.json").read_text())
    assert len(conditions) == 108
    assert all((ROOT / c["rank_source"]).is_file() for c in conditions)
    for folder in ["src", "scripts", "experiments", "tests"]:
        for path in (ROOT / folder).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            assert not re.search("[\u3400-\u9fff]", source), path
    print(
        f"PASS: {count} file checksums; both catalog/support definitions; 108 rank conditions; Python syntax."
    )


if __name__ == "__main__":
    main()
