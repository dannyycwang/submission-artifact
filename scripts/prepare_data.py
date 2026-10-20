"""Reconstruct and compare the supplied evaluation records from upstream raw files."""

from __future__ import annotations
from pathlib import Path
import argparse, gzip, hashlib, json, sys
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, load, read

RAW = Path(".")


def clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def wands(cfg: dict) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    products = pd.read_csv(RAW / "product.csv", sep="\t").sort_values("product_id")
    queries = pd.read_csv(RAW / "query.csv", sep="\t").sort_values("query_id")
    labels = pd.read_csv(RAW / "label.csv", sep="\t")
    conflicts = labels.groupby(["query_id", "product_id"]).label.nunique()
    bad = set(conflicts[conflicts > 1].index)
    mask = labels.set_index(["query_id", "product_id"]).index.isin(bad)
    labels = labels[~mask].drop_duplicates(["query_id", "product_id"]).copy()
    labels["grade"] = labels.label.map(cfg["wands_gains"])
    records = []
    for row in products.itertuples(index=False):
        item = row._asdict()
        raw_features = clean(item["product_features"])
        records.append(
            {
                "product_id": int(item["product_id"]),
                "title": clean(item["product_name"]),
                "class": clean(item["product_class"]),
                "category": clean(
                    item["_3"] if "_3" in item else item.get("category hierarchy", "")
                ),
                "description": clean(item["product_description"]),
                "attributes": raw_features.split("|") if raw_features else [],
                "attribute_separator": "|",
                "section_order": [
                    "title",
                    "class",
                    "category",
                    "description",
                    "attributes",
                ],
            }
        )
    queries = queries[["query_id", "query"]].copy()
    return records, queries, labels[["query_id", "product_id", "label", "grade"]]


def esci(cfg: dict) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    base = RAW
    spec = cfg["datasets"]["esci"]
    examples = pd.read_parquet(
        base / "shopping_queries_dataset_examples.parquet",
        filters=[
            ("product_locale", "==", spec["locale"]),
            (spec["version_flag"], "==", 1),
            ("split", "==", spec["split"]),
        ],
    )
    query_ids = sorted(
        examples.query_id.unique(),
        key=lambda q: hashlib.sha256(f"{cfg['seed']}:{int(q)}".encode()).hexdigest(),
    )[: spec["query_sample_size"]]
    examples = examples[examples.query_id.isin(query_ids)].copy()
    assert not examples.duplicated(["query_id", "product_id"]).any()
    product_ids = sorted(examples.product_id.unique())
    dataset = ds.dataset(
        base / "shopping_queries_dataset_products.parquet", format="parquet"
    )
    table = dataset.to_table(
        filter=(ds.field("product_locale") == spec["locale"])
        & ds.field("product_id").isin(product_ids)
    )
    products = table.to_pandas().drop_duplicates("product_id").set_index("product_id")
    missing = set(product_ids) - set(products.index)
    assert not missing, f"Missing {len(missing)} selected products"
    records = []
    for product_id in product_ids:
        row = products.loc[product_id]
        attributes = []
        if clean(row.product_brand):
            attributes.append(clean(row.product_brand))
        if clean(row.product_color):
            attributes.append(clean(row.product_color))
        if clean(row.product_bullet_point):
            attributes.append(clean(row.product_bullet_point))
        records.append(
            {
                "product_id": product_id,
                "title": clean(row.product_title),
                "class": "",
                "category": "",
                "description": clean(row.product_description),
                "attributes": attributes,
                "attribute_separator": "\n",
                "section_order": ["title", "description", "attributes"],
            }
        )
    queries = examples[["query_id", "query"]].drop_duplicates().sort_values("query_id")
    labels = examples[["query_id", "product_id", "esci_label"]].copy()
    labels = labels.rename(columns={"esci_label": "label"})
    labels["grade"] = labels.label.map(cfg["esci_official_task1_gains"])
    return records, queries, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["wands", "esci"], required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    RAW = args.raw.resolve()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory")
    gains = read(ROOT / "configs/evaluation.json")["gains"]
    cfg = {
        "seed": 20260904,
        "wands_gains": gains["wands"],
        "esci_official_task1_gains": gains["esci"],
        "datasets": {
            "esci": {
                "locale": "us",
                "version_flag": "small_version",
                "split": "test",
                "query_sample_size": 500,
            }
        },
    }
    products, queries, labels = (wands if args.dataset == "wands" else esci)(cfg)
    labels = labels[["query_id", "product_id", "label"]].copy()
    queries = queries[["query_id", "query"]].copy()
    expected_products, expected_queries, expected_labels = load(args.dataset)
    assert products == expected_products, "Processed product records differ"
    pd.testing.assert_frame_equal(
        queries.reset_index(drop=True),
        expected_queries[["query_id", "query"]].reset_index(drop=True),
        check_dtype=False,
    )
    labels["product_id"] = labels.product_id.astype(str)
    keys = ["query_id", "product_id"]
    columns = keys + ["label"]
    pd.testing.assert_frame_equal(
        labels[columns].sort_values(keys).reset_index(drop=True),
        expected_labels[columns].sort_values(keys).reset_index(drop=True),
        check_dtype=False,
    )
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(
        out / f"{args.dataset}_products.jsonl.gz", "wt", encoding="utf-8", newline="\n"
    ) as f:
        for record in products:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    queries.to_csv(out / f"{args.dataset}_queries.csv", index=False)
    labels.to_csv(out / f"{args.dataset}_judgments.csv", index=False)
    print(
        f"PASS: {len(products)} product records, {len(queries)} queries, and {len(labels)} labels match the supplied data."
    )
