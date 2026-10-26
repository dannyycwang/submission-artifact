"""Portable inputs, immutable run profiles, and verified inference checkpoints."""

from pathlib import Path
import gzip, hashlib, importlib.metadata, json, os, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
MODEL = os.environ.get("ARTIFACT_RECENT_MODEL", "embeddinggemma2")
HERE = (
    Path(
        os.environ.get("ARTIFACT_RECENT_OUTPUT", ROOT / "outputs/recent_models")
    ).resolve()
    / MODEL
)
sys.path.insert(0, str(ROOT))
from src import representations as serializer
from scripts.run_retrieval import replacement as rank_target

threadpool_limits(4)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
STEMS = serializer.SCHEDULES
RULES = serializer.RULES
SEED, DRAWS = 2026091701, 10000


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


EVALUATION = read(ROOT / "configs/evaluation.json")
GAINS = EVALUATION["gains"]
SPLITS = {"wands_dev": EVALUATION["wands_development"]}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return digest(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


def dump(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE), path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def source(path):
    path = Path(path).resolve()
    return dict(
        path=path.relative_to(HERE).as_posix(),
        sha256=sha(path),
        bytes=path.stat().st_size,
    )


def highest(dataset):
    return "Exact" if dataset == "wands" else "E"


def load(dataset):
    with gzip.open(
        ROOT / f"data/catalogs/{dataset}_products.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        products = list(map(json.loads, stream))
    queries = pd.read_csv(
        ROOT / f"data/catalogs/{dataset}_queries.csv", keep_default_na=False
    )
    judgments = pd.read_csv(
        ROOT / f"data/catalogs/{dataset}_judgments.csv", dtype={"product_id": str}
    )
    return products, queries, judgments


def support(dataset):
    products, queries, judgments = load(dataset)
    ids = EVALUATION["eligible_queries"][dataset]
    return (
        products,
        queries[queries.query_id.isin(ids)]
        .sort_values("query_id")
        .reset_index(drop=True),
        judgments[judgments.query_id.isin(ids)].copy(),
    )


def texts_for(products, condition):
    return [serializer.serialize(product, condition) for product in products]


def protocol():
    profile = read(ROOT / "configs/recent_models.json")[MODEL]
    return dict(
        models={MODEL: profile["model"]},
        active_models=[MODEL],
        software=profile["software"],
        technical_tests=profile["technical_tests"],
        boundary_audit=profile["boundary_audit"],
    )


def initialize():
    design = protocol()
    path = HERE / "PROTOCOL.json"
    if path.exists():
        assert read(path) == design, "Profile changed: use a new output directory."
    else:
        dump(path, design)
    for package, expected in design["software"].items():
        actual = importlib.metadata.version(package)
        assert actual == expected, f"{package}: expected {expected}, found {actual}"
    import torch

    runtime = dict(
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cuda=torch.version.cuda,
        python=sys.version.split()[0],
        software=design["software"],
    )
    runtime_path = HERE / "runtime.json"
    if runtime_path.exists():
        assert (
            read(runtime_path) == runtime
        ), "Runtime changed: use a new output directory."
    else:
        dump(runtime_path, runtime)


def status(stage, model="all", dataset="all", state="completed", **extra):
    dump(
        HERE / f"status/{model}_{dataset}_{stage}.json",
        dict(
            stage=stage,
            model=model,
            dataset=dataset,
            status=state,
            updated_utc=now(),
            **extra,
        ),
    )


def complete(path, inputs, outputs, **extra):
    dump(
        path,
        dict(
            status="completed",
            completed_utc=now(),
            inputs_fingerprint=fingerprint(inputs),
            outputs=[source(p) for p in outputs],
            **extra,
        ),
    )


def valid(path, inputs):
    if not Path(path).exists():
        return False
    record = read(path)
    return (
        record.get("status") == "completed"
        and record.get("inputs_fingerprint") == fingerprint(inputs)
        and all(
            (HERE / f["path"]).is_file() and sha(HERE / f["path"]) == f["sha256"]
            for f in record["outputs"]
        )
    )
