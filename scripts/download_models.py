"""Download only the exact frozen model revisions required by a chosen run."""

from pathlib import Path
import argparse, json
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["minilm", "bge_base", "gte_modernbert"],
        default=["minilm", "bge_base", "gte_modernbert"],
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for spec in json.loads((root / "configs/models.json").read_text())["models"]:
        if spec["key"] in args.models:
            path = snapshot_download(repo_id=spec["name"], revision=spec["revision"])
            print(spec["key"], spec["revision"], path)
