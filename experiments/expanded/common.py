from pathlib import Path
import sys, os, hashlib, json, shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import *
from src import representations as REP
from src.representations import serialize as text_for, ordered_attributes as attrs

CFG = read(ROOT / "configs/models.json")
FINAL = ROOT
HERE = ROOT / "outputs" / "expanded"
HERE.mkdir(parents=True, exist_ok=True)


def digest(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()
