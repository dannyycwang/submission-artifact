"""Recompute the French-molding PAD-reference Integrated Gradients diagnostic."""

from pathlib import Path
import argparse, csv, hashlib, json, math, sys, traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from captum.attr import IntegratedGradients
from src.common import ROOT, read, dump, load
from src import representations as rep
from types import SimpleNamespace

common = SimpleNamespace(rep=rep, serialize=rep.serialize)


def csv_write(path, rows):
    if not rows:
        raise ValueError("Empty result table: " + str(path))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def serialize_with_spans(common, product, schedule):
    indices = list(range(len(product["attributes"])))
    if schedule == "C1":
        indices.reverse()
    text, parts = "", []
    for field, value in common.rep._blocks(
        product, [product["attributes"][i] for i in indices]
    ):
        if text:
            parts.append((len(text), len(text) + 1, "separator", "section separator"))
            text += "\n"
        if field == "attributes":
            for j, occurrence in enumerate(indices):
                if j:
                    sep = product["attribute_separator"]
                    parts.append(
                        (
                            len(text),
                            len(text) + len(sep),
                            "separator",
                            "attribute separator",
                        )
                    )
                    text += sep
                value = product["attributes"][occurrence]
                parts.append(
                    (len(text), len(text) + len(value), f"attr_{occurrence:03d}", value)
                )
                text += value
        else:
            parts.append((len(text), len(text) + len(value), "fixed_" + field, value))
            text += value
    if text != common.serialize(product, schedule):
        raise ValueError("Serializer disagreement: " + schedule)
    return text, parts


def run(out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic uses the recorded CUDA FP16/FP32 profile")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fixture = read(ROOT / "data/cases/french_molding/fixture.json")
    native = {v["schedule"]: v for v in fixture["native_records"]}
    model_spec = next(
        m for m in read(ROOT / "configs/models.json")["models"] if m["key"] == "minilm"
    )
    expected_revision = model_spec["revision"]
    products, queries, _ = load("wands")
    product_index = 12575
    product = products[product_index]
    assert str(product["product_id"]) == "12575"
    assert product["attributes"] == fixture["source_attributes"]
    q = np.load(ROOT / "data/cases/french_molding/fixed_query_embedding.npy").astype(
        "float32"
    )
    qv = torch.from_numpy(q).cuda()
    environment = dict(
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        reference="supplied fixed query and source-order competitor scores",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["name"], revision=expected_revision, local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(
            model_spec["name"], revision=expected_revision, local_files_only=True
        )
        .cuda()
        .half()
        .eval()
    )
    model.requires_grad_(False)

    def score_forward(inputs, embedding=None):
        kwargs = dict(inputs)
        if embedding is not None:
            kwargs["inputs_embeds"] = embedding
        hidden = model(**kwargs).last_hidden_state.float()
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(pooled, dim=1) @ qv

    prepared = []
    for schedule in ("C0", "C1"):
        text, spans = serialize_with_spans(common, product, schedule)
        attrs = (
            fixture["source_attributes"]
            if schedule == "C0"
            else fixture["source_attributes"][::-1]
        )
        expected_text = fixture["non_attribute_prefix"] + fixture[
            "attribute_separator"
        ].join(attrs)
        if text != expected_text:
            raise ValueError("Frozen input mismatch: " + schedule)
        (out / (schedule + "_input.txt")).write_text(
            text, encoding="utf-8", newline="\n"
        )
        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=False,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        special = encoded.pop("special_tokens_mask")[0].bool().cuda()
        inputs = {k: v.cuda() for k, v in encoded.items()}
        length = inputs["input_ids"].shape[1]
        if length != native[schedule]["token_length"] or length > 256:
            raise ValueError("Unexpected tokenization/truncation")
        inputs["position_ids"] = torch.arange(length, device="cuda")[None, :]
        dump(
            out / (schedule + "_tokenized_input.json"),
            dict(
                inputs={k: v.cpu().tolist() for k, v in inputs.items()},
                offsets=offsets,
                special_tokens_mask=special.cpu().tolist(),
                spans=spans,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            ),
        )
        with torch.no_grad():
            fresh_native = float(score_forward(inputs).item())
        prepared.append((schedule, spans, offsets, special, inputs, fresh_native))
    # Match the historical diagnostic: retain native FP16-rounded parameters,
    # represented in FP32 for gradients. Query vector is unchanged.
    model.float()
    tokens, entries, checks = [], [], []
    for schedule, spans, offsets, special, inputs, fresh_native in prepared:
        emb = model.get_input_embeddings()(inputs["input_ids"]).detach()
        keep = {k: v for k, v in inputs.items() if k != "input_ids"}

        def forward_embeddings(value):
            expanded = {k: v.expand(value.shape[0], -1) for k, v in keep.items()}
            return score_forward(expanded, value)

        with torch.no_grad():
            fid = float(score_forward(inputs).item())
            actual = float(forward_embeddings(emb).item())
        if abs(fid - actual) > 1e-6:
            raise ValueError("input_ids/inputs_embeds forward disagreement")
        old = native[schedule]
        error = abs(actual - old["score"])
        included = actual > old["competitor_threshold_score"] or (
            actual == old["competitor_threshold_score"]
            and product_index < old["competitor_threshold_index"]
        )
        fresh_included = fresh_native > old["competitor_threshold_score"] or (
            fresh_native == old["competitor_threshold_score"]
            and product_index < old["competitor_threshold_index"]
        )
        guard = (
            included == old["inclusion"]
            and fresh_included == old["inclusion"]
            and abs(old["signed_margin"]) > 5 * error
            and abs(old["signed_margin"]) > 5 * abs(fresh_native - old["score"])
        )
        ids = inputs["input_ids"][0].tolist()
        labels = []
        for (a, b), is_special in zip(offsets, special.tolist()):
            found = [
                (name, value)
                for start, end, name, value in spans
                if a >= start and b <= end and b > a
            ]
            labels.append(
                ("special", "special token")
                if is_special
                else (
                    found[0]
                    if len(found) == 1
                    else ("boundary", "token spans content boundary")
                )
            )
        ig = IntegratedGradients(forward_embeddings, multiply_by_inputs=True)
        for baseline_name in ("pad",):
            baseline = emb.clone()
            change = (~special) & inputs["attention_mask"][0].bool()
            baseline[:, change, :] = (
                0
                if baseline_name == "zero"
                else model.get_input_embeddings()
                .weight[tokenizer.pad_token_id]
                .detach()
            )
            with torch.no_grad():
                baseline_score = float(forward_embeddings(baseline).item())
            attempts = []
            try:
                for n in (64, 128, 256):
                    attribution, delta = ig.attribute(
                        emb,
                        baselines=baseline,
                        n_steps=n,
                        method="gausslegendre",
                        internal_batch_size=8,
                        return_convergence_delta=True,
                    )
                    token_values = (
                        attribution[0].sum(-1).detach().cpu().double().tolist()
                    )
                    residual = math.fsum(token_values) - (actual - baseline_score)
                    tolerance = max(1e-4, 0.01 * abs(actual - baseline_score))
                    passed = math.isfinite(residual) and abs(residual) <= tolerance
                    attempts.append(
                        dict(
                            nodes=n,
                            residual=residual,
                            tolerance=tolerance,
                            captum_delta=float(delta.item()),
                            passed=bool(passed),
                        )
                    )
                    if passed:
                        break
            except Exception as exc:
                failure = dict(
                    schedule=schedule,
                    baseline=baseline_name,
                    status="failed",
                    completeness_passed=False,
                    boundary_guard=bool(guard),
                    error=repr(exc),
                    traceback=traceback.format_exc(),
                    attempts=attempts,
                )
                checks.append(failure)
                dump(out / (schedule + "_" + baseline_name + "_qa.json"), failure)
                torch.cuda.empty_cache()
                continue
            np.savez_compressed(
                out / (schedule + "_" + baseline_name + "_features.npz"),
                attribution=attribution.detach().cpu().numpy(),
                input_embeddings=emb.cpu().numpy(),
                baseline_embeddings=baseline.cpu().numpy(),
            )
            groups = {}
            for i, ((name, value), number) in enumerate(zip(labels, token_values)):
                groups.setdefault((name, value), []).append(number)
                tokens.append(
                    dict(
                        schedule=schedule,
                        baseline=baseline_name,
                        token_index=i,
                        token_id=ids[i],
                        token=tokenizer.convert_ids_to_tokens(ids[i]),
                        start=offsets[i][0],
                        end=offsets[i][1],
                        entry_id=name,
                        entry_text=value,
                        attribution=number,
                    )
                )
            for (name, value), numbers in groups.items():
                entries.append(
                    dict(
                        schedule=schedule,
                        baseline=baseline_name,
                        entry_id=name,
                        entry_text=value,
                        attribution=math.fsum(numbers),
                    )
                )
            check = dict(
                schedule=schedule,
                baseline=baseline_name,
                nodes=n,
                status="passed" if passed and guard else "diagnostic_checks_failed",
                token_count=len(ids),
                input_ids_vs_embeds_error=abs(fid - actual),
                unchanged_special_and_padding=bool(
                    torch.equal(baseline[:, ~change], emb[:, ~change])
                ),
                native_score=old["score"],
                native_rank=old["saved_rank"],
                native_margin=old["signed_margin"],
                fresh_native_score=fresh_native,
                fresh_native_included=bool(fresh_included),
                fresh_native_error=abs(fresh_native - old["score"]),
                diagnostic_score=actual,
                diagnostic_native_error=error,
                diagnostic_included=bool(included),
                boundary_guard=bool(guard),
                baseline_score=baseline_score,
                attribution_sum=math.fsum(token_values),
                completeness_residual=residual,
                completeness_tolerance=tolerance,
                completeness_passed=bool(passed),
                captum_delta=float(delta.item()),
                attempts=attempts,
            )
            checks.append(check)
            dump(out / (schedule + "_" + baseline_name + "_qa.json"), check)
            csv_write(out / "token_attributions.csv", tokens)
            csv_write(out / "entry_attributions.csv", entries)
            print(
                schedule,
                baseline_name,
                "nodes",
                n,
                "completeness",
                passed,
                "boundary",
                guard,
                flush=True,
            )
    if any(v.get("status") == "failed" for v in checks):
        dump(
            out / "run_report.json",
            dict(
                status="inference_failed",
                plot_ready=False,
                contrast="C1 minus C0",
                checks=checks,
                environment=environment,
            ),
        )
        return 2
    aligned = []
    for baseline_name in ("pad",):
        by_schedule = {
            s: {
                r["entry_id"]: r
                for r in entries
                if r["schedule"] == s and r["baseline"] == baseline_name
            }
            for s in ("C0", "C1")
        }
        for entry_id in sorted(set(by_schedule["C0"]) | set(by_schedule["C1"])):
            a = by_schedule["C0"].get(entry_id)
            b = by_schedule["C1"].get(entry_id)
            if entry_id.startswith("attr_") and (a is None or b is None):
                raise ValueError(
                    "Attribute occurrence missing; inspect token boundary mapping"
                )
            if a and b and a["entry_text"] != b["entry_text"]:
                raise ValueError("Entry identity mismatch")
            v0 = a["attribution"] if a else 0.0
            v1 = b["attribution"] if b else 0.0
            aligned.append(
                dict(
                    baseline=baseline_name,
                    entry_id=entry_id,
                    entry_text=(a or b)["entry_text"],
                    C0=v0,
                    C1=v1,
                    C1_minus_C0=v1 - v0,
                )
            )
    csv_write(out / "aligned_entry_changes.csv", aligned)
    assert len(checks) == 2
    dump(
        out / "run_report.json",
        dict(
            status=(
                "passed"
                if all(c["status"] == "passed" for c in checks)
                else "diagnostic_checks_failed"
            ),
            checks=checks,
            environment=environment,
        ),
    )
    return 0 if all(c["status"] == "passed" for c in checks) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    sys.exit(run(parser.parse_args().output.resolve()))
