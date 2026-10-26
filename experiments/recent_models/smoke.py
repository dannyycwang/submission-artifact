"""Development-only validation and estimates, never a held-out retrieval result."""

from common import *
from adapters import NativeAdapter
from metrics import rank
import argparse, time, traceback


def run(model):
    from types import SimpleNamespace

    args = SimpleNamespace(model=model)
    design = (
        protocol()
        if (HERE / "PROTOCOL.json").exists()
        else read(HERE / "audit/protocol_candidate.json")
    )
    spec = design["models"][args.model]
    policy = design["technical_tests"]
    directory = HERE / "smoke" / args.model
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(
        model=args.model,
        started_utc=now(),
        profile_fingerprint=fingerprint(spec),
        adapter_sha256=sha(CODE / "adapters.py"),
        status="running",
        heldout_evaluated=False,
    )
    dump(directory / "validation.json", result)
    try:
        products, queries, judgments = load("wands")
        dev = set(SPLITS["wands_dev"])
        ids = set(
            judgments.loc[
                judgments.query_id.isin(dev) & judgments.label.eq("Exact"), "product_id"
            ]
        )
        candidates = sorted(
            (p for p in products if str(p["product_id"]) in ids),
            key=lambda p: digest("recent-model-tech|" + str(p["product_id"])),
        )[:32]
        ordered = sorted(
            candidates, key=lambda p: (len(serializer._plain(p)), str(p["product_id"]))
        )
        chosen = [
            ordered[0],
            ordered[-1],
            *sorted(candidates, key=lambda p: digest(str(p["product_id"])))[:6],
        ]
        doc = texts_for(chosen, "C0")
        qs = sorted(
            queries[queries.query_id.isin(dev)].to_dict("records"),
            key=lambda q: digest("recent-model-tech|" + str(q["query_id"])),
        )[:3]
        dump(
            directory / "technical_inputs.json",
            dict(
                products=[str(p["product_id"]) for p in chosen],
                documents=doc,
                queries=qs,
                selection=policy["selection"],
            ),
        )
        adapter = NativeAdapter(spec)
        torch = adapter.torch
        torch.cuda.reset_peak_memory_stats()
        comparisons = []
        saved = {}
        features_out = {}
        for role, texts in [
            ("document", doc[:2]),
            ("query", [q["query"] for q in qs[:2]]),
        ]:
            for batch_size in [1, 2]:
                start = time.perf_counter()
                arr = np.concatenate(
                    [
                        adapter.encode(texts[i : i + batch_size], role)
                        for i in range(0, len(texts), batch_size)
                    ]
                )
                ref = adapter.official(texts, role, batch_size)
                cosine = np.sum(arr * ref, axis=1)
                err = float(np.max(abs(arr - ref)))
                passed = bool(
                    np.allclose(arr, ref, atol=policy["atol"], rtol=policy["rtol"])
                    and min(cosine) >= policy["min_cosine"]
                )
                comparisons.append(
                    dict(
                        role=role,
                        batch_size=batch_size,
                        max_abs_error=err,
                        min_cosine=float(min(cosine)),
                        seconds=time.perf_counter() - start,
                        passed=passed,
                    )
                )
                saved[f"{role}_adapter_b{batch_size}"] = arr
                saved[f"{role}_official_b{batch_size}"] = ref
                feat = adapter.inputs(texts, role)
                official = adapter.reference.preprocess(
                    texts, prompt=spec["prefixes"][role]
                )
                assert torch.equal(
                    feat["input_ids"], official["input_ids"]
                ) and torch.equal(feat["attention_mask"], official["attention_mask"])
                # Independent tokenizer comparison establishes prefix exactly once, preserving raw content.
                direct = adapter.tokenizer(
                    [spec["prefixes"][role] + s for s in texts],
                    padding=True,
                    truncation=True,
                    max_length=spec["max_length"],
                    return_tensors="pt",
                )
                assert torch.equal(
                    feat["input_ids"], direct["input_ids"]
                ), "processor differs from direct tokenizer: inspect template/EOS before continuing"
                features_out[f"{role}_b{batch_size}"] = {
                    k: v.tolist()
                    for k, v in feat.items()
                    if isinstance(v, torch.Tensor)
                }
                assert passed, comparisons[-1]
            single = saved[f"{role}_adapter_b1"]
            batched = saved[f"{role}_adapter_b2"]
            comparisons.append(
                dict(
                    role=role,
                    comparison="batch1 versus batch2",
                    max_abs_error=float(np.max(abs(single - batched))),
                    min_cosine=float(np.min(np.sum(single * batched, axis=1))),
                )
            )
        torch.cuda.synchronize()
        start = time.perf_counter()
        repeats = []
        scores = []
        ranks = []
        for repeat in range(policy["repeat_count"]):
            qe = np.concatenate([adapter.encode([q["query"]], "query") for q in qs])
            de = np.concatenate([adapter.encode([d], "document") for d in doc])
            score = qe @ de.T
            rk = np.stack([rank(x)[1] for x in score])
            repeats.append((qe, de))
            scores.append(score)
            ranks.append(rk)
            saved[f"repeat{repeat}_query"] = qe
            saved[f"repeat{repeat}_document"] = de
            saved[f"repeat{repeat}_score"] = score
            saved[f"repeat{repeat}_rank"] = rk
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        score_range = float(np.ptp(np.stack(scores), axis=0).max())
        result.update(
            status="completed",
            comparisons=comparisons,
            repeat_count=3,
            max_repeat_embedding_delta=float(
                max(np.max(abs(a - repeats[0][0])) for a, b in repeats)
                + max(np.max(abs(b - repeats[0][1])) for a, b in repeats)
            ),
            max_repeat_score_range=score_range,
            repeat_rank_changes=int(np.sum(np.stack(ranks) != ranks[0])),
            token_lengths=adapter.lengths(doc, "document"),
            projection=adapter.projection,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            estimates=dict(
                label="ESTIMATE from fixed development technical inputs, not measured full run",
                technical_seconds=elapsed,
                seconds_per_input=elapsed / (3 * (len(doc) + len(qs))),
                raw_input_upper_bound=7 * (42994 + 10076) + 308 + 499,
                raw_seconds_upper_estimate=elapsed
                / (3 * (len(doc) + len(qs)))
                * (7 * (42994 + 10076) + 308 + 499),
                raw_embedding_bytes_upper_estimate=(7 * (42994 + 10076) + 308 + 499)
                * spec["dimension"]
                * 4,
                full_score_bytes=7 * (308 * 42994 + 499 * 10076) * 4,
            ),
        )
        np.savez(directory / "outputs.npz", **saved)
        dump(directory / "effective_inputs.json", features_out)
        adapter.close()
        status("smoke", args.model, outputs=[source(directory / "outputs.npz")])
    except Exception as exc:
        result.update(
            status="blocked",
            exception=type(exc).__name__,
            reason=str(exc),
            traceback=traceback.format_exc(),
        )
        status("smoke", args.model, state="blocked", reason=str(exc))
        traceback.print_exc()
    result["finished_utc"] = now()
    dump(directory / "validation.json", result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0 if result["status"] == "completed" else 2
