"""Resumable content-deduplicated full-catalog encoding, with per-block hashes."""

from common import *
from adapters import NativeAdapter
import time


def encode_collection(adapter, texts, role, directory):
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = dict(
        spec=adapter.spec,
        role=role,
        adapter_sha256=sha(CODE / "adapters.py"),
        software=protocol()["software"],
        text_hashes=[digest(t) for t in texts],
    )
    completion = directory / "complete.json"
    if valid(completion, descriptor):
        return np.load(directory / "embeddings.npy", mmap_mode="r"), np.load(
            directory / "lengths.npy"
        )
    if (directory / "descriptor.json").exists():
        assert (
            read(directory / "descriptor.json") == descriptor
        ), "Profile/input changed: use new directory; never mix cache profiles"
    else:
        dump(directory / "descriptor.json", descriptor)
    unique = {}
    aliases = []
    unique_texts = []
    for text in texts:
        key = digest(text)
        if key not in unique:
            unique[key] = len(unique_texts)
            unique_texts.append(text)
        else:
            assert unique_texts[unique[key]] == text, "content hash collision"
        aliases.append(unique[key])
    aliases = np.array(aliases, dtype=np.int32)
    if (directory / "unique_lengths.npy").exists():
        lengths = np.load(directory / "unique_lengths.npy")
    else:
        lengths = []
        for start in range(0, len(unique_texts), 128):
            feats = adapter.inputs(unique_texts[start : start + 128], role, False)
            lengths.extend(feats["attention_mask"].sum(1).tolist())
            if start % 4096 == 0:
                print("TOKENS", directory.name, start, len(unique_texts), flush=True)
        lengths = np.asarray(lengths, dtype=np.int32)
        np.save(directory / "unique_lengths.npy", lengths)
    assert len(lengths) == len(unique_texts)
    order = sorted(
        range(len(unique_texts)),
        key=lambda i: (
            min(int(lengths[i]), adapter.spec["max_length"]),
            digest(unique_texts[i]),
        ),
    )
    plan = []
    offset = 0
    while offset < len(order):
        batch = adapter.spec["batching"]["base"]
        remaining = order[offset : offset + batch]
        # Select size from the longest potential input, so no batch exceeds its length tier.
        longest = max(
            min(int(lengths[i]), adapter.spec["max_length"]) for i in remaining
        )
        batch = min(
            batch,
            next(
                size
                for limit, size in adapter.spec["batching"]["limits"]
                if longest <= limit
            ),
        )
        plan.append(order[offset : offset + batch])
        offset += batch
    dump(
        directory / "plan.json",
        dict(
            batches=plan,
            aliases=aliases.tolist(),
            unique_text_hashes=[digest(x) for x in unique_texts],
            effective_length=adapter.spec["max_length"],
        ),
    )
    with gzip.open(directory / "unique_inputs.jsonl.gz", "wt", encoding="utf8") as f:
        for i, t in enumerate(unique_texts):
            f.write(
                json.dumps(
                    dict(
                        index=i,
                        text=t,
                        input_hash=digest(adapter.spec["prefixes"][role] + t),
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
    # Group deterministic batches into checkpoints without changing their batch companions.
    chunk_size = 128
    chunk_outputs = []
    started = time.time()
    for block in range(0, len(plan), chunk_size):
        batches = plan[block : block + chunk_size]
        indices = [i for batch in batches for i in batch]
        chunk = directory / f"block_{block:07d}.npz"
        flag = chunk.with_suffix(".json")
        inputs = dict(
            profile=fingerprint(descriptor),
            batches=batches,
            text_hashes=[digest(unique_texts[i]) for i in indices],
        )
        if not valid(flag, inputs):
            values = []
            begin = time.time()
            for batch in batches:
                try:
                    values.append(
                        adapter.encode([unique_texts[i] for i in batch], role)
                    )
                except adapter.torch.cuda.OutOfMemoryError as exc:
                    dump(
                        directory / f"oom_{block:07d}.json",
                        dict(
                            time=now(),
                            batch=batch,
                            lengths=lengths[batch].tolist(),
                            profile=adapter.spec,
                            error=str(exc),
                        ),
                    )
                    raise
            tmp = chunk.with_suffix(".tmp.npz")
            np.savez(
                tmp,
                indices=np.array(indices, dtype=np.int32),
                embeddings=np.concatenate(values),
            )
            tmp.replace(chunk)
            complete(flag, inputs, [chunk], seconds=time.time() - begin)
        chunk_outputs.append(chunk)
        print(
            "ENCODE",
            directory.name,
            min(block + chunk_size, len(plan)),
            len(plan),
            "batches",
            round(time.time() - started, 1),
            "sec",
            flush=True,
        )
    matrix = np.lib.format.open_memmap(
        directory / "unique_embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(unique_texts), adapter.spec["dimension"]),
    )
    for chunk in chunk_outputs:
        with np.load(chunk) as data:
            matrix[data["indices"]] = data["embeddings"]
    matrix.flush()
    array = np.asarray(matrix[aliases])
    np.save(directory / "embeddings.npy", array)
    np.save(directory / "lengths.npy", lengths[aliases])
    complete(
        completion,
        descriptor,
        [
            directory / "embeddings.npy",
            directory / "lengths.npy",
            directory / "plan.json",
            directory / "unique_inputs.jsonl.gz",
        ],
        total_inputs=len(texts),
        unique_inputs=len(unique_texts),
        seconds=time.time() - started,
        cache_key_contents="model/tokenizer SHA, complete raw inputs plus role prefixes, projection, pooling, dtype, backend, length, software, adapter hash, batching",
    )
    return np.load(directory / "embeddings.npy", mmap_mode="r"), lengths[aliases]


def run(model, dataset, canonical=False):
    design = protocol()
    assert model in design.get(
        "active_models", list(design["models"])
    ), "Model is not in this run profile"
    spec = design["models"][model]
    validation = read(HERE / f"smoke/{model}/validation.json")
    assert (
        validation["status"] == "completed"
    ), "Official reference validation has not passed"
    assert validation["profile_fingerprint"] == fingerprint(spec) and validation[
        "adapter_sha256"
    ] == sha(CODE / "adapters.py")
    products, queries, _ = support(dataset)
    directory = HERE / "runs" / model / dataset
    conditions = RULES if canonical else STEMS
    all_texts = [
        text for condition in conditions for text in texts_for(products, condition)
    ]
    adapter = NativeAdapter(spec)
    try:
        # Query input collection has one checkpoint shared by all raw and canonical conditions.
        encode_collection(
            adapter, queries["query"].tolist(), "query", directory / "queries"
        )
        _, lengths = encode_collection(
            adapter,
            all_texts,
            "document",
            directory / ("canonical" if canonical else "raw"),
        )
        audit = []
        for s, condition in enumerate(conditions):
            for i, p in enumerate(products):
                n = int(lengths[s * len(products) + i])
                audit.append(
                    dict(
                        product_id=str(p["product_id"]),
                        schedule=condition,
                        tokens_untruncated=n,
                        native_length=spec["native_supported_length"],
                        effective_max_length=spec["max_length"],
                        truncated=n > spec["max_length"],
                    )
                )
        pd.DataFrame(audit).to_parquet(
            directory
            / ("canonical_token_audit.parquet" if canonical else "token_audit.parquet"),
            index=False,
        )
        status(
            "canonical_encode" if canonical else "encode",
            model,
            dataset,
            profile_fingerprint=fingerprint(spec),
            query_embeddings_once=True,
        )
    finally:
        adapter.close()
