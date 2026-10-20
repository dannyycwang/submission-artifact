"""Native frozen-model encoder retained from the research implementation."""

from __future__ import annotations
import gc, hashlib, json, time
from pathlib import Path
import numpy as np


class DenseEncoder:

    def __init__(self, model_spec: dict, config: dict, cache_dir: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.spec = model_spec
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(config["seed"])
        if self.device == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_spec["name"], revision=model_spec["revision"]
        )
        kwargs = {}
        self.attention_implementation = "model_default"
        if model_spec["key"] == "gte_modernbert":
            self.attention_implementation = "sdpa"
            kwargs["attn_implementation"] = self.attention_implementation
        self.model = (
            AutoModel.from_pretrained(
                model_spec["name"], revision=model_spec["revision"], **kwargs
            )
            .to(self.device)
            .eval()
        )
        if self.device == "cuda":
            self.model.half()
        self.hidden_size = int(self.model.config.hidden_size)
        self.base_batch_size = int(
            config["encoding"][f"batch_size_{model_spec['key']}"]
        )
        print(
            f"loaded {model_spec['name']}@{model_spec['revision']} on {self.device}; hidden={self.hidden_size}",
            flush=True,
        )

    def _pool(self, hidden, attention_mask, pooling: str):
        if pooling == "model_native":
            pooling = "mean" if "mean" in self.spec["native_pooling"].lower() else "cls"
        if pooling == "cls":
            return hidden[:, 0].float()
        mask = attention_mask.unsqueeze(-1)
        return (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)

    def _forward_native(
        self, texts: list[str], pooling: str, batch_size: int
    ) -> np.ndarray:
        torch = self.torch
        try:
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.spec["max_tokens"],
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for (key, value) in batch.items()}
            with torch.inference_mode():
                output = self.model(**batch).last_hidden_state
                pooled = self._pool(output, batch["attention_mask"], pooling)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            torch.cuda.empty_cache()
            midpoint = len(texts) // 2
            left = self._forward_native(
                texts[:midpoint], pooling, max(1, batch_size // 2)
            )
            right = self._forward_native(
                texts[midpoint:], pooling, max(1, batch_size // 2)
            )
            return np.concatenate([left, right])

    def _encode_native(self, texts: list[str], pooling: str) -> tuple[np.ndarray, dict]:
        lengths = []
        started = time.time()
        for offset in range(0, len(texts), 256):
            block = texts[offset : offset + 256]
            lengths.extend(
                (
                    len(ids)
                    for ids in self.tokenizer(
                        block, add_special_tokens=True, truncation=False, verbose=False
                    )["input_ids"]
                )
            )
        processing_order = np.argsort(
            np.minimum(lengths, self.spec["max_tokens"]), kind="stable"
        )
        array = np.empty((len(texts), self.hidden_size), dtype=np.float32)
        completed = 0
        capped_lengths = np.minimum(lengths, self.spec["max_tokens"])
        offset = 0
        while offset < len(texts):
            current_length = int(capped_lengths[processing_order[offset]])
            batch_size = self.base_batch_size
            if self.spec["key"] == "gte_modernbert":
                if current_length > 1500:
                    batch_size = 1
                elif current_length > 1024:
                    batch_size = min(batch_size, 2)
                elif current_length > 768:
                    batch_size = min(batch_size, 4)
                elif current_length > 500:
                    batch_size = min(batch_size, 8)
            indices = processing_order[offset : offset + batch_size]
            block = [texts[int(index)] for index in indices]
            array[indices] = self._forward_native(block, pooling, len(block))
            completed += len(block)
            offset += len(block)
            if completed == len(block) or completed % max(
                1024, self.base_batch_size * 32
            ) < len(block):
                print(
                    f"native {completed}/{len(texts)} {time.time() - started:.0f}s",
                    flush=True,
                )
        return (
            array,
            {
                "mean_wordpieces_with_special": float(np.mean(lengths)),
                "p95_wordpieces_with_special": float(np.quantile(lengths, 0.95)),
                "fraction_truncated": float(
                    np.mean(np.asarray(lengths) > self.spec["max_tokens"])
                ),
            },
        )

    def encode(
        self, texts: list[str], cache_name: str, profile: dict, source_sha256: str
    ) -> np.ndarray:
        payload = {
            "model": self.spec,
            "profile": profile,
            "source_sha256": source_sha256,
            "seed": self.config["seed"],
            "implementation": "phase2-encoder-v3-length-bucket-sdpa",
            "attention_implementation": self.attention_implementation,
            "dynamic_batching": "GTE: base through 500 tokens, then 8/4/2/1 above 500/768/1024/1500; stable length buckets restored to source order",
            "precision": "fp16_model_fp32_pooling" if self.device == "cuda" else "fp32",
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:20]
        path = self.cache_dir / f"{cache_name}_{fingerprint}.npy"
        meta_path = path.with_suffix(".json")
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["fingerprint"] == fingerprint:
                print(f"cache hit {path.name}", flush=True)
                return np.load(path, mmap_mode="r")
        started = time.time()
        (result, length_meta) = self._encode_native(texts, profile["pooling"])
        if not np.isfinite(result).all():
            raise AssertionError("non-finite embedding")
        norms = np.linalg.norm(result, axis=1)
        if not np.allclose(norms, 1, atol=0.002):
            raise AssertionError((float(norms.min()), float(norms.max())))
        np.save(path, result)
        meta_path.write_text(
            json.dumps(
                {
                    **payload,
                    **length_meta,
                    "fingerprint": fingerprint,
                    "shape": list(result.shape),
                    "seconds": time.time() - started,
                    "tokenizer_class": type(self.tokenizer).__name__,
                    "model_class": type(self.model).__name__,
                    "device": self.device,
                },
                indent=2,
            )
        )
        return np.load(path, mmap_mode="r")

    def close(self):
        del self.model
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
