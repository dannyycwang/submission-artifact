"""Native adapters for the two additional encoders.

Official ST preprocessing/backbone are shared, but pooling is explicitly implemented
and checked against the official end-to-end encode path. Only strings are accepted.
"""

from common import *
import gc


def last_token_pool(hidden, mask):
    import torch

    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(mask == 0, -1).max(1).values
    assert (last >= 0).all()
    return hidden[torch.arange(len(hidden), device=hidden.device), last]


def mean_pool(hidden, mask):
    weights = mask.unsqueeze(-1).expand_as(hidden).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)


def native_pool(hidden, mask, spec):
    if spec["key"] == "harrier06b":
        assert spec["pooling"] == "last nonpadding token"
        return last_token_pool(hidden, mask)
    assert spec["key"] == "embeddinggemma2"
    return mean_pool(hidden, mask)


def verify_pooling_module(pooling):
    # ST 6 converts legacy pooling_mode_* config flags to one public mode value.
    assert pooling.pooling_mode == "lasttoken"
    assert pooling.embedding_dimension == 1024 and pooling.include_prompt is True


def reference_prompt(spec, role):
    name = spec["prompt_names"][role]
    if name is None:
        assert role == "document" and spec["prefixes"][role] == ""
        return {"prompt": ""}
    return {"prompt_name": name}


class NativeAdapter:
    def __init__(self, spec):
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        self.spec = spec
        self.key = spec["key"]
        assert self.key in ["embeddinggemma2", "harrier06b"]
        assert (
            torch.cuda.is_available()
        ), "CUDA GPU required by frozen execution profile"
        assert torch.cuda.is_bf16_supported(), "BF16 unsupported; new profile required"
        torch.manual_seed(spec["execution_seed"])
        torch.cuda.manual_seed_all(spec["execution_seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        self.path = Path(os.environ["ARTIFACT_MODEL_DIR"])
        kwargs = (
            {"vision_config": None, "audio_config": None}
            if self.key == "embeddinggemma2"
            else {"use_cache": False}
        )
        self.reference = SentenceTransformer(
            str(self.path),
            device="cuda",
            local_files_only=True,
            trust_remote_code=False,
            config_kwargs=kwargs,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": spec["attention_backend"],
            },
            processor_kwargs={"model_max_length": spec["max_length"]},
        )
        self.reference.eval()
        self.reference.max_seq_length = spec["max_length"]
        self.reference[0].unpad_inputs = False
        self.reference.tokenizer.padding_side = spec["padding_side"]
        self.tokenizer = self.reference.tokenizer
        assert self.reference.get_sentence_embedding_dimension() == spec["dimension"]
        for role in ["query", "document"]:
            name = spec["prompt_names"][role]
            if name is None:
                assert (
                    spec["prefixes"][role] == ""
                    and self.reference.default_prompt_name is None
                )
            else:
                assert self.reference.prompts[name] == spec["prefixes"][role]
        if self.key == "harrier06b":
            verify_pooling_module(self.reference[1])
            assert self.reference[2].__class__.__name__ == "Normalize"
            assert self.reference[0].model.config.hidden_size == 1024
            assert self.reference[0].model.config.max_position_embeddings == 32768
        if self.key == "embeddinggemma2":
            backbone = self.reference[0].model
            assert backbone.vision_tower is None and backbone.audio_tower is None
            projection = backbone.language_model.embedding_projection
            assert (projection.in_features, projection.out_features) == (512, 768)
        self.projection = (
            "native language_model.embedding_projection (512->768)"
            if self.key == "embeddinggemma2"
            else "identity"
        )

    def inputs(self, texts, role, truncation=True):
        assert role in ("query", "document") and all(isinstance(t, str) for t in texts)
        # Prefix is supplied to the input module once. Text is neither stripped nor re-serialized here.
        return self.reference[0].preprocess(
            texts,
            prompt=self.spec["prefixes"][role],
            processing_kwargs={
                "text": {
                    "truncation": truncation,
                    "max_length": self.spec["max_length"] if truncation else None,
                    "padding": True,
                }
            },
        )

    def lengths(self, texts, role):
        # The same official preprocessing path, including BOS/EOS/template tokens.
        out = []
        for text in texts:
            features = self.inputs([text], role, False)
            out.append(int(features["attention_mask"].sum()))
        return out

    def encode(self, texts, role):
        torch = self.torch
        features = self.inputs(texts, role)
        features = {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }
        with torch.inference_mode():
            out = self.reference[0](features)
            hidden = out["token_embeddings"]
            assert hidden.shape[-1] == self.spec["dimension"]
            pooled = native_pool(hidden, out["attention_mask"], self.spec)
            native = torch.nn.functional.normalize(pooled, p=2, dim=1)
            result = torch.nn.functional.normalize(native.float(), p=2, dim=1)
        array = result.cpu().numpy()
        assert np.isfinite(array).all()
        assert np.allclose(np.linalg.norm(array, axis=1), 1, atol=2e-6, rtol=0)
        return array

    def official(self, texts, role, batch_size):
        torch = self.torch
        with torch.inference_mode():
            out = self.reference.encode(
                texts,
                **reference_prompt(self.spec, role),
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False
            )
            return torch.nn.functional.normalize(out.float(), p=2, dim=1).cpu().numpy()

    def close(self):
        del self.reference
        gc.collect()
        self.torch.cuda.empty_cache()
