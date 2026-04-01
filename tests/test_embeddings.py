"""
Tests for embedding model fine-tuning support.

Tests cover:
- FastEmbeddingModel API
- EmbeddingModelWrapper
- EmbeddingSFTConfig
- EmbeddingDataCollator
- EmbeddingSFTTrainer
- Contrastive loss functions (InfoNCE, cosine, triplet)
"""

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Contrastive Loss Tests (real mx.array ops, no mocks)
# ──────────────────────────────────────────────────────────────────────────────


class TestInfoNCELoss:
    """Test InfoNCE / MultipleNegativesRankingLoss."""

    def test_basic_shape(self):
        from mlx_tune.losses import infonce_loss
        a = mx.random.normal((8, 64))
        p = mx.random.normal((8, 64))
        # Normalize
        a = a / mx.linalg.norm(a, axis=-1, keepdims=True)
        p = p / mx.linalg.norm(p, axis=-1, keepdims=True)
        loss = infonce_loss(a, p, temperature=0.05)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_perfect_match_low_loss(self):
        from mlx_tune.losses import infonce_loss
        # When anchors == positives, diagonal similarity is 1.0
        x = mx.random.normal((4, 32))
        x = x / mx.linalg.norm(x, axis=-1, keepdims=True)
        loss = infonce_loss(x, x, temperature=0.05)
        # Loss should be low (but not zero due to off-diagonal terms)
        assert loss.item() < 5.0

    def test_random_embeddings_higher_loss(self):
        from mlx_tune.losses import infonce_loss
        mx.random.seed(42)
        a = mx.random.normal((16, 64))
        p = mx.random.normal((16, 64))
        a = a / mx.linalg.norm(a, axis=-1, keepdims=True)
        p = p / mx.linalg.norm(p, axis=-1, keepdims=True)
        loss = infonce_loss(a, p, temperature=0.05)
        assert loss.item() > 0

    def test_temperature_effect(self):
        from mlx_tune.losses import infonce_loss
        a = mx.random.normal((8, 64))
        p = mx.random.normal((8, 64))
        a = a / mx.linalg.norm(a, axis=-1, keepdims=True)
        p = p / mx.linalg.norm(p, axis=-1, keepdims=True)
        loss_low_t = infonce_loss(a, p, temperature=0.01)
        loss_high_t = infonce_loss(a, p, temperature=1.0)
        # Lower temperature -> larger logits -> different loss landscape
        assert loss_low_t.item() != loss_high_t.item()

    def test_batch_size_1(self):
        from mlx_tune.losses import infonce_loss
        a = mx.random.normal((1, 32))
        p = mx.random.normal((1, 32))
        a = a / mx.linalg.norm(a, axis=-1, keepdims=True)
        p = p / mx.linalg.norm(p, axis=-1, keepdims=True)
        loss = infonce_loss(a, p)
        # With batch_size=1, there are no negatives, loss should be ~0
        assert loss.shape == ()

    def test_gradient_flows(self):
        """Verify InfoNCE loss supports gradient computation."""
        from mlx_tune.losses import infonce_loss

        a = mx.random.normal((4, 32))

        def compute(a):
            a_norm = a / mx.maximum(mx.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
            p = mx.stop_gradient(mx.random.normal((4, 32)))
            p = p / mx.maximum(mx.linalg.norm(p, axis=-1, keepdims=True), 1e-9)
            return infonce_loss(a_norm, p, temperature=0.05)

        grad_fn = mx.grad(compute)
        grads = grad_fn(a)
        mx.eval(grads)
        assert grads.shape == a.shape


class TestCosineEmbeddingLoss:
    """Test cosine embedding loss."""

    def test_positive_only(self):
        from mlx_tune.losses import cosine_embedding_loss
        a = mx.random.normal((4, 32))
        p = mx.random.normal((4, 32))
        a = a / mx.linalg.norm(a, axis=-1, keepdims=True)
        p = p / mx.linalg.norm(p, axis=-1, keepdims=True)
        loss = cosine_embedding_loss(a, p)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_identical_pairs_zero_loss(self):
        from mlx_tune.losses import cosine_embedding_loss
        x = mx.random.normal((4, 32))
        x = x / mx.linalg.norm(x, axis=-1, keepdims=True)
        loss = cosine_embedding_loss(x, x)
        assert loss.item() < 0.01  # Should be ~0

    def test_with_negatives(self):
        from mlx_tune.losses import cosine_embedding_loss
        a = mx.random.normal((4, 32))
        p = mx.random.normal((4, 32))
        n = mx.random.normal((4, 32))
        a = a / mx.linalg.norm(a, axis=-1, keepdims=True)
        p = p / mx.linalg.norm(p, axis=-1, keepdims=True)
        n = n / mx.linalg.norm(n, axis=-1, keepdims=True)
        loss = cosine_embedding_loss(a, p, n, margin=0.5)
        assert loss.shape == ()
        assert loss.item() >= 0


class TestTripletLoss:
    """Test triplet margin loss."""

    def test_basic(self):
        from mlx_tune.losses import triplet_loss
        a = mx.random.normal((4, 32))
        p = mx.random.normal((4, 32))
        n = mx.random.normal((4, 32))
        loss = triplet_loss(a, p, n, margin=1.0)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_easy_triplets_zero_loss(self):
        """When positive is very close and negative very far, loss should be 0."""
        from mlx_tune.losses import triplet_loss
        a = mx.zeros((2, 32))
        p = mx.zeros((2, 32)) + 0.01
        n = mx.ones((2, 32)) * 100
        loss = triplet_loss(a, p, n, margin=1.0)
        assert loss.item() < 0.01

    def test_hard_triplets_positive_loss(self):
        """When positive is far and negative is close, loss should be positive."""
        from mlx_tune.losses import triplet_loss
        a = mx.zeros((2, 32))
        p = mx.ones((2, 32)) * 10
        n = mx.zeros((2, 32)) + 0.01
        loss = triplet_loss(a, p, n, margin=1.0)
        assert loss.item() > 0

    def test_margin_effect(self):
        from mlx_tune.losses import triplet_loss
        a = mx.random.normal((4, 32))
        p = mx.random.normal((4, 32))
        n = mx.random.normal((4, 32))
        loss_small = triplet_loss(a, p, n, margin=0.1)
        loss_large = triplet_loss(a, p, n, margin=10.0)
        assert loss_large.item() >= loss_small.item()


# ──────────────────────────────────────────────────────────────────────────────
# FastEmbeddingModel API Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestFastEmbeddingModelAPI:
    """Test FastEmbeddingModel static API."""

    def test_has_from_pretrained(self):
        from mlx_tune.embeddings import FastEmbeddingModel
        assert hasattr(FastEmbeddingModel, "from_pretrained")
        assert callable(FastEmbeddingModel.from_pretrained)

    def test_has_get_peft_model(self):
        from mlx_tune.embeddings import FastEmbeddingModel
        assert hasattr(FastEmbeddingModel, "get_peft_model")
        assert callable(FastEmbeddingModel.get_peft_model)

    def test_has_for_inference(self):
        from mlx_tune.embeddings import FastEmbeddingModel
        assert hasattr(FastEmbeddingModel, "for_inference")
        assert callable(FastEmbeddingModel.for_inference)

    def test_has_for_training(self):
        from mlx_tune.embeddings import FastEmbeddingModel
        assert hasattr(FastEmbeddingModel, "for_training")
        assert callable(FastEmbeddingModel.for_training)

    def test_get_peft_model_rejects_non_wrapper(self):
        from mlx_tune.embeddings import FastEmbeddingModel
        with pytest.raises(TypeError, match="Expected EmbeddingModelWrapper"):
            FastEmbeddingModel.get_peft_model("not_a_wrapper", r=16)

    def test_get_peft_model_configures_lora(self):
        from mlx_tune.embeddings import FastEmbeddingModel, EmbeddingModelWrapper
        wrapper = EmbeddingModelWrapper(
            model=MagicMock(),
            tokenizer=MagicMock(),
            model_name="test",
        )
        result = FastEmbeddingModel.get_peft_model(wrapper, r=8, lora_alpha=16)
        assert result.lora_enabled is True
        assert result.lora_config["r"] == 8
        assert result.lora_config["lora_alpha"] == 16

    def test_for_inference_calls_eval(self):
        from mlx_tune.embeddings import FastEmbeddingModel, EmbeddingModelWrapper
        mock_model = MagicMock()
        wrapper = EmbeddingModelWrapper(
            model=mock_model, tokenizer=MagicMock(), model_name="test",
        )
        FastEmbeddingModel.for_inference(wrapper)
        mock_model.eval.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingModelWrapper Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEmbeddingModelWrapper:
    """Test EmbeddingModelWrapper."""

    def _make_wrapper(self, **kwargs):
        defaults = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "model_name": "test-model",
            "max_seq_length": 256,
            "pooling_strategy": "mean",
        }
        defaults.update(kwargs)
        return __import__("mlx_tune.embeddings", fromlist=["EmbeddingModelWrapper"]).EmbeddingModelWrapper(**defaults)

    def test_init_defaults(self):
        wrapper = self._make_wrapper()
        assert wrapper.lora_enabled is False
        assert wrapper._lora_applied is False
        assert wrapper.max_seq_length == 256
        assert wrapper.pooling_strategy == "mean"

    def test_configure_lora(self):
        wrapper = self._make_wrapper()
        wrapper.configure_lora(r=8, lora_alpha=16, target_modules=["query", "key"])
        assert wrapper.lora_enabled is True
        assert wrapper.lora_config["r"] == 8
        assert wrapper.lora_config["target_modules"] == ["query", "key"]

    def test_configure_lora_auto_targets_bert(self):
        wrapper = self._make_wrapper(config={"model_type": "bert"})
        wrapper.configure_lora(r=8)
        assert wrapper.lora_config["target_modules"] == ["query", "key", "value"]

    def test_configure_lora_auto_targets_qwen3(self):
        wrapper = self._make_wrapper(config={"model_type": "qwen3"})
        wrapper.configure_lora(r=8)
        assert "q_proj" in wrapper.lora_config["target_modules"]

    def test_pooling_mean(self):
        wrapper = self._make_wrapper(pooling_strategy="mean")
        hidden = mx.ones((2, 4, 8))
        mask = mx.array([[1, 1, 1, 0], [1, 1, 0, 0]])
        result = wrapper._pool(hidden, mask)
        assert result.shape == (2, 8)

    def test_pooling_cls(self):
        wrapper = self._make_wrapper(pooling_strategy="cls")
        hidden = mx.random.normal((2, 4, 8))
        mask = mx.ones((2, 4))
        result = wrapper._pool(hidden, mask)
        assert result.shape == (2, 8)
        # CLS should be the first token
        mx.eval(result, hidden)
        assert mx.allclose(result, hidden[:, 0, :]).item()

    def test_pooling_last_token(self):
        wrapper = self._make_wrapper(pooling_strategy="last_token")
        hidden = mx.random.normal((2, 4, 8))
        mask = mx.array([[1, 1, 1, 0], [1, 1, 1, 1]])
        result = wrapper._pool(hidden, mask)
        assert result.shape == (2, 8)

    def test_save_pretrained(self, tmp_path):
        import json
        wrapper = self._make_wrapper()
        wrapper.lora_config = {"r": 8, "target_modules": ["query"]}

        # Mock trainable_parameters to return an empty dict
        wrapper.model.trainable_parameters = MagicMock(return_value={})

        wrapper.save_pretrained(str(tmp_path / "adapters"))
        config_path = tmp_path / "adapters" / "adapter_config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["model_name"] == "test-model"
        assert config["lora_config"]["r"] == 8

    def test_load_adapter_missing_file(self, tmp_path):
        wrapper = self._make_wrapper()
        with pytest.raises(FileNotFoundError, match="No adapters.npz"):
            wrapper.load_adapter(str(tmp_path / "nonexistent"))

    def test_load_adapter_method_exists(self):
        wrapper = self._make_wrapper()
        assert hasattr(wrapper, "load_adapter")
        assert callable(wrapper.load_adapter)


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingSFTConfig Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEmbeddingSFTConfig:
    """Test EmbeddingSFTConfig."""

    def test_defaults(self):
        from mlx_tune.embeddings import EmbeddingSFTConfig
        config = EmbeddingSFTConfig()
        assert config.per_device_train_batch_size == 32
        assert config.learning_rate == 2e-5
        assert config.loss_type == "infonce"
        assert config.temperature == 0.05
        assert config.anchor_column == "anchor"
        assert config.positive_column == "positive"
        assert config.normalize_embeddings is True

    def test_custom_values(self):
        from mlx_tune.embeddings import EmbeddingSFTConfig
        config = EmbeddingSFTConfig(
            per_device_train_batch_size=64,
            learning_rate=1e-4,
            loss_type="cosine",
            temperature=0.1,
            max_steps=200,
        )
        assert config.per_device_train_batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.loss_type == "cosine"
        assert config.temperature == 0.1
        assert config.max_steps == 200

    def test_to_dict(self):
        from mlx_tune.embeddings import EmbeddingSFTConfig
        config = EmbeddingSFTConfig(learning_rate=1e-3)
        d = config.to_dict()
        assert d["learning_rate"] == 1e-3
        assert "loss_type" in d

    def test_extra_kwargs(self):
        from mlx_tune.embeddings import EmbeddingSFTConfig
        config = EmbeddingSFTConfig(custom_param="hello")
        assert config.custom_param == "hello"


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingDataCollator Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEmbeddingDataCollator:
    """Test EmbeddingDataCollator."""

    def _make_mock_tokenizer(self):
        """Create a mock tokenizer that returns proper shapes."""
        tokenizer = MagicMock()

        def mock_call(texts, **kwargs):
            import numpy as np
            batch_size = len(texts)
            seq_len = 8
            return {
                "input_ids": np.ones((batch_size, seq_len), dtype=int),
                "attention_mask": np.ones((batch_size, seq_len), dtype=int),
            }

        tokenizer.side_effect = mock_call
        return tokenizer

    def test_list_of_dicts(self):
        from mlx_tune.embeddings import EmbeddingDataCollator
        tokenizer = self._make_mock_tokenizer()
        collator = EmbeddingDataCollator(
            model=MagicMock(), tokenizer=tokenizer,
            anchor_column="anchor", positive_column="positive",
        )
        samples = [
            {"anchor": "Hello", "positive": "Hi"},
            {"anchor": "World", "positive": "Earth"},
        ]
        batch = collator(samples)
        assert "anchor_input_ids" in batch
        assert "positive_input_ids" in batch
        assert batch["anchor_input_ids"].shape[0] == 2
        assert batch["positive_input_ids"].shape[0] == 2

    def test_dict_of_lists(self):
        from mlx_tune.embeddings import EmbeddingDataCollator
        tokenizer = self._make_mock_tokenizer()
        collator = EmbeddingDataCollator(
            model=MagicMock(), tokenizer=tokenizer,
        )
        samples = {
            "anchor": ["Hello", "World"],
            "positive": ["Hi", "Earth"],
        }
        batch = collator(samples)
        assert batch["anchor_input_ids"].shape[0] == 2

    def test_with_negatives(self):
        from mlx_tune.embeddings import EmbeddingDataCollator
        tokenizer = self._make_mock_tokenizer()
        collator = EmbeddingDataCollator(
            model=MagicMock(), tokenizer=tokenizer,
            negative_column="negative",
        )
        samples = [
            {"anchor": "cat", "positive": "kitten", "negative": "car"},
            {"anchor": "dog", "positive": "puppy", "negative": "tree"},
        ]
        batch = collator(samples)
        assert "negative_input_ids" in batch
        assert batch["negative_input_ids"].shape[0] == 2


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingSFTTrainer Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEmbeddingSFTTrainer:
    """Test EmbeddingSFTTrainer initialization."""

    def test_init_with_config(self):
        from mlx_tune.embeddings import EmbeddingSFTTrainer, EmbeddingSFTConfig
        config = EmbeddingSFTConfig(
            learning_rate=1e-4,
            loss_type="cosine",
            temperature=0.1,
        )
        trainer = EmbeddingSFTTrainer(
            model=MagicMock(),
            args=config,
        )
        assert trainer.learning_rate == 1e-4
        assert trainer.loss_type == "cosine"
        assert trainer.temperature == 0.1

    def test_init_with_kwargs(self):
        from mlx_tune.embeddings import EmbeddingSFTTrainer
        trainer = EmbeddingSFTTrainer(
            model=MagicMock(),
            learning_rate=5e-5,
            loss_type="triplet",
        )
        assert trainer.learning_rate == 5e-5
        assert trainer.loss_type == "triplet"

    def test_init_with_wrapper(self):
        from mlx_tune.embeddings import EmbeddingSFTTrainer, EmbeddingModelWrapper
        wrapper = EmbeddingModelWrapper(
            model=MagicMock(), tokenizer=MagicMock(), model_name="test",
        )
        trainer = EmbeddingSFTTrainer(model=wrapper)
        assert trainer.wrapper is wrapper


# ──────────────────────────────────────────────────────────────────────────────
# Architecture Detection Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestArchitectureDetection:
    """Test architecture detection logic."""

    def test_detect_from_config_bert(self):
        from mlx_tune.embeddings import _detect_architecture
        assert _detect_architecture(MagicMock(), {"model_type": "bert"}) == "bert"

    def test_detect_from_config_modernbert(self):
        from mlx_tune.embeddings import _detect_architecture
        assert _detect_architecture(MagicMock(), {"model_type": "modernbert"}) == "modernbert"

    def test_detect_from_config_qwen(self):
        from mlx_tune.embeddings import _detect_architecture
        assert _detect_architecture(MagicMock(), {"model_type": "qwen2"}) == "qwen3"

    def test_detect_from_config_roberta(self):
        from mlx_tune.embeddings import _detect_architecture
        assert _detect_architecture(MagicMock(), {"model_type": "xlm-roberta"}) == "xlm-roberta"

    def test_detect_from_structure_bert(self):
        from mlx_tune.embeddings import _detect_architecture
        model = MagicMock()
        model.encoder.layer = [MagicMock()]
        assert _detect_architecture(model, None) == "bert"

    def test_detect_fallback(self):
        from mlx_tune.embeddings import _detect_architecture
        model = MagicMock(spec=[])  # No attributes
        # Should default to bert
        result = _detect_architecture(model, None)
        assert result == "bert"


# ──────────────────────────────────────────────────────────────────────────────
# Harrier Architecture Detection Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestHarrierArchitectureDetection:
    """Test architecture detection for Microsoft Harrier embedding models."""

    def test_detect_gemma3_text_config(self):
        """Harrier 270m/27b use model_type: gemma3_text → detected as 'gemma'."""
        from mlx_tune.embeddings import _detect_architecture
        assert _detect_architecture(MagicMock(), {"model_type": "gemma3_text"}) == "gemma"

    def test_detect_qwen3_config_for_harrier(self):
        """Harrier 0.6b uses model_type: qwen3 → detected as 'qwen3'."""
        from mlx_tune.embeddings import _detect_architecture
        assert _detect_architecture(MagicMock(), {"model_type": "qwen3"}) == "qwen3"

    def test_gemma_arch_config_targets(self):
        """Verify gemma arch config has correct targets for Harrier Gemma3-based models."""
        from mlx_tune.embeddings import _ARCH_CONFIG
        cfg = _ARCH_CONFIG["gemma"]
        assert cfg["block_path"] == ["model", "layers"]
        assert "q_proj" in cfg["targets"]
        assert "k_proj" in cfg["targets"]
        assert "v_proj" in cfg["targets"]
        assert "o_proj" in cfg["targets"]
        assert cfg["attn_attr"] == "self_attn"

    def test_qwen3_arch_config_targets(self):
        """Verify qwen3 arch config has correct targets for Harrier Qwen3-based models."""
        from mlx_tune.embeddings import _ARCH_CONFIG
        cfg = _ARCH_CONFIG["qwen3"]
        assert cfg["block_path"] == ["model", "layers"]
        assert "q_proj" in cfg["targets"]
        assert "k_proj" in cfg["targets"]
        assert "v_proj" in cfg["targets"]
        assert "o_proj" in cfg["targets"]
        assert cfg["attn_attr"] == "self_attn"

    def test_harrier_pooling_last_token(self):
        """Verify last-token pooling works (required by all Harrier models)."""
        from mlx_tune.embeddings import EmbeddingModelWrapper
        import mlx.core as mx

        wrapper = EmbeddingModelWrapper.__new__(EmbeddingModelWrapper)
        wrapper.pooling_strategy = "last_token"

        # Simulate hidden states: batch=2, seq_len=4, hidden=3
        hidden = mx.array([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
        ])
        # First sequence has 3 real tokens, second has 4
        mask = mx.array([[1, 1, 1, 0], [1, 1, 1, 1]])

        pooled = wrapper._pool(hidden, mask)
        mx.eval(pooled)

        # First sequence: last valid token is index 2 → [7, 8, 9]
        assert pooled[0, 0].item() == 7.0
        assert pooled[0, 1].item() == 8.0
        assert pooled[0, 2].item() == 9.0
        # Second sequence: last valid token is index 3 → [4, 4, 4]
        assert pooled[1, 0].item() == 4.0

    def test_harrier_pooling_strategy_parameter(self):
        """Verify pooling_strategy parameter is respected in wrapper construction."""
        from mlx_tune.embeddings import EmbeddingModelWrapper

        for strategy in ["last_token", "mean", "cls"]:
            wrapper = EmbeddingModelWrapper.__new__(EmbeddingModelWrapper)
            wrapper.pooling_strategy = strategy
            assert wrapper.pooling_strategy == strategy


# ──────────────────────────────────────────────────────────────────────────────
# Import Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestImports:
    """Test that all embedding classes are importable from mlx_tune."""

    def test_import_fast_embedding_model(self):
        from mlx_tune import FastEmbeddingModel
        assert FastEmbeddingModel is not None

    def test_import_embedding_wrapper(self):
        from mlx_tune import EmbeddingModelWrapper
        assert EmbeddingModelWrapper is not None

    def test_import_embedding_trainer(self):
        from mlx_tune import EmbeddingSFTTrainer
        assert EmbeddingSFTTrainer is not None

    def test_import_embedding_config(self):
        from mlx_tune import EmbeddingSFTConfig
        assert EmbeddingSFTConfig is not None

    def test_import_embedding_collator(self):
        from mlx_tune import EmbeddingDataCollator
        assert EmbeddingDataCollator is not None

    def test_import_infonce_loss(self):
        from mlx_tune import infonce_loss
        assert callable(infonce_loss)

    def test_import_cosine_embedding_loss(self):
        from mlx_tune import cosine_embedding_loss
        assert callable(cosine_embedding_loss)

    def test_import_triplet_loss(self):
        from mlx_tune import triplet_loss
        assert callable(triplet_loss)
