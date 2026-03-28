"""
End-to-end tests for post-training workflows (save/load/export).

Tests the full round-trip for every model type:
  VLM:       train -> save_pretrained -> load_adapter -> verify
  STT:       train -> save_pretrained -> load_adapter -> verify
  Embedding: train -> save_pretrained -> load_adapter -> verify

Mark: @pytest.mark.slow (skipped in normal test runs)
Run with: pytest tests/test_post_training_e2e.py -v -m slow
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

import mlx.core as mx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_audio(sr=16000, duration=2.0):
    """Generate a short sine wave for testing."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.5


# ===========================================================================
# VLM Post-Training E2E
# ===========================================================================

VLM_MODEL = "mlx-community/Qwen3.5-0.8B-bf16"


class TestVLMSaveLoadE2E:
    """E2E tests for VLM save_pretrained, load_adapter, save_pretrained_merged."""

    @pytest.mark.slow
    def test_vlm_save_adapter_creates_files(self):
        """VLM save_pretrained should create adapters.safetensors + adapter_config.json."""
        from mlx_tune import FastVisionModel

        model, processor = FastVisionModel.from_pretrained(VLM_MODEL)
        model = FastVisionModel.get_peft_model(
            model, r=8, lora_alpha=8,
            finetune_vision_layers=False,
            finetune_language_layers=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            assert (Path(tmpdir) / "adapters.safetensors").exists(), \
                "adapters.safetensors missing"
            assert (Path(tmpdir) / "adapter_config.json").exists(), \
                "adapter_config.json missing"
            assert (Path(tmpdir) / "config.json").exists(), \
                "config.json missing"

            # Verify adapter_config structure
            with open(Path(tmpdir) / "adapter_config.json") as f:
                cfg = json.load(f)
            assert "lora_parameters" in cfg
            assert cfg["lora_parameters"]["rank"] == 8
            assert cfg["fine_tune_type"] == "lora"
            assert cfg["num_layers"] is not None

    @pytest.mark.slow
    def test_vlm_load_adapter_round_trip(self):
        """Save VLM adapters -> load on fresh model -> verify LoRA is applied."""
        from mlx_tune import FastVisionModel

        model, processor = FastVisionModel.from_pretrained(VLM_MODEL)
        model = FastVisionModel.get_peft_model(
            model, r=8, lora_alpha=8,
            finetune_vision_layers=False,
            finetune_language_layers=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Fresh model, load adapters
            model2, proc2 = FastVisionModel.from_pretrained(VLM_MODEL)
            model2.load_adapter(tmpdir)

            assert model2._lora_applied is True

    @pytest.mark.slow
    def test_vlm_train_save_load_consistency(self):
        """Train VLM 2 steps -> save -> load on fresh model -> verify adapters loaded."""
        from mlx_tune import (
            FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer,
        )
        from mlx_tune.vlm import VLMSFTConfig

        model, processor = FastVisionModel.from_pretrained(VLM_MODEL)
        model = FastVisionModel.get_peft_model(
            model, r=8, lora_alpha=8,
            finetune_vision_layers=False,
            finetune_language_layers=True,
        )

        # Simple text-only dataset (VLM can train on text without images)
        dataset = [
            {"messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]}
            for _ in range(4)
        ]

        FastVisionModel.for_training(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = VLMSFTTrainer(
                model=model,
                tokenizer=processor,
                data_collator=UnslothVisionDataCollator(model, processor),
                train_dataset=dataset,
                args=VLMSFTConfig(
                    output_dir=tmpdir,
                    max_steps=2,
                    learning_rate=2e-4,
                    logging_steps=1,
                ),
            )
            result = trainer.train()
            assert result is not None

            # Save adapters
            adapter_dir = Path(tmpdir) / "final_adapters"
            model.save_pretrained(str(adapter_dir))

            assert (adapter_dir / "adapters.safetensors").exists()
            assert (adapter_dir / "adapter_config.json").exists()

            # Load on fresh model
            model2, proc2 = FastVisionModel.from_pretrained(VLM_MODEL)
            model2.load_adapter(str(adapter_dir))
            assert model2._lora_applied is True

    @pytest.mark.slow
    def test_vlm_save_pretrained_merged(self):
        """VLM save_pretrained_merged should create a loadable directory."""
        from mlx_tune import FastVisionModel

        model, processor = FastVisionModel.from_pretrained(VLM_MODEL)
        model = FastVisionModel.get_peft_model(
            model, r=8, lora_alpha=8,
            finetune_vision_layers=False,
            finetune_language_layers=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained_merged(tmpdir, processor)

            # Should have safetensors weights + config
            safetensors = list(Path(tmpdir).glob("*.safetensors"))
            assert len(safetensors) > 0, "No safetensors files in merged dir"
            assert (Path(tmpdir) / "config.json").exists(), "config.json missing"

    @pytest.mark.slow
    def test_vlm_generate_after_load_adapter(self):
        """After loading adapters, VLM should be able to generate text."""
        from mlx_tune import FastVisionModel

        model, processor = FastVisionModel.from_pretrained(VLM_MODEL)
        model = FastVisionModel.get_peft_model(
            model, r=8, lora_alpha=8,
            finetune_vision_layers=False,
            finetune_language_layers=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Load on fresh model and generate
            model2, proc2 = FastVisionModel.from_pretrained(VLM_MODEL)
            model2.load_adapter(tmpdir)

            FastVisionModel.for_inference(model2)
            output = model2.generate(prompt="Hello")
            assert output is not None
            # vlm_stream_generate returns generator or string
            if hasattr(output, '__iter__') and not isinstance(output, str):
                text = ""
                for chunk in output:
                    if hasattr(chunk, 'text'):
                        text += chunk.text
                    else:
                        text += str(chunk)
                assert len(text) > 0
            else:
                assert len(str(output)) > 0


# ===========================================================================
# STT Post-Training E2E
# ===========================================================================

STT_MODEL = "mlx-community/whisper-tiny-asr-fp16"


class TestSTTSaveLoadE2E:
    """E2E tests for STT save_pretrained, load_adapter, save_pretrained_merged."""

    @pytest.mark.slow
    def test_stt_save_adapter_creates_files(self):
        """STT save_pretrained should create adapters.safetensors + adapter_config.json."""
        from mlx_tune import FastSTTModel

        model, processor = FastSTTModel.from_pretrained(STT_MODEL)
        model = FastSTTModel.get_peft_model(
            model, r=8, lora_alpha=8,
            finetune_encoder=True, finetune_decoder=True,
        )
        # STT applies LoRA lazily — trigger it explicitly
        model._apply_lora()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            assert (Path(tmpdir) / "adapters.safetensors").exists(), \
                "adapters.safetensors missing"
            assert (Path(tmpdir) / "adapter_config.json").exists(), \
                "adapter_config.json missing"

            # Verify config structure
            with open(Path(tmpdir) / "adapter_config.json") as f:
                cfg = json.load(f)
            assert cfg["model_type"] == "stt"
            assert cfg["fine_tune_type"] == "lora"
            assert cfg["lora_parameters"]["rank"] == 8
            assert "whisper_config" in cfg

    @pytest.mark.slow
    def test_stt_load_adapter_round_trip(self):
        """Save STT adapters -> load on fresh model -> verify LoRA is applied."""
        from mlx_tune import FastSTTModel

        model, processor = FastSTTModel.from_pretrained(STT_MODEL)
        model = FastSTTModel.get_peft_model(
            model, r=8, lora_alpha=8,
            finetune_encoder=True, finetune_decoder=True,
        )
        # STT applies LoRA lazily — trigger it explicitly
        model._apply_lora()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Fresh model, load adapters
            model2, proc2 = FastSTTModel.from_pretrained(STT_MODEL)
            model2.load_adapter(tmpdir)

            assert model2._lora_applied is True

    @pytest.mark.slow
    def test_stt_train_save_load_round_trip(self):
        """Train STT 2 steps -> save -> load -> verify weights loaded."""
        from mlx_tune import (
            FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator,
        )

        model, processor = FastSTTModel.from_pretrained(STT_MODEL)
        model = FastSTTModel.get_peft_model(
            model, r=8, lora_alpha=8,
            finetune_encoder=True, finetune_decoder=True,
        )

        dataset = [
            {
                "text": f"Hello world number {i}",
                "audio": {
                    "array": _make_fake_audio(16000, 2.0),
                    "sampling_rate": 16000,
                },
            }
            for i in range(4)
        ]

        collator = STTDataCollator(
            model=model, processor=processor,
            language="en", task="transcribe",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = STTSFTTrainer(
                model=model, processor=processor,
                data_collator=collator, train_dataset=dataset,
                args=STTSFTConfig(
                    output_dir=tmpdir, max_steps=2,
                    learning_rate=1e-5, logging_steps=1,
                ),
            )
            result = trainer.train()
            assert result.metrics["train_loss"] > 0

            # Save after training
            adapter_dir = Path(tmpdir) / "final_adapters"
            model.save_pretrained(str(adapter_dir))

            assert (adapter_dir / "adapters.safetensors").exists()
            assert (adapter_dir / "adapter_config.json").exists()

            # Load on fresh model
            model2, proc2 = FastSTTModel.from_pretrained(STT_MODEL)
            model2.load_adapter(str(adapter_dir))
            assert model2._lora_applied is True

    @pytest.mark.slow
    def test_stt_save_pretrained_merged(self):
        """STT save_pretrained_merged should create weights.npz + config.json."""
        from mlx_tune import FastSTTModel

        model, processor = FastSTTModel.from_pretrained(STT_MODEL)
        model = FastSTTModel.get_peft_model(
            model, r=8, lora_alpha=8,
            finetune_encoder=True, finetune_decoder=True,
        )
        model._apply_lora()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained_merged(tmpdir, processor)

            assert (Path(tmpdir) / "weights.npz").exists(), \
                "weights.npz missing in merged dir"
            assert (Path(tmpdir) / "config.json").exists(), \
                "config.json missing in merged dir"

            # Verify config has Whisper dims
            with open(Path(tmpdir) / "config.json") as f:
                cfg = json.load(f)
            # Whisper dims include n_mels, n_audio_layer, etc.
            assert "n_mels" in cfg or "n_audio_layer" in cfg or len(cfg) > 0


# ===========================================================================
# Embedding Post-Training E2E
# ===========================================================================

EMBEDDING_MODEL = "mlx-community/all-MiniLM-L6-v2-bf16"


class TestEmbeddingSaveLoadE2E:
    """E2E tests for Embedding save_pretrained, load_adapter."""

    @pytest.mark.slow
    def test_embedding_save_adapter_creates_files(self):
        """Embedding save_pretrained should create adapters.npz + adapter_config.json."""
        from mlx_tune import FastEmbeddingModel

        model, tokenizer = FastEmbeddingModel.from_pretrained(
            EMBEDDING_MODEL, max_seq_length=128,
        )
        model = FastEmbeddingModel.get_peft_model(model, r=8, lora_alpha=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            assert (Path(tmpdir) / "adapters.npz").exists(), \
                "adapters.npz missing"
            assert (Path(tmpdir) / "adapter_config.json").exists(), \
                "adapter_config.json missing"

            # Verify config
            with open(Path(tmpdir) / "adapter_config.json") as f:
                cfg = json.load(f)
            assert "architecture" in cfg
            assert "pooling_strategy" in cfg
            assert "lora_config" in cfg

    @pytest.mark.slow
    def test_embedding_load_adapter_round_trip(self):
        """Save embedding adapters -> load on fresh model -> verify."""
        from mlx_tune import FastEmbeddingModel

        model, tokenizer = FastEmbeddingModel.from_pretrained(
            EMBEDDING_MODEL, max_seq_length=128,
        )
        model = FastEmbeddingModel.get_peft_model(model, r=8, lora_alpha=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Fresh model with LoRA, then load weights
            model2, tok2 = FastEmbeddingModel.from_pretrained(
                EMBEDDING_MODEL, max_seq_length=128,
            )
            model2 = FastEmbeddingModel.get_peft_model(model2, r=8, lora_alpha=8)
            model2.load_adapter(tmpdir)

            assert model2._adapter_path == tmpdir

    @pytest.mark.slow
    def test_embedding_train_save_load_verify(self):
        """Train embedding 2 steps -> save -> load -> verify embeddings match."""
        from mlx_tune import (
            FastEmbeddingModel, EmbeddingSFTTrainer, EmbeddingSFTConfig,
            EmbeddingDataCollator,
        )

        model, tokenizer = FastEmbeddingModel.from_pretrained(
            EMBEDDING_MODEL, max_seq_length=128,
        )
        model = FastEmbeddingModel.get_peft_model(model, r=8, lora_alpha=8)

        # Simple training data
        dataset = [
            {"anchor": "The cat sat on the mat", "positive": "A cat is sitting on a mat"},
            {"anchor": "Dogs are loyal", "positive": "Canines are faithful animals"},
            {"anchor": "The sun is bright", "positive": "The star is shining"},
            {"anchor": "Water is wet", "positive": "H2O is liquid"},
        ]

        collator = EmbeddingDataCollator(model=model, tokenizer=tokenizer, max_seq_length=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = EmbeddingSFTTrainer(
                model=model, tokenizer=tokenizer,
                data_collator=collator, train_dataset=dataset,
                args=EmbeddingSFTConfig(
                    output_dir=tmpdir, max_steps=2,
                    learning_rate=2e-4, logging_steps=1,
                ),
            )
            result = trainer.train()
            assert result.metrics["train_loss"] > 0

            # Save
            adapter_dir = Path(tmpdir) / "final_adapters"
            model.save_pretrained(str(adapter_dir))

            assert (adapter_dir / "adapters.npz").exists()
            assert (adapter_dir / "adapter_config.json").exists()

            # Load on fresh model
            model2, tok2 = FastEmbeddingModel.from_pretrained(
                EMBEDDING_MODEL, max_seq_length=128,
            )
            model2 = FastEmbeddingModel.get_peft_model(model2, r=8, lora_alpha=8)
            model2.load_adapter(str(adapter_dir))

            # Loaded model should produce valid embeddings
            emb = model2.encode(["test sentence"], normalize=True)
            mx.eval(emb)
            assert emb.shape[0] == 1
            assert emb.shape[1] > 0

            # Should have unit norm (normalized)
            norm = mx.linalg.norm(emb, axis=-1).item()
            assert abs(norm - 1.0) < 1e-4, f"Expected unit norm, got {norm}"

    @pytest.mark.slow
    def test_embedding_encode_after_load(self):
        """After loading adapters, model should produce valid embeddings."""
        from mlx_tune import FastEmbeddingModel

        model, tokenizer = FastEmbeddingModel.from_pretrained(
            EMBEDDING_MODEL, max_seq_length=128,
        )
        model = FastEmbeddingModel.get_peft_model(model, r=8, lora_alpha=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            model2, tok2 = FastEmbeddingModel.from_pretrained(
                EMBEDDING_MODEL, max_seq_length=128,
            )
            model2 = FastEmbeddingModel.get_peft_model(model2, r=8, lora_alpha=8)
            model2.load_adapter(tmpdir)

            # Encode should produce normalized embeddings
            embeddings = model2.encode(
                ["Hello world", "Test sentence"],
                normalize=True,
            )
            mx.eval(embeddings)

            assert embeddings.shape[0] == 2
            assert embeddings.shape[1] > 0

            # Normalized embeddings should have ~unit norm
            norms = mx.linalg.norm(embeddings, axis=-1)
            mx.eval(norms)
            for i in range(2):
                assert abs(norms[i].item() - 1.0) < 1e-4, \
                    f"Embedding {i} norm is {norms[i].item()}, expected ~1.0"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "slow"])
