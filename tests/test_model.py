"""
Unit tests for FastLanguageModel
"""

import pytest
from unsloth_mlx import FastLanguageModel


class TestFastLanguageModel:
    """Test cases for FastLanguageModel class"""

    @pytest.fixture
    def model_name(self):
        """Fixture providing a small test model"""
        return "mlx-community/Llama-3.2-1B-Instruct-4bit"

    def test_from_pretrained_loads_model(self, model_name):
        """Test that from_pretrained successfully loads a model"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        assert model is not None
        assert tokenizer is not None
        assert hasattr(model, 'model')
        assert hasattr(model, 'tokenizer')
        assert model.max_seq_length == 512
        assert model.model_name == model_name

    def test_from_pretrained_with_custom_params(self, model_name):
        """Test from_pretrained with various parameter combinations"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            load_in_4bit=True,
            trust_remote_code=False,
        )

        assert model is not None
        assert model.max_seq_length == 1024

    def test_get_peft_model_configures_lora(self, model_name):
        """Test that get_peft_model properly configures LoRA"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
        )

        # Check LoRA configuration
        assert hasattr(model, 'lora_config')
        assert model.lora_enabled is True
        assert model.lora_config['r'] == 8
        assert model.lora_config['lora_alpha'] == 16
        assert model.lora_config['lora_dropout'] == 0.1
        assert model.lora_config['target_modules'] == ["q_proj", "v_proj"]
        assert model.lora_config['bias'] == "none"

    def test_get_peft_model_with_default_modules(self, model_name):
        """Test that get_peft_model uses default target modules when not specified"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(model, r=16)

        assert hasattr(model, 'lora_config')
        assert model.lora_enabled is True
        # Default modules should be set
        assert len(model.lora_config['target_modules']) > 0
        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        assert model.lora_config['target_modules'] == expected_modules

    def test_for_inference_enables_inference_mode(self, model_name):
        """Test that for_inference enables inference mode"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Enable inference mode
        FastLanguageModel.for_inference(model)

        assert hasattr(model, 'inference_mode')
        assert model.inference_mode is True
        assert model.use_cache is True

    def test_for_inference_with_no_cache(self, model_name):
        """Test for_inference with caching disabled"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(model, use_cache=False)

        assert model.inference_mode is True
        assert model.use_cache is False

    def test_tokenizer_functionality(self, model_name):
        """Test that tokenizer works correctly"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Test encoding
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

        # Test decoding
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        assert "Hello" in decoded or "hello" in decoded

    def test_model_wrapper_attributes(self, model_name):
        """Test that model wrapper has expected attributes"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Check wrapper attributes
        assert hasattr(model, 'model')
        assert hasattr(model, 'tokenizer')
        assert hasattr(model, 'max_seq_length')
        assert hasattr(model, 'model_name')
        assert hasattr(model, 'lora_config')
        assert hasattr(model, 'lora_enabled')
        assert hasattr(model, 'inference_mode')
        assert hasattr(model, 'use_cache')

        # Check initial values
        assert model.lora_config is None
        assert model.lora_enabled is False
        assert model.inference_mode is False
        assert model.use_cache is True

    def test_full_workflow(self, model_name):
        """Test complete workflow: load, configure LoRA, enable inference"""
        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Configure LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
        )

        # Enable inference
        FastLanguageModel.for_inference(model)

        # Verify all configurations
        assert model.max_seq_length == 512
        assert model.lora_enabled is True
        assert model.lora_config['r'] == 8
        assert model.inference_mode is True


class TestMLXModelWrapper:
    """Test cases for MLXModelWrapper class"""

    @pytest.fixture
    def wrapped_model(self):
        """Fixture providing a wrapped model"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
            max_seq_length=512,
            load_in_4bit=True,
        )
        return model, tokenizer

    def test_configure_lora_method(self, wrapped_model):
        """Test configure_lora method"""
        model, _ = wrapped_model

        model.configure_lora(
            r=16,
            target_modules=["q_proj"],
            lora_alpha=32,
        )

        assert model.lora_config is not None
        assert model.lora_config['r'] == 16
        assert model.lora_config['lora_alpha'] == 32

    def test_enable_inference_mode_method(self, wrapped_model):
        """Test enable_inference_mode method"""
        model, _ = wrapped_model

        model.enable_inference_mode(use_cache=True)

        assert model.inference_mode is True
        assert model.use_cache is True
