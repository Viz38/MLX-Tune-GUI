"""
Unsloth-MLX: MLX-powered LLM fine-tuning for Apple Silicon

A drop-in replacement for Unsloth that uses Apple's MLX framework instead of CUDA/Triton kernels.

Supported Training Methods:
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- GRPO (Group Relative Policy Optimization) - DeepSeek R1 style
- KTO (Kahneman-Tversky Optimization)
- SimPO (Simple Preference Optimization)
- VLM (Vision Language Model) fine-tuning
"""

__version__ = "0.2.0"  # Bumped for RL trainers and VLM support

from unsloth_mlx.model import FastLanguageModel
from unsloth_mlx.trainer import (
    prepare_dataset,
    format_chat_template,
    create_training_data,
    save_model_hf_format,
    export_to_gguf,
    get_training_config,
)
from unsloth_mlx.sft_trainer import SFTTrainer, SFTConfig, TrainingArguments

# RL Trainers
from unsloth_mlx.rl_trainers import (
    DPOTrainer,
    DPOConfig,
    ORPOTrainer,
    ORPOConfig,
    GRPOTrainer,
    GRPOConfig,
    KTOTrainer,
    SimPOTrainer,
    prepare_preference_dataset,
    create_reward_function,
)

# Vision Language Models
from unsloth_mlx.vlm import (
    FastVisionModel,
    VLMSFTTrainer,
    load_vlm_dataset,
)

__all__ = [
    # Core
    "FastLanguageModel",
    "__version__",
    # SFT Training
    "SFTTrainer",
    "SFTConfig",
    "TrainingArguments",
    # RL Trainers
    "DPOTrainer",
    "DPOConfig",
    "ORPOTrainer",
    "ORPOConfig",
    "GRPOTrainer",
    "GRPOConfig",
    "KTOTrainer",
    "SimPOTrainer",
    # Vision Models
    "FastVisionModel",
    "VLMSFTTrainer",
    # Utilities
    "prepare_dataset",
    "prepare_preference_dataset",
    "format_chat_template",
    "create_training_data",
    "save_model_hf_format",
    "export_to_gguf",
    "get_training_config",
    "create_reward_function",
    "load_vlm_dataset",
]
