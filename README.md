# Unsloth-MLX

> **Unsloth for Mac** - MLX-powered LLM fine-tuning for Apple Silicon

Bringing the [Unsloth](https://github.com/unslothai/unsloth) experience to Mac users via Apple's [MLX](https://github.com/ml-explore/mlx) framework.

## What This Is (and Isn't)

**This is NOT** a replacement for Unsloth or an attempt to compete with it. Unsloth is incredible - it's the gold standard for efficient LLM fine-tuning on CUDA.

**This IS** a bridge for Mac users who want to:
- 🧪 **Prototype locally** - Experiment with fine-tuning on your Mac before committing to cloud GPU costs
- 📚 **Learn & iterate** - Develop your training pipeline with fast local feedback loops
- 🔄 **Then scale up** - Move to cloud NVIDIA GPUs + original Unsloth for production training

**The workflow:**
```
Local Mac (Unsloth-MLX)     →     Cloud GPU (Unsloth)
   Prototype & experiment          Full-scale training
   Small datasets                  Large datasets
   Quick iterations                Production runs
```

Same code, just change the import. Start local, scale to cloud.

## Why Unsloth-MLX?

- 🚀 **Fine-tune LLMs locally** on your Mac (M1/M2/M3/M4/M5)
- 💾 **Leverage unified memory** (up to 192GB on Mac Studio)
- 🔄 **Same API as Unsloth** - just change the import line!
- 📦 **Save anywhere** - HuggingFace format, GGUF for Ollama/llama.cpp

## Project Status

> 🚧 **Building in Public** - This project is actively developed. Core features work, advanced features in progress.

| Feature | Status | Notes |
|---------|--------|-------|
| SFT Training | ✅ Stable | Full LoRA fine-tuning |
| Model Loading | ✅ Stable | Any HuggingFace model |
| Save/Export | ✅ Stable | HF format, GGUF |
| DPO/ORPO/GRPO | ⚠️ Beta | API ready, full loss coming |
| Vision Models | ⚠️ Beta | Via mlx-vlm |
| PyPI Package | 🔜 Soon | Install from source for now |
| M5 Optimization | 🔜 Planned | Neural Accelerator support |

## Installation

```bash
# From source (recommended for now)
git clone https://github.com/ARahim3/unsloth-mlx.git
cd unsloth-mlx
pip install -e .

# PyPI coming soon!
# pip install unsloth-mlx
```

## Quick Start

```python
# Just change this import - rest of your Unsloth code works!
from unsloth_mlx import FastLanguageModel, SFTTrainer, SFTConfig

# Load any HuggingFace model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)

# Train with SFTTrainer (same API as TRL!)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        max_steps=100,
    ),
)
trainer.train()

# Save (same API as Unsloth!)
model.save_pretrained("lora_model")  # Adapters only
model.save_pretrained_merged("merged", tokenizer)  # Full model
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")  # GGUF
```

## Supported Training Methods

| Method | Trainer | Status | Use Case |
|--------|---------|--------|----------|
| **SFT** | `SFTTrainer` | ✅ Stable | Instruction fine-tuning |
| **DPO** | `DPOTrainer` | ⚠️ Beta | Preference learning |
| **ORPO** | `ORPOTrainer` | ⚠️ Beta | Combined SFT + preference |
| **GRPO** | `GRPOTrainer` | ⚠️ Beta | Reasoning (DeepSeek R1 style) |
| **KTO** | `KTOTrainer` | ⚠️ Beta | Kahneman-Tversky optimization |
| **SimPO** | `SimPOTrainer` | ⚠️ Beta | Simple preference optimization |
| **VLM** | `VLMSFTTrainer` | ⚠️ Beta | Vision-Language models |

## API Compatibility

The goal is **zero code changes** when switching between Mac and CUDA:

```python
# Unsloth (CUDA)
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Unsloth-MLX (Apple Silicon) - just change the import!
from unsloth_mlx import FastLanguageModel, SFTTrainer, SFTConfig
```

## Examples

Check [`examples/`](examples/) for working code:
- Basic model loading and inference
- Complete SFT fine-tuning pipeline
- RL training methods (DPO, GRPO, ORPO)

## Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4/M5)
- **OS**: macOS 13.0+ (15.0+ for large models)
- **Memory**: 16GB+ unified RAM (32GB+ for 7B+ models)
- **Python**: 3.9+

## What's Next

- [ ] PyPI package release
- [ ] Full DPO/GRPO loss implementations
- [ ] M5 Neural Accelerator optimization
- [ ] Custom MLX kernels
- [ ] More documentation & tutorials

## Contributing

Contributions welcome! Areas that need help:
- Full RL loss implementations (DPO, GRPO)
- Custom MLX kernels for performance
- Documentation and examples
- Testing on different M-series chips

## Comparison with Unsloth

| Feature | Unsloth (CUDA) | Unsloth-MLX |
|---------|----------------|-------------|
| Platform | NVIDIA GPUs | Apple Silicon |
| Backend | Triton Kernels | MLX Framework |
| Memory | VRAM (limited) | Unified (up to 192GB) |
| API | Original | 100% Compatible |
| Best For | Production training | Local dev, large models |

## License

Apache 2.0

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - The original, incredible CUDA library
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [MLX-LM](https://github.com/ml-explore/mlx-lm) - LLM utilities for MLX
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) - Vision model support

---

**Note**: Community project, not affiliated with Unsloth AI or Apple.

⭐ Star this repo if you find it useful!
