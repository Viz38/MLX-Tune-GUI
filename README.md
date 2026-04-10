<p align="center">
  <img src="./assets/logo.png" alt="MLX-Tune Logo" width="300"/>
</p>

<p align="center">
  <strong>The Professional ML Fine-Tuning Suite for Apple Silicon</strong><br>
  <em>High-performance SFT, DPO, GRPO, Vision, Audio, and OCR training — natively on MLX.</em>
</p>

<p align="center">
  <br>
  <a href="#installation"><img src="https://img.shields.io/badge/Platform-Apple%20Silicon-black?logo=apple" alt="Platform"></a>
  <a href="#requirements"><img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/ml-explore/mlx"><img src="https://img.shields.io/badge/MLX-0.20+-green" alt="MLX"></a>
</p>

---

## 🖥️ The MLX-Tune GUI Orchestrator
MLX-Tune now includes a **Pro GUI Orchestrator**—a Mac-native, "Deep Dark" interface designed for researchers and developers who want to manage training runs visually.

- **Unified Dashboard**: Orchestrate LLM, Vision, Audio, and OCR training in one window.
- **Model Library**: Integrated scanning for Ollama, LM Studio, and Hugging Face caches.
- **Live Diagnostics**: Real-time terminal streaming with memory and step tracking.
- **System Settings**: Global environment management and persistence.

---

## 🔥 Why MLX-Tune?
Bringing the [Unsloth](https://github.com/unslothai/unsloth) experience to Mac users via Apple's [MLX](https://github.com/ml-explore/mlx) framework.

- 🚀 **Hardware Native**: Specifically optimized for M-series (M1/M2/M3/M4/M5) chips.
- 💾 **Unified Memory**: Fine-tune massive models (up to 70B+) using up to 512GB of shared RAM.
- 🔄 **Code Portability**: 100% Unsloth-compatible API. Write once on Mac, scale to CUDA later.
- 🎙️ **Multi-Modal**: Support for 7+ training specialized modes including TTS, STT, and OCR.

---

## 🛠️ Installation

### 1. Install the Library
```bash
# Core library
pip install mlx-tune

# Full suite (with Audio support)
pip install 'mlx-tune[audio]'
brew install ffmpeg
```

### 2. Launch the GUI
The GUI is built with Next.js and interfaces directly with your local Python environment.
```bash
cd web
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to start your first run.

---

## 📋 v0.4.20 Feature Support

| Training Type | Features | Status |
| :--- | :--- | :--- |
| **LLM SFT** | LoRA/QLoRA, Response-only training, Chat templates | ✅ Stable |
| **Vision SFT** | Gemma 4, Qwen3.5, PaliGemma, LLaVA | ✅ Stable |
| **RLHF (DPO/ORPO)** | Explicit DPO loss, Odds-ratio optimization | ✅ Stable |
| **Reasoning (GRPO)** | Multi-generation with rewards (DeepSeek R1 style) | ✅ Stable |
| **Audio (TTS/STT)** | Orpheus, Whisper, Moonshine, Qwen3-Audio | ✅ Stable |
| **OCR** | DeepSeek-OCR, olmOCR, Handwriting recognition | ✅ Stable |
| **Embedding** | BERT, ModernBERT, Harrier (InfoNCE loss) | ✅ Stable |

---

## 🚀 Quick Start (CLI)
```python
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)

trainer = SFTTrainer(
    model=model,
    train_dataset=load_dataset("yahma/alpaca-cleaned", split="train[:100]"),
    args=SFTConfig(output_dir="outputs", per_device_train_batch_size=2, max_steps=50),
)
trainer.train()
```

---

## 👥 Contributors & Credits
This is a **separate, specialized repository** focused on high-level orchestration and the Pro GUI suite. We acknowledge the incredible foundation built by the core contributors of the `mlx-tune` ecosystem:

- **[ARahim3](https://github.com/ARahim3)**: Project lead and core `mlx-tune` architecture.
- **[Prince Canuma](https://github.com/Blaizzy)**: Creator of MLX-VLM, MLX-Audio, and key multi-modal infrastructure.
- **The MLX Team**: For Apple's foundational machine learning framework.

---

## 🤝 Acknowledgments
Special thanks to the [Unsloth](https://github.com/unslothai/unsloth) team for the API inspiration.

<p align="center">
  <strong>Community project, not affiliated with Unsloth AI or Apple.</strong><br>
  ⭐ Star this repo if you find it useful!
</p>
