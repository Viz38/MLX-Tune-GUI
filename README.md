<p align="center">
  <img src="https://raw.githubusercontent.com/ARahim3/mlx-tune/main/mlx-tune-logo.png" alt="MLX-Tune Logo" width="300"/>
</p>

<p align="center">
  <strong>Fine-tune LLMs, Vision, Audio, and OCR models on your Mac</strong><br>
  <em>SFT, DPO, GRPO, Vision, TTS, STT, Embedding, and OCR fine-tuning — natively on MLX. Unsloth-compatible API.</em>
</p>

<p align="center">
  <br>
  <a href="https://github.com/ARahim3/mlx-tune#installation"><img src="https://img.shields.io/badge/Platform-Apple%20Silicon-black?logo=apple" alt="Platform"></a>
  <a href="https://github.com/ARahim3/mlx-tune#requirements"><img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/ml-explore/mlx"><img src="https://img.shields.io/badge/MLX-0.20+-green" alt="MLX"></a>
  <a href="https://github.com/ARahim3/mlx-tune#license"><img src="https://img.shields.io/badge/License-Apache%202.0-orange" alt="License"></a>
</p>

<p align="center">
  <a href="https://arahim3.github.io/mlx-tune/">Documentation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="https://github.com/ARahim3/mlx-tune#supported-training-methods">Training Methods</a>
</p>

---

## 🖥️ Overview
**MLX-Tune GUI** is a pro-grade graphical orchestrator designed for researchers and developers working locally on M-series Macs. It brings a "Deep Dark" Mac-native experience to the complex world of MLX fine-tuning, allowing you to manage, monitor, and execute training runs without touching the command line.

### 🚀 Key Features
- **Unified Training Dashboard**: Orchestrate LLM (SFT/RLHF), Vision, Audio (TTS/STT), and OCR training from a single window.
- **Integrated Model Library**: Automatically scan and manage weights from Ollama, LM Studio, and Hugging Face local caches.
- **Real-time Diagnostics**: Follow every step with a built-in, resizable terminal and live log streaming.
- **Persistent Settings**: Save and sync your environment paths and output directories across sessions.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have the core library installed:
```bash
pip install mlx-tune
```

### 2. Launch the Orchestrator
The GUI is built with Next.js and interfaces directly with your local Python environment.
```bash
# Clone the repository
git clone https://github.com/Viz38/MLX-Tune-GUI.git
cd mlx-tune-gui/web

# Install and Run
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser to begin training.

---

## 📋 v0.4.20 Orchestration Matrix

| Mode | Capabilities |
| :--- | :--- |
| **LLM Orchestrator** | SFT, DPO, GRPO, ORPO, SimPO support |
| **Vision Orchestrator** | Gemma 4, Qwen3.5, PaliGemma fine-tuning |
| **Audio Suite** | TTS (Orpheus, OuteTTS) & STT (Whisper, Moonshine) |
| **Document OCR** | DeepSeek-OCR and olmOCR orchestration |
| **Embedding** | Semantic search training via Harrier/BERT |

---

## 👥 Contributors & Credits
This is a **separate, specialized repository** dedicated to the MLX-Tune Graphical Suite. We acknowledge the foundational work by:

- **[ARahim3](https://github.com/ARahim3)**: Project lead and core `mlx-tune` architecture.
- **[Prince Canuma](https://github.com/Blaizzy)**: Creator of the multi-modal MLX infrastructure.
- **The MLX Team**: For Apple's foundational machine learning framework.

---

<p align="center">
  <strong>Community project, not affiliated with Unsloth AI or Apple.</strong><br>
  ⭐ Star this repo if you find it useful!
</p>
