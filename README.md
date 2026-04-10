<p align="center">
  <strong>MLX-Tune GUI: The Visual Orchestrator for Apple Silicon</strong><br>
  <em>The professional, Mac-native interface for high-performance MLX fine-tuning.</em>
</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#model-library">Model Library</a> ·
  <a href="#contributors--credits">Credits</a>
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
