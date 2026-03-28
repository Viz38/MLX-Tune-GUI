"""
Example 28: Qwen3-Embedding Fine-Tuning with MLX-Tune

Fine-tune Qwen3-Embedding (0.6B, 4-bit quantized) for domain-specific
semantic search on Apple Silicon using contrastive learning (InfoNCE loss).

Qwen3-Embedding uses decoder architecture with bidirectional attention.
It uses last-token pooling instead of mean pooling.

Requirements:
    uv pip install mlx-tune

Usage:
    python examples/28_qwen3_embedding_finetuning.py
"""

from mlx_tune import (
    FastEmbeddingModel,
    EmbeddingSFTTrainer,
    EmbeddingSFTConfig,
    EmbeddingDataCollator,
)


def main():
    print("=" * 70)
    print("MLX-Tune: Qwen3-Embedding Fine-Tuning")
    print("=" * 70)

    # ── Step 1: Load Qwen3-Embedding ──────────────────────────────────────
    print("\n1. Loading Qwen3-Embedding-0.6B (4-bit)...")
    model, tokenizer = FastEmbeddingModel.from_pretrained(
        model_name="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        max_seq_length=512,
        pooling_strategy="last_token",  # Decoder-based models use last token
    )
    print(f"   Model: {model.model_name}")
    print(f"   Architecture: {model.architecture}")
    print(f"   Pooling: {model.pooling_strategy}")

    # ── Step 2: Add LoRA ──────────────────────────────────────────────────
    print("\n2. Adding LoRA adapters...")
    model = FastEmbeddingModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        # Auto-detected targets for Qwen3: q_proj, k_proj, v_proj, o_proj
    )
    print(f"   LoRA rank: {model.lora_config['r']}")
    print(f"   Targets: {model.lora_config['target_modules']}")

    # ── Step 3: Prepare Dataset ───────────────────────────────────────────
    print("\n3. Preparing training data...")

    # Synthetic technical Q&A pairs for domain-specific search
    train_data = [
        {"anchor": "How do I fine-tune a language model?", "positive": "Use LoRA adapters with a training framework like Unsloth or mlx-tune to efficiently fine-tune LLMs."},
        {"anchor": "What is contrastive learning?", "positive": "A training paradigm where the model learns to distinguish similar from dissimilar pairs of examples."},
        {"anchor": "How does InfoNCE loss work?", "positive": "InfoNCE uses in-batch negatives and cross-entropy over the similarity matrix to train embedding models."},
        {"anchor": "What is mean pooling in NLP?", "positive": "Averaging token embeddings weighted by attention mask to get a fixed-size sentence representation."},
        {"anchor": "How to use Apple Silicon for ML?", "positive": "Apple's MLX framework provides GPU-accelerated machine learning on M-series chips."},
        {"anchor": "What is LoRA?", "positive": "Low-Rank Adaptation adds small trainable matrices to frozen model weights for efficient fine-tuning."},
        {"anchor": "How to compute sentence similarity?", "positive": "Encode sentences into embedding vectors and compute cosine similarity between them."},
        {"anchor": "What is a sentence transformer?", "positive": "A model architecture that encodes text into dense embedding vectors for semantic search."},
        {"anchor": "How to train embeddings on Mac?", "positive": "Use mlx-tune with mlx-embeddings to fine-tune BERT or Qwen3 embedding models natively on Apple Silicon."},
        {"anchor": "What is cosine similarity?", "positive": "The dot product of two normalized vectors, measuring the angle between them as a similarity score."},
        {"anchor": "How to build a search engine?", "positive": "Index documents as embedding vectors and retrieve the most similar ones for a given query."},
        {"anchor": "What is quantization in ML?", "positive": "Reducing model precision from float32 to lower bit-widths like 4-bit or 8-bit to save memory."},
        {"anchor": "How to export a trained model?", "positive": "Save LoRA adapters with save_pretrained() or merge them into the base model for deployment."},
        {"anchor": "What is gradient accumulation?", "positive": "Summing gradients over multiple micro-batches before updating weights, simulating a larger batch size."},
        {"anchor": "How to evaluate embeddings?", "positive": "Use metrics like NDCG@10, MRR, or cosine similarity on held-out query-document pairs."},
        {"anchor": "What is the difference between BERT and GPT?", "positive": "BERT uses bidirectional attention for understanding, while GPT uses causal attention for generation."},
        {"anchor": "How does attention mechanism work?", "positive": "Attention computes weighted sums of value vectors using query-key similarity scores."},
        {"anchor": "What is transfer learning?", "positive": "Pre-training on a large corpus then fine-tuning on a specific task with a smaller dataset."},
        {"anchor": "How to reduce memory usage in training?", "positive": "Use quantized models, gradient checkpointing, LoRA adapters, and smaller batch sizes."},
        {"anchor": "What is a tokenizer?", "positive": "A component that converts text into numerical token IDs that the model can process."},
    ]
    print(f"   Samples: {len(train_data)}")

    # ── Step 4: Train ─────────────────────────────────────────────────────
    print("\n4. Starting training...")
    collator = EmbeddingDataCollator(
        model=model, tokenizer=tokenizer,
        anchor_column="anchor", positive_column="positive",
        max_seq_length=512,
    )

    config = EmbeddingSFTConfig(
        output_dir="./qwen3_embedding_output",
        per_device_train_batch_size=10,
        learning_rate=2e-5,
        max_steps=30,
        temperature=0.05,
        loss_type="infonce",
        logging_steps=5,
        max_seq_length=512,
    )

    trainer = EmbeddingSFTTrainer(
        model=model, tokenizer=tokenizer,
        data_collator=collator, train_dataset=train_data,
        args=config,
    )
    result = trainer.train()
    print(f"\n   Final avg loss: {result.metrics['train_loss']:.4f}")

    # ── Step 5: Test Embeddings ───────────────────────────────────────────
    print("\n5. Testing embeddings...")
    FastEmbeddingModel.for_inference(model)

    test_pairs = [
        ("How to train a model on Mac?", "Use mlx-tune with Apple Silicon for local fine-tuning."),
        ("How to train a model on Mac?", "The weather forecast shows rain tomorrow."),
        ("What is LoRA fine-tuning?", "Low-rank adapters enable efficient model adaptation."),
        ("What is LoRA fine-tuning?", "A recipe for chocolate chip cookies."),
    ]

    for text_a, text_b in test_pairs:
        emb = model.encode([text_a, text_b])
        similarity = (emb[0] * emb[1]).sum().item()
        print(f"   sim={similarity:.4f} | '{text_a[:45]}' vs '{text_b[:45]}'")

    # ── Step 6: Save Adapter ──────────────────────────────────────────────
    print("\n6. Saving adapter...")
    model.save_pretrained("./qwen3_embedding_output/final_adapter")

    # ── Step 7: Test Load ─────────────────────────────────────────────────
    print("\n7. Testing adapter reload...")
    # Load a fresh model
    model2, tok2 = FastEmbeddingModel.from_pretrained(
        model_name="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        max_seq_length=512,
        pooling_strategy="last_token",
    )
    model2 = FastEmbeddingModel.get_peft_model(model2, r=8, lora_alpha=16)
    model2._apply_lora()

    # Load saved adapter weights
    model2.load_adapter("./qwen3_embedding_output/final_adapter")
    FastEmbeddingModel.for_inference(model2)

    # Compare embeddings from original and reloaded model
    test_text = "How does LoRA work?"
    emb_original = model.encode([test_text])
    emb_reloaded = model2.encode([test_text])
    similarity = (emb_original[0] * emb_reloaded[0]).sum().item()
    print(f"   Original vs reloaded similarity: {similarity:.4f}")
    print(f"   (Should be ~1.0 if adapter loaded correctly)")

    print("\nDone!")


if __name__ == "__main__":
    main()
