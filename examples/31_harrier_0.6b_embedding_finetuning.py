"""
Example 31: Microsoft Harrier 0.6B Embedding Fine-Tuning with MLX-Tune

Fine-tune Microsoft Harrier-OSS-v1-0.6B for cross-lingual semantic search
on Apple Silicon using contrastive learning (InfoNCE loss).

Harrier is a multilingual embedding model supporting 94 languages.
The 0.6B variant uses a Qwen3 decoder backbone with last-token pooling.

Requirements:
    uv pip install mlx-tune

Usage:
    python examples/31_harrier_0.6b_embedding_finetuning.py
"""

from mlx_tune import (
    FastEmbeddingModel,
    EmbeddingSFTTrainer,
    EmbeddingSFTConfig,
    EmbeddingDataCollator,
)


def main():
    print("=" * 70)
    print("MLX-Tune: Microsoft Harrier 0.6B Embedding Fine-Tuning")
    print("=" * 70)

    # ── Step 1: Load Harrier 0.6B ────────────────────────────────────────
    # Harrier uses a Qwen3 decoder backbone (model_type: "qwen3")
    # with last-token pooling — architecture auto-detected
    print("\n1. Loading Microsoft Harrier-OSS-v1-0.6B...")
    model, tokenizer = FastEmbeddingModel.from_pretrained(
        model_name="microsoft/harrier-oss-v1-0.6b",
        max_seq_length=512,
        pooling_strategy="last_token",  # Harrier uses last-token pooling
    )
    print(f"   Model: {model.model_name}")
    print(f"   Architecture: {model.architecture}")
    print(f"   Pooling: {model.pooling_strategy}")

    # ── Step 2: Add LoRA ─────────────────────────────────────────────────
    print("\n2. Adding LoRA adapters...")
    model = FastEmbeddingModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        # Auto-detected targets: q_proj, k_proj, v_proj, o_proj
    )
    print(f"   LoRA rank: {model.lora_config['r']}")
    print(f"   Targets: {model.lora_config['target_modules']}")

    # ── Step 3: Prepare Dataset ──────────────────────────────────────────
    print("\n3. Preparing cross-lingual training data...")

    # Cross-lingual retrieval pairs — Harrier supports 94 languages
    # Anchor: English query, Positive: answer in another language
    train_data = [
        # English ↔ Spanish
        {"anchor": "What is machine learning?", "positive": "El aprendizaje automatico es una rama de la inteligencia artificial que permite a los sistemas aprender de datos."},
        {"anchor": "How to sort a list in Python?", "positive": "Utiliza sorted() o list.sort() para ordenar listas en Python."},
        {"anchor": "What causes climate change?", "positive": "El cambio climatico es causado principalmente por las emisiones de gases de efecto invernadero."},
        # English ↔ French
        {"anchor": "How does photosynthesis work?", "positive": "La photosynthese convertit la lumiere du soleil en energie chimique dans les plantes."},
        {"anchor": "What is quantum computing?", "positive": "L'informatique quantique utilise des qubits pour effectuer des calculs complexes."},
        {"anchor": "How to train a neural network?", "positive": "Entrainer un reseau de neurones necessite des donnees, une fonction de perte et un optimiseur."},
        # English ↔ German
        {"anchor": "What is the theory of relativity?", "positive": "Die Relativitaetstheorie beschreibt die Beziehung zwischen Raum, Zeit und Gravitation."},
        {"anchor": "How does encryption work?", "positive": "Verschluesselung wandelt Klartext in unlesbaren Chiffretext um, der nur mit einem Schluessel entschluesselt werden kann."},
        {"anchor": "What is deep learning?", "positive": "Deep Learning ist ein Teilgebiet des maschinellen Lernens mit mehrschichtigen neuronalen Netzen."},
        # English ↔ Japanese
        {"anchor": "What is natural language processing?", "positive": "自然言語処理はコンピュータが人間の言語を理解し生成する技術です。"},
        {"anchor": "How does a search engine work?", "positive": "検索エンジンはウェブページをクロールしインデックスを作成して関連する結果を返します。"},
        {"anchor": "What is transfer learning?", "positive": "転移学習は事前学習済みモデルを新しいタスクに適用する手法です。"},
        # English ↔ Chinese
        {"anchor": "How to fine-tune a language model?", "positive": "微调语言模型需要使用LoRA等高效方法在特定任务数据上训练。"},
        {"anchor": "What is contrastive learning?", "positive": "对比学习通过区分相似和不相似的样本对来训练模型的表示能力。"},
        {"anchor": "How does attention mechanism work?", "positive": "注意力机制通过查询-键值相似度计算加权和来捕捉序列中的依赖关系。"},
        # English ↔ English (monolingual pairs)
        {"anchor": "What is LoRA fine-tuning?", "positive": "Low-Rank Adaptation adds small trainable matrices to frozen weights for efficient model adaptation."},
        {"anchor": "How to compute sentence similarity?", "positive": "Encode sentences into embedding vectors and compute cosine similarity between them."},
        {"anchor": "What is a tokenizer?", "positive": "A component that splits text into subword tokens and maps them to numerical IDs."},
        {"anchor": "How to build a semantic search system?", "positive": "Index documents as dense vectors and retrieve the most similar ones using approximate nearest neighbors."},
        {"anchor": "What is embedding model distillation?", "positive": "Training a smaller embedding model to mimic a larger one while maintaining retrieval quality."},
    ]
    print(f"   Samples: {len(train_data)} (cross-lingual: EN↔ES/FR/DE/JA/ZH)")

    # ── Step 4: Train ────────────────────────────────────────────────────
    print("\n4. Starting training...")
    collator = EmbeddingDataCollator(
        model=model, tokenizer=tokenizer,
        anchor_column="anchor", positive_column="positive",
        max_seq_length=512,
    )

    config = EmbeddingSFTConfig(
        output_dir="./harrier_0.6b_embedding_output",
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

    # ── Step 5: Test Embeddings ──────────────────────────────────────────
    print("\n5. Testing cross-lingual embeddings...")
    FastEmbeddingModel.for_inference(model)

    test_pairs = [
        # Cross-lingual: should be similar
        ("What is machine learning?", "El aprendizaje automatico es una rama de la IA."),
        # Cross-lingual: should be similar
        ("How does deep learning work?", "Deep Learning nutzt mehrschichtige neuronale Netze."),
        # Unrelated: should be dissimilar
        ("What is machine learning?", "La recette du gateau au chocolat est simple."),
        # Monolingual: should be similar
        ("What is LoRA?", "Low-rank adaptation enables efficient fine-tuning."),
    ]

    for text_a, text_b in test_pairs:
        emb = model.encode([text_a, text_b])
        similarity = (emb[0] * emb[1]).sum().item()
        print(f"   sim={similarity:.4f} | '{text_a[:40]}' vs '{text_b[:40]}'")

    # ── Step 6: Save Adapter ─────────────────────────────────────────────
    print("\n6. Saving adapter...")
    model.save_pretrained("./harrier_0.6b_embedding_output/final_adapter")

    # ── Step 7: Test Load ────────────────────────────────────────────────
    print("\n7. Testing adapter reload...")
    model2, tok2 = FastEmbeddingModel.from_pretrained(
        model_name="microsoft/harrier-oss-v1-0.6b",
        max_seq_length=512,
        pooling_strategy="last_token",
    )
    model2 = FastEmbeddingModel.get_peft_model(model2, r=8, lora_alpha=16)
    model2._apply_lora()

    model2.load_adapter("./harrier_0.6b_embedding_output/final_adapter")
    FastEmbeddingModel.for_inference(model2)

    test_text = "How does transfer learning work?"
    emb_original = model.encode([test_text])
    emb_reloaded = model2.encode([test_text])
    similarity = (emb_original[0] * emb_reloaded[0]).sum().item()
    print(f"   Original vs reloaded similarity: {similarity:.4f}")
    print(f"   (Should be ~1.0 if adapter loaded correctly)")

    print("\nDone!")


if __name__ == "__main__":
    main()
