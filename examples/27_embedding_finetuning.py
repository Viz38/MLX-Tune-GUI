"""
Example 27: Embedding Model Fine-Tuning with MLX-Tune

Fine-tune sentence embedding models (BERT, ModernBERT, etc.) for semantic
similarity on Apple Silicon using contrastive learning (InfoNCE loss).

In-batch negatives: each other positive in the batch acts as a negative,
so larger batch sizes give better training signal.

Requirements:
    uv pip install mlx-tune

Usage:
    python examples/27_embedding_finetuning.py
"""

from mlx_tune import (
    FastEmbeddingModel,
    EmbeddingSFTTrainer,
    EmbeddingSFTConfig,
    EmbeddingDataCollator,
)


def main():
    print("=" * 70)
    print("MLX-Tune: Embedding Model Fine-Tuning")
    print("=" * 70)

    # ── Step 1: Load Embedding Model ──────────────────────────────────────
    print("\n1. Loading embedding model...")
    model, tokenizer = FastEmbeddingModel.from_pretrained(
        model_name="mlx-community/all-MiniLM-L6-v2-bf16",
        max_seq_length=256,
        pooling_strategy="mean",
    )
    print(f"   Model: {model.model_name}")
    print(f"   Architecture: {model.architecture}")
    print(f"   Pooling: {model.pooling_strategy}")

    # ── Step 2: Add LoRA ──────────────────────────────────────────────────
    print("\n2. Adding LoRA adapters...")
    model = FastEmbeddingModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["query", "key", "value"],
    )
    print(f"   LoRA rank: {model.lora_config['r']}")
    print(f"   Targets: {model.lora_config['target_modules']}")

    # ── Step 3: Prepare Dataset ───────────────────────────────────────────
    print("\n3. Preparing training data...")

    # Synthetic NLI-style anchor-positive pairs
    train_data = [
        {"anchor": "A man is eating food.", "positive": "A man is having a meal."},
        {"anchor": "A woman is playing guitar.", "positive": "Someone is playing a musical instrument."},
        {"anchor": "Two dogs are running in a park.", "positive": "Dogs are playing outside."},
        {"anchor": "A child is reading a book.", "positive": "A young person is studying."},
        {"anchor": "The sun is setting over the ocean.", "positive": "A beautiful sunset by the sea."},
        {"anchor": "A cat is sleeping on a couch.", "positive": "A feline is resting on furniture."},
        {"anchor": "People are dancing at a party.", "positive": "Guests are enjoying music and movement."},
        {"anchor": "A bird is flying in the sky.", "positive": "An avian creature soars through the air."},
        {"anchor": "The car is parked in the garage.", "positive": "A vehicle is stored in a covered space."},
        {"anchor": "A student is typing on a laptop.", "positive": "Someone is working on a computer."},
        {"anchor": "Rain is falling on the street.", "positive": "Water drops are hitting the pavement."},
        {"anchor": "A chef is cooking in the kitchen.", "positive": "A cook is preparing food."},
        {"anchor": "Children are playing in the playground.", "positive": "Kids are having fun outside."},
        {"anchor": "A train is arriving at the station.", "positive": "A locomotive is pulling into the terminal."},
        {"anchor": "The flowers are blooming in spring.", "positive": "Plants are blossoming in the garden."},
        {"anchor": "A dog is barking at the mailman.", "positive": "A canine is making noise at the postal worker."},
        {"anchor": "Snow is covering the mountains.", "positive": "White precipitation blankets the peaks."},
        {"anchor": "A teacher is writing on the board.", "positive": "An educator is explaining a lesson."},
        {"anchor": "Fish are swimming in the pond.", "positive": "Aquatic creatures are moving through water."},
        {"anchor": "A plane is taking off from the runway.", "positive": "An aircraft is departing from the airport."},
        {"anchor": "The moon is shining brightly tonight.", "positive": "The lunar body illuminates the night sky."},
        {"anchor": "A baby is crying in the crib.", "positive": "An infant is making sounds in bed."},
        {"anchor": "Trees are swaying in the wind.", "positive": "The breeze is moving the branches."},
        {"anchor": "A painter is creating a masterpiece.", "positive": "An artist is working on a painting."},
        {"anchor": "Waves are crashing on the shore.", "positive": "Ocean water is hitting the beach."},
        {"anchor": "A musician is playing the piano.", "positive": "Someone is performing a keyboard instrument."},
        {"anchor": "The leaves are changing color in autumn.", "positive": "Fall foliage is turning red and orange."},
        {"anchor": "A runner is jogging in the park.", "positive": "Someone is exercising outdoors."},
        {"anchor": "Stars are twinkling in the night sky.", "positive": "Celestial objects are visible above."},
        {"anchor": "A baker is making fresh bread.", "positive": "Someone is preparing baked goods."},
    ]
    print(f"   Samples: {len(train_data)}")

    # ── Step 4: Create Data Collator ──────────────────────────────────────
    collator = EmbeddingDataCollator(
        model=model,
        tokenizer=tokenizer,
        anchor_column="anchor",
        positive_column="positive",
        max_seq_length=256,
    )

    # ── Step 5: Configure and Train ───────────────────────────────────────
    print("\n4. Starting training...")
    config = EmbeddingSFTConfig(
        output_dir="./embedding_output",
        per_device_train_batch_size=15,  # Large batch for better in-batch negatives
        learning_rate=2e-5,
        max_steps=50,
        temperature=0.05,
        loss_type="infonce",
        logging_steps=5,
        max_seq_length=256,
    )

    trainer = EmbeddingSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_data,
        args=config,
    )
    result = trainer.train()
    print(f"\n   Final avg loss: {result.metrics['train_loss']:.4f}")

    # ── Step 6: Test Embeddings ───────────────────────────────────────────
    print("\n5. Testing embeddings...")
    FastEmbeddingModel.for_inference(model)

    test_pairs = [
        ("A dog is running.", "A canine is moving fast."),
        ("A dog is running.", "The stock market crashed."),
        ("Machine learning models are powerful.", "AI systems can solve complex problems."),
        ("Machine learning models are powerful.", "The cake recipe requires butter."),
    ]

    for text_a, text_b in test_pairs:
        emb = model.encode([text_a, text_b])
        similarity = (emb[0] * emb[1]).sum().item()
        print(f"   sim={similarity:.4f} | '{text_a[:40]}' vs '{text_b[:40]}'")

    # ── Step 7: Save ──────────────────────────────────────────────────────
    print("\n6. Saving adapter...")
    model.save_pretrained("./embedding_output/final_adapter")

    print("\nDone!")


if __name__ == "__main__":
    main()
