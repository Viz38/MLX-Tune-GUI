"""
Example 38: Gemma 4 Text Fine-Tuning with MLX-Tune

Fine-tune Google's Gemma 4 E4B model on a text task through the VLM path.
Gemma 4 models are all multimodal VLMs and must use FastVisionModel,
even for text-only tasks.

All 4 Gemma 4 variants are supported:
  - gemma-4-E2B-it  (~1GB 4-bit)  — edge/mobile, vision+audio
  - gemma-4-E4B-it  (~2GB 4-bit)  — edge/mobile, vision+audio
  - gemma-4-27b-it  (~14GB 4-bit) — 26B MoE, vision
  - gemma-4-31b-it  (~17GB 4-bit) — 31B dense, vision

NOTE: Vision training with images currently produces NaN gradients on
Gemma 4 (mlx-vlm backward pass issue). Use text-only training for now.
See example 39 for Text-to-SQL, or use Qwen3.5 for vision fine-tuning.

Usage:
    python examples/38_gemma4_vision_finetuning.py
"""

from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
from mlx_tune.vlm import VLMSFTConfig


def main():
    print("=" * 70)
    print("GEMMA 4 TEXT FINE-TUNING (VLM Path)")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load Gemma 4 model
    # ========================================================================
    print("\n[Step 1] Loading Gemma 4 E4B model...")

    model, processor = FastVisionModel.from_pretrained(
        "mlx-community/gemma-4-e4b-it-4bit",
        load_in_4bit=True,
    )

    # ========================================================================
    # Step 2: Add LoRA adapters
    # ========================================================================
    print("\n[Step 2] Adding LoRA adapters...")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,     # Text-only training
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
    )

    # ========================================================================
    # Step 3: Prepare dataset (text-only, no images)
    # ========================================================================
    print("\n[Step 3] Preparing text dataset...")

    dataset = [
        {"instruction": "Explain the Pythagorean theorem.", "output": "The Pythagorean theorem states that in a right triangle, a^2 + b^2 = c^2, where c is the hypotenuse. For example, 3^2 + 4^2 = 5^2."},
        {"instruction": "What is photosynthesis?", "output": "Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll in their chloroplasts."},
        {"instruction": "Explain Newton's first law.", "output": "Newton's first law states that an object at rest stays at rest, and an object in motion stays in motion at constant velocity, unless acted upon by an external force."},
        {"instruction": "What causes rain?", "output": "Rain forms when water evaporates, rises, cools and condenses into clouds, then falls as precipitation when droplets become heavy enough."},
        {"instruction": "Explain what DNA is.", "output": "DNA (deoxyribonucleic acid) is a double-helix molecule that carries genetic instructions for development, functioning, and reproduction of all living organisms."},
    ] * 10

    converted_dataset = [
        {"messages": [
            {"role": "user", "content": [{"type": "text", "text": s["instruction"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": s["output"]}]},
        ]}
        for s in dataset
    ]
    print(f"Dataset: {len(converted_dataset)} samples")

    # ========================================================================
    # Step 4: Pre-training inference
    # ========================================================================
    print("\n[Step 4] Pre-training inference test...")

    FastVisionModel.for_inference(model)

    try:
        response = model.generate(
            prompt="Explain the Pythagorean theorem.",
            max_tokens=128,
            temperature=0.3,
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Pre-training inference error: {e}")

    # ========================================================================
    # Step 5: Train
    # ========================================================================
    print("\n[Step 5] Training...")

    FastVisionModel.for_training(model)

    trainer = VLMSFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=converted_dataset,
        args=VLMSFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adam",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs_gemma4_text",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=512,
        ),
    )

    trainer_stats = trainer.train()
    print(f"\nTraining metrics: {trainer_stats.metrics}")

    # ========================================================================
    # Step 6: Post-training inference
    # ========================================================================
    print("\n[Step 6] Post-training inference test...")

    FastVisionModel.for_inference(model)

    try:
        response = model.generate(
            prompt="Explain the Pythagorean theorem.",
            max_tokens=128,
            temperature=0.3,
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Post-training inference error: {e}")

    # ========================================================================
    # Step 7: Save
    # ========================================================================
    print("\n[Step 7] Saving LoRA adapters...")

    model.save_pretrained("gemma4_vision_lora")
    print("Saved to gemma4_vision_lora/")

    print("\n" + "=" * 70)
    print("Done! Gemma 4 vision fine-tuning complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
