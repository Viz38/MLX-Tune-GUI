"""
Example 40: Gemma 4 26B MoE Vision Fine-Tuning

Fine-tune Gemma 4 26B-A4B, a Mixture of Experts VLM with 26B total
parameters but only ~4B active per token (128 experts, top-8 routing
+ 1 shared expert always active).

MLX-Tune automatically detects MoE expert layers and applies LoRA
via LoRASwitchLinear — same API as any other model.

Requirements: ~20GB unified memory for 4-bit model + LoRA training.

Usage:
    python examples/40_gemma4_moe_finetuning.py
"""

from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
from mlx_tune.vlm import VLMSFTConfig


def main():
    print("=" * 70)
    print("GEMMA 4 MoE FINE-TUNING: 26B-A4B (26B total, ~4B active)")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load MoE VLM Model
    # ========================================================================
    print("\n[Step 1] Loading Gemma 4 26B MoE model (4-bit)...")
    print("128 experts, top-8 routing + 1 shared expert per token")

    model, processor = FastVisionModel.from_pretrained(
        "mlx-community/gemma-4-26b-a4b-it-4bit",
        load_in_4bit=True,
    )

    # ========================================================================
    # Step 2: Add LoRA adapters — MoE expert layers detected automatically
    # ========================================================================
    print("\n[Step 2] Adding LoRA adapters (MoE-aware)...")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,             # Smaller rank for larger model
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    # ========================================================================
    # Step 3: Prepare dataset
    # ========================================================================
    print("\n[Step 3] Loading LaTeX OCR dataset...")

    from datasets import load_dataset

    dataset = load_dataset("unsloth/LaTeX_OCR", split="train")
    print(f"Dataset: {len(dataset)} samples")

    instruction = "Write the LaTeX representation for this image."

    def convert_to_conversation(sample):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": sample["image"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["text"]}],
                },
            ]
        }

    converted_dataset = [convert_to_conversation(s) for s in dataset]
    print(f"Converted {len(converted_dataset)} samples")

    # ========================================================================
    # Step 4: Train
    # ========================================================================
    print("\n[Step 4] Training...")

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
            max_steps=20,
            learning_rate=1e-4,
            logging_steps=1,
            optim="adam",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs_gemma4_moe",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )

    trainer_stats = trainer.train()
    print(f"\nTraining metrics: {trainer_stats.metrics}")

    # ========================================================================
    # Step 5: Save
    # ========================================================================
    print("\n[Step 5] Saving LoRA adapters...")

    model.save_pretrained("gemma4_moe_lora")
    print("Saved to gemma4_moe_lora/")

    print("\n" + "=" * 70)
    print("Done! Gemma 4 MoE fine-tuning complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
