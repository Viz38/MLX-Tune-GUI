"""
Example 39: Gemma 4 Text-to-SQL Fine-Tuning

Text-only fine-tuning through the VLM path (no images needed).
Uses the same dataset as Google's official Gemma 4 fine-tuning guide:
https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

Dataset: philschmid/gretel-synthetic-text-to-sql
Task: Convert natural language questions to SQL queries

NOTE: Gemma 4 models are all VLMs — use FastVisionModel even for text tasks.

Usage:
    python examples/39_gemma4_text_to_sql.py
"""

from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
from mlx_tune.vlm import VLMSFTConfig


def main():
    print("=" * 70)
    print("GEMMA 4 TEXT-TO-SQL: Google's Official Fine-Tuning Example")
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
    # Step 2: Add LoRA adapters (text layers only)
    # ========================================================================
    print("\n[Step 2] Adding LoRA adapters (language layers only)...")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,     # No vision for text-only task
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    # ========================================================================
    # Step 3: Prepare Text-to-SQL dataset
    # ========================================================================
    print("\n[Step 3] Loading Text-to-SQL dataset...")

    from datasets import load_dataset

    dataset = load_dataset("philschmid/gretel-synthetic-text-to-sql", split="train")
    print(f"Dataset: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")

    # Take a subset for demo
    dataset = dataset.select(range(min(500, len(dataset))))
    print(f"Using {len(dataset)} samples for demo")

    def format_text_to_sql(sample):
        """Format as chat conversation for text-only VLM fine-tuning."""
        # Build the user prompt with schema context
        user_msg = f"Given the following SQL schema:\n{sample['sql_context']}\n\n"
        user_msg += f"Write a SQL query for: {sample['sql_prompt']}"

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_msg}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["sql"]}],
                },
            ]
        }

    converted_dataset = [format_text_to_sql(s) for s in dataset]
    print(f"Converted {len(converted_dataset)} samples")
    print(f"\nSample prompt:\n{converted_dataset[0]['messages'][0]['content'][0]['text'][:200]}...")
    print(f"\nSample SQL:\n{converted_dataset[0]['messages'][1]['content'][0]['text'][:200]}...")

    # ========================================================================
    # Step 4: Pre-training inference
    # ========================================================================
    print("\n[Step 4] Pre-training inference test...")

    FastVisionModel.for_inference(model)

    test_prompt = (
        "Given the following SQL schema:\n"
        "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary DECIMAL);\n\n"
        "Write a SQL query for: Find the average salary by department"
    )

    try:
        response = model.generate(
            prompt=test_prompt,
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
            output_dir="outputs_gemma4_sql",
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
            prompt=test_prompt,
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

    model.save_pretrained("gemma4_sql_lora")
    print("Saved to gemma4_sql_lora/")

    print("\n" + "=" * 70)
    print("Done! Gemma 4 Text-to-SQL fine-tuning complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
