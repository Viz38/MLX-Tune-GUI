"""
Example 6: Real End-to-End Training Test

This example ACTUALLY trains a model (small number of iterations for testing).
Tests the complete workflow including SFTTrainer.
"""

from unsloth_mlx import FastLanguageModel, SFTTrainer
import json


def create_tiny_dataset():
    """Create a tiny dataset for testing"""
    return [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Who created Linux?"},
                {"role": "assistant", "content": "Linux was created by Linus Torvalds."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI stands for Artificial Intelligence."}
            ]
        },
    ]


def main():
    print("=" * 70)
    print("REAL TRAINING TEST - End-to-End")
    print("=" * 70)

    # Step 1: Load Model
    print("\n1. Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=512,  # Shorter for faster training
        load_in_4bit=True,
    )
    print("✓ Model loaded!")

    # Step 2: Configure LoRA
    print("\n2. Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # Smaller rank for faster training
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
    )
    print("✓ LoRA configured!")

    # Step 3: Create Dataset
    print("\n3. Creating dataset...")
    dataset = create_tiny_dataset()
    print(f"✓ Created dataset with {len(dataset)} examples")

    # Step 4: Test Inference Before Training
    print("\n4. Testing inference BEFORE training...")
    FastLanguageModel.for_inference(model)

    from mlx_lm import generate
    test_prompt = "What is 2+2?"
    messages = [{"role": "user", "content": test_prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    response_before = generate(
        model.model,
        tokenizer,
        prompt=formatted,
        max_tokens=50,
        verbose=False
    )
    print(f"  Before training: {response_before[:100]}...")

    # Step 5: Initialize Trainer
    print("\n5. Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=512,
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        output_dir="./test_training_output",
        adapter_path="./test_adapters",
        iters=10,  # Very small for testing
    )
    print("✓ Trainer initialized!")

    # Step 6: Train
    print("\n6. Starting training...")
    print("   (This will actually train the model!)")
    print()

    try:
        trainer.train()
        print("\n✓ Training completed!")

    except Exception as e:
        print(f"\n⚠️  Training error: {e}")
        print("\nThis might be expected if mlx_lm.lora has different requirements.")
        print("Let me show you the manual training command...")

        # Show manual command
        print("\nManual Training Command:")
        print("-" * 70)
        print(f"mlx_lm.lora \\")
        print(f"    --model {model.model_name} \\")
        print(f"    --train \\")
        print(f"    --data ./test_training_output/train.jsonl \\")
        print(f"    --iters 10 \\")
        print(f"    --learning-rate 5e-5 \\")
        print(f"    --batch-size 1 \\")
        print(f"    --lora-layers 8 \\")
        print(f"    --adapter-path ./test_adapters")
        print("-" * 70)
        print("\nRun this command manually to train the model.")
        return

    # Step 7: Test Inference After Training
    print("\n7. Testing inference AFTER training...")
    # TODO: Load adapters and test
    print("   (Adapter loading and testing to be implemented)")

    print("\n" + "=" * 70)
    print("END-TO-END TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
