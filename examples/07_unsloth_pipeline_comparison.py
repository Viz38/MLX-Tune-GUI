"""
Example 7: Unsloth Pipeline - Direct Comparison

This shows how Unsloth-MLX provides the SAME API as Unsloth.
Just change the import and it works!
"""

# ============================================================================
# UNSLOTH (Original) - For reference
# ============================================================================
"""
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(...)
trainer = SFTTrainer(model=model, train_dataset=dataset, args=SFTConfig(...))
trainer.train()

# Save options
model.save_pretrained("lora_model")
model.save_pretrained_merged("merged_16bit", tokenizer)
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
"""

# ============================================================================
# UNSLOTH-MAC (Ours) - Exact same workflow!
# ============================================================================

from unsloth_mlx import FastLanguageModel, SFTTrainer  # ← Only change!
from datasets import load_dataset

print("=" * 70)
print("Unsloth-MLX: Exact Same Pipeline as Unsloth")
print("=" * 70)

# 1. Load model - SAME API
print("\n1. Loading model (SAME API as Unsloth)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
print("✓ Model loaded!")

# 2. Apply LoRA - SAME API
print("\n2. Applying LoRA adapters (SAME API)...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
print("✓ LoRA configured!")

# 3. Prepare dataset - SAME as Unsloth
print("\n3. Preparing dataset (SAME as Unsloth)...")

# Create sample dataset (in real use: load_dataset("yahma/alpaca-cleaned"))
sample_dataset = [
    {
        "instruction": "What is Python?",
        "input": "",
        "output": "Python is a high-level programming language."
    },
    {
        "instruction": "Explain machine learning",
        "input": "",
        "output": "Machine learning is a subset of AI that enables systems to learn from data."
    },
    {
        "instruction": "What is 2+2?",
        "input": "",
        "output": "2+2 equals 4."
    },
]

# Format with chat template - WORKS THE SAME
def format_prompts(examples):
    texts = []
    for inst, inp, out in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    ):
        messages = [
            {"role": "user", "content": f"{inst}\n{inp}"},
            {"role": "assistant", "content": out}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

# Convert to dataset format
formatted_data = []
for item in sample_dataset:
    messages = [
        {"role": "user", "content": f"{item['instruction']}\n{item['input']}"},
        {"role": "assistant", "content": item['output']}
    ]
    formatted_data.append({"messages": messages})

print(f"✓ Dataset prepared with {len(formatted_data)} examples")

# 4. Train - SAME API (but uses MLX under the hood, not TRL)
print("\n4. Training with SFTTrainer (SAME API)...")
print("   Note: Uses MLX under the hood (not TRL), but API is compatible!")

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_data,
    tokenizer=tokenizer,
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    output_dir="unsloth_comparison_output",
    adapter_path="unsloth_comparison_adapters",
    iters=10,  # Small number for demo
)

print("\nStarting training...")
trainer.train()
print("✓ Training complete!")

# 5. Inference - SAME API
print("\n5. Inference (SAME API as Unsloth)...")
FastLanguageModel.for_inference(model)

from mlx_lm import generate
test_prompt = "What is Python?"
messages = [{"role": "user", "content": test_prompt}]
formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

response = generate(model.model, tokenizer, prompt=formatted, max_tokens=50, verbose=False)
print(f"  Q: {test_prompt}")
print(f"  A: {response[:100]}...")

# 6. Save options - NOW SAME AS UNSLOTH!
print("\n6. Save Options (NOW SAME AS UNSLOTH!)...")

print("\n   a) Save LoRA adapters only:")
print("      model.save_pretrained('lora_model')  # ✅ SAME API!")
# model.save_pretrained("lora_model")

print("\n   b) Save merged model (base + adapters):")
print("      model.save_pretrained_merged('merged_16bit', tokenizer)  # ✅ SAME API!")
# model.save_pretrained_merged("merged_16bit", tokenizer, save_method="merged_16bit")

print("\n   c) Save as GGUF for llama.cpp/Ollama:")
print("      model.save_pretrained_gguf('model', tokenizer, quantization_method='q4_k_m')  # ✅ SAME API!")
# model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")

print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print("\n✅ SAME as Unsloth:")
print("  - FastLanguageModel.from_pretrained()")
print("  - FastLanguageModel.get_peft_model()")
print("  - SFTTrainer(...)")
print("  - trainer.train()")
print("  - FastLanguageModel.for_inference()")
print("  - model.save_pretrained()")
print("  - model.save_pretrained_merged()")
print("  - model.save_pretrained_gguf()")
print("  - load_dataset() from HuggingFace")
print("  - tokenizer.apply_chat_template()")

print("\n⚠️  DIFFERENT (but compatible):")
print("  - Backend: MLX instead of CUDA/Triton")
print("  - Trainer: MLX-based instead of TRL-based")
print("  - Platform: Apple Silicon instead of NVIDIA")

print("\n💡 ADVANTAGE:")
print("  - Develop locally on Mac")
print("  - Deploy to CUDA just by changing import!")
print("  - Code is 99% identical")

print("\n" + "=" * 70)
print("Just like Unsloth, but for Mac! 🚀")
print("=" * 70)


if __name__ == "__main__":
    pass
