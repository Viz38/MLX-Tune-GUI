"""
Example 1: Simple Model Loading

This example demonstrates how to load a model using Unsloth-MLX,
showing the API compatibility with Unsloth.
"""

from unsloth_mlx import FastLanguageModel

def main():
    print("=" * 60)
    print("Unsloth-MLX Example: Simple Model Loading")
    print("=" * 60)

    # Load a small quantized model from MLX community
    print("\n1. Loading Llama 3.2 1B Instruct (4-bit quantized)...")
    print("   Model: mlx-community/Llama-3.2-1B-Instruct-4bit")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
        load_in_4bit=True,  # This is handled by MLX automatically
    )

    print("✓ Model loaded successfully!")
    print(f"   Model type: {type(model)}")
    print(f"   Tokenizer type: {type(tokenizer)}")
    print(f"   Max sequence length: {model.max_seq_length}")

    # Test tokenization
    print("\n2. Testing tokenization...")
    test_text = "Hello, how are you?"
    tokens = tokenizer.encode(test_text)
    print(f"   Input: '{test_text}'")
    print(f"   Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"   Tokens: {tokens}")
    print(f"   Number of tokens: {len(tokens)}")

    # Test decoding
    decoded = tokenizer.decode(tokens)
    print(f"   Decoded: '{decoded}'")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
