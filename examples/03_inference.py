"""
Example 3: Inference with Unsloth-MLX

This example demonstrates how to use the model for text generation,
showing compatibility with Unsloth's inference API.
"""

from unsloth_mlx import FastLanguageModel

def main():
    print("=" * 60)
    print("Unsloth-MLX Example: Inference")
    print("=" * 60)

    # Load model
    print("\n1. Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    print("✓ Model loaded!")

    # Enable inference mode
    print("\n2. Enabling inference mode...")
    FastLanguageModel.for_inference(model)
    print("✓ Inference mode enabled!")

    # Prepare a prompt
    print("\n3. Generating text...")
    prompt = "What is the capital of France?"

    # Format as chat message (Llama 3 format)
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    print(f"   Prompt: {prompt}")
    print(f"   Formatted prompt length: {len(formatted_prompt)} chars")

    # Generate response
    print("\n   Generating response...")
    print("   " + "-" * 56)

    from mlx_lm import generate

    response = generate(
        model.model,  # Access the underlying MLX model
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=100,
        verbose=False,
    )

    print(f"   Response: {response}")
    print("   " + "-" * 56)

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
