"""
Example 32: Microsoft Harrier 270M Embedding Fine-Tuning with MLX-Tune

Fine-tune Microsoft Harrier-OSS-v1-270M for code/documentation search
on Apple Silicon using contrastive learning (InfoNCE loss).

Harrier 270M is a lightweight multilingual embedding model (~540MB).
It uses a Gemma3 decoder backbone with last-token pooling — ideal for
fast iteration and resource-constrained environments.

Requirements:
    uv pip install mlx-tune

Usage:
    python examples/32_harrier_270m_embedding_finetuning.py
"""

from mlx_tune import (
    FastEmbeddingModel,
    EmbeddingSFTTrainer,
    EmbeddingSFTConfig,
    EmbeddingDataCollator,
)


def main():
    print("=" * 70)
    print("MLX-Tune: Microsoft Harrier 270M Embedding Fine-Tuning")
    print("=" * 70)

    # ── Step 1: Load Harrier 270M ────────────────────────────────────────
    # Harrier 270M uses a Gemma3 decoder backbone (model_type: "gemma3_text")
    # with last-token pooling — architecture auto-detected as "gemma"
    print("\n1. Loading Microsoft Harrier-OSS-v1-270M...")
    model, tokenizer = FastEmbeddingModel.from_pretrained(
        model_name="microsoft/harrier-oss-v1-270m",
        max_seq_length=256,
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
    print("\n3. Preparing code/documentation search data...")

    # Code and documentation Q&A pairs for domain-specific search
    train_data = [
        {"anchor": "How to sort a list in Python?", "positive": "Use sorted() for a new list or list.sort() for in-place sorting. Both accept key and reverse parameters."},
        {"anchor": "What is a decorator in Python?", "positive": "A decorator wraps a function to modify its behavior using @syntax. It takes a function and returns a new function."},
        {"anchor": "How to handle exceptions?", "positive": "Use try/except blocks to catch exceptions. Add finally for cleanup and else for code that runs when no exception occurs."},
        {"anchor": "What is list comprehension?", "positive": "A concise way to create lists: [expr for item in iterable if condition]. Faster than equivalent for loops."},
        {"anchor": "How to read a file in Python?", "positive": "Use open() with a context manager: with open('file.txt', 'r') as f: content = f.read()"},
        {"anchor": "What is a virtual environment?", "positive": "An isolated Python installation that keeps project dependencies separate. Create with python -m venv or uv venv."},
        {"anchor": "How to install packages?", "positive": "Use pip install package_name or uv pip install for faster resolution. Pin versions in requirements.txt."},
        {"anchor": "What is type hinting?", "positive": "Optional annotations that specify expected types: def greet(name: str) -> str. Checked by mypy, not enforced at runtime."},
        {"anchor": "How to create a class?", "positive": "Define with class keyword, use __init__ for constructor. Methods take self as first argument for instance access."},
        {"anchor": "What is async/await?", "positive": "Coroutine-based concurrency: async def marks coroutines, await pauses execution until the awaited task completes."},
        {"anchor": "How to use git branches?", "positive": "Create with git checkout -b name, switch with git checkout name, merge with git merge name, delete with git branch -d name."},
        {"anchor": "What is a REST API?", "positive": "An architectural style using HTTP methods (GET, POST, PUT, DELETE) on resource URLs with JSON request/response bodies."},
        {"anchor": "How to write unit tests?", "positive": "Use pytest: define functions starting with test_, use assert statements, run with pytest command. Fixtures provide test data."},
        {"anchor": "What is dependency injection?", "positive": "A design pattern where dependencies are passed to a class rather than created internally, improving testability and flexibility."},
        {"anchor": "How to debug Python code?", "positive": "Use breakpoint() or pdb.set_trace() for interactive debugging. Print statements, logging module, or IDE debuggers also work."},
        {"anchor": "What is Docker?", "positive": "A containerization platform that packages applications with dependencies into portable images that run consistently everywhere."},
        {"anchor": "How to use environment variables?", "positive": "Access with os.environ['KEY'] or os.getenv('KEY', default). Store secrets in .env files loaded with python-dotenv."},
        {"anchor": "What is CI/CD?", "positive": "Continuous Integration runs tests on every commit. Continuous Deployment automatically releases passing builds to production."},
        {"anchor": "How to optimize Python performance?", "positive": "Profile with cProfile, use generators for memory, leverage numpy for arrays, cache with functools.lru_cache."},
        {"anchor": "What is a context manager?", "positive": "An object implementing __enter__ and __exit__ for resource management. Use with statement for automatic cleanup."},
    ]
    print(f"   Samples: {len(train_data)} (code/documentation pairs)")

    # ── Step 4: Train ────────────────────────────────────────────────────
    print("\n4. Starting training...")
    collator = EmbeddingDataCollator(
        model=model, tokenizer=tokenizer,
        anchor_column="anchor", positive_column="positive",
        max_seq_length=256,
    )

    config = EmbeddingSFTConfig(
        output_dir="./harrier_270m_embedding_output",
        per_device_train_batch_size=10,
        learning_rate=2e-5,
        max_steps=30,
        temperature=0.05,
        loss_type="infonce",
        logging_steps=5,
        max_seq_length=256,
    )

    trainer = EmbeddingSFTTrainer(
        model=model, tokenizer=tokenizer,
        data_collator=collator, train_dataset=train_data,
        args=config,
    )
    result = trainer.train()
    print(f"\n   Final avg loss: {result.metrics['train_loss']:.4f}")

    # ── Step 5: Test Embeddings ──────────────────────────────────────────
    print("\n5. Testing code search embeddings...")
    FastEmbeddingModel.for_inference(model)

    test_pairs = [
        # Related: should be similar
        ("How to sort a list?", "Use sorted() or list.sort() to order elements."),
        # Related: should be similar
        ("How to write tests?", "Use pytest with assert statements for unit testing."),
        # Unrelated: should be dissimilar
        ("How to sort a list?", "Docker packages applications into containers."),
        # Related: should be similar
        ("What is async programming?", "Coroutines with async/await for concurrent execution."),
    ]

    for text_a, text_b in test_pairs:
        emb = model.encode([text_a, text_b])
        similarity = (emb[0] * emb[1]).sum().item()
        print(f"   sim={similarity:.4f} | '{text_a[:40]}' vs '{text_b[:40]}'")

    # ── Step 6: Save Adapter ─────────────────────────────────────────────
    print("\n6. Saving adapter...")
    model.save_pretrained("./harrier_270m_embedding_output/final_adapter")

    # ── Step 7: Test Load ────────────────────────────────────────────────
    print("\n7. Testing adapter reload...")
    model2, tok2 = FastEmbeddingModel.from_pretrained(
        model_name="microsoft/harrier-oss-v1-270m",
        max_seq_length=256,
        pooling_strategy="last_token",
    )
    model2 = FastEmbeddingModel.get_peft_model(model2, r=8, lora_alpha=16)
    model2._apply_lora()

    model2.load_adapter("./harrier_270m_embedding_output/final_adapter")
    FastEmbeddingModel.for_inference(model2)

    test_text = "How to handle exceptions in Python?"
    emb_original = model.encode([test_text])
    emb_reloaded = model2.encode([test_text])
    similarity = (emb_original[0] * emb_reloaded[0]).sum().item()
    print(f"   Original vs reloaded similarity: {similarity:.4f}")
    print(f"   (Should be ~1.0 if adapter loaded correctly)")

    print("\nDone!")


if __name__ == "__main__":
    main()
