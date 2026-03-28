"""
Example 26: Vision GRPO (Group Relative Policy Optimization) Training

End-to-end Vision GRPO training for building vision-language reasoning models.
Extends GRPO to VLMs — the model generates text completions conditioned on
image+text prompts, scored by reward functions, and updated via policy gradient
with group-normalized advantages.

Similar to Unsloth's Qwen VL GRPO notebooks but running natively on Apple
Silicon via MLX. Unsloth uses TRL's GRPOTrainer with datasets like MathVista
for hard math reasoning with visual diagrams.

NOTE: With simple tasks and a capable model, all completions may receive equal
rewards (steps show "skipped, equal rewards"). This is expected — GRPO requires
reward variance between completions to compute advantages. For real training,
use challenging tasks where the model sometimes fails (e.g., MathVista,
geometric reasoning, chart interpretation).

Usage:
    python examples/26_vision_grpo_training.py
"""

import re
from PIL import Image, ImageDraw
from mlx_tune import FastVisionModel
from mlx_tune.vlm import VLMGRPOTrainer, VLMGRPOConfig


# System prompt that encourages structured visual reasoning
SYSTEM_PROMPT = """You are a helpful assistant that analyzes images and answers questions.
Always show your reasoning before giving the final answer.
Format your response as:
<reasoning>
[Your visual analysis here]
</reasoning>
<answer>
[Your final answer here]
</answer>"""


# ---- Synthetic image generators ----

def create_color_image(color, size=(200, 200)):
    """Create a solid color image."""
    return Image.new('RGB', size, color=color)


def create_split_image(left_color, right_color, size=(200, 200)):
    """Create an image split vertically into two colors."""
    img = Image.new('RGB', size)
    draw = ImageDraw.Draw(img)
    mid = size[0] // 2
    draw.rectangle([0, 0, mid, size[1]], fill=left_color)
    draw.rectangle([mid, 0, size[0], size[1]], fill=right_color)
    return img


def create_circles_image(count, size=(200, 200)):
    """Create an image with a specific number of filled circles."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    positions = [
        (30, 30), (100, 30), (170, 30),
        (30, 100), (100, 100), (170, 100),
        (30, 170), (100, 170), (170, 170),
    ]
    for i in range(min(count, len(positions))):
        x, y = positions[i]
        c = colors[i % len(colors)]
        draw.ellipse([x - 15, y - 15, x + 15, y + 15], fill=c)
    return img


# ---- Reward functions ----

def correctness_reward(response: str, ground_truth: str) -> float:
    """
    Reward for getting the correct answer.
    Strict: only checks inside <answer> tags. No tags = no credit.
    This ensures reward variance when some completions use tags and others don't.
    """
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if not match:
        return 0.0  # Must use <answer> tags

    extracted = match.group(1).strip().lower()
    truth = ground_truth.strip().lower()

    # Exact match
    if truth in extracted:
        return 1.0

    # Numeric comparison
    response_nums = re.findall(r'-?\d+\.?\d*', extracted)
    truth_nums = re.findall(r'-?\d+\.?\d*', truth)
    if response_nums and truth_nums:
        try:
            if float(response_nums[-1]) == float(truth_nums[-1]):
                return 1.0
        except ValueError:
            pass

    return 0.0


def format_reward(response: str, ground_truth: str) -> float:
    """
    Reward for following the expected output format.
    Checks for <reasoning> and <answer> tags, and rewards detailed reasoning.
    Varying reasoning lengths create reward variance between completions.
    """
    reasoning_match = re.search(
        r'<reasoning>(.*?)</reasoning>', response, re.DOTALL
    )
    has_answer = bool(
        re.search(r'<answer>.*?</answer>', response, re.DOTALL)
    )

    if reasoning_match and has_answer:
        reasoning_text = reasoning_match.group(1).strip()
        word_count = len(reasoning_text.split())
        if word_count >= 20:
            return 1.0
        elif word_count >= 10:
            return 0.7
        elif word_count >= 5:
            return 0.5
        return 0.3
    elif reasoning_match or has_answer:
        return 0.2
    return 0.0


def combined_reward(response: str, ground_truth: str) -> float:
    """Combined reward: 70% correctness + 30% format."""
    return (0.7 * correctness_reward(response, ground_truth) +
            0.3 * format_reward(response, ground_truth))


def main():
    print("=" * 70)
    print("Vision GRPO Training — End-to-End (VLM Reasoning)")
    print("=" * 70)

    # 1. Load VLM
    # Any mlx-vlm model works: Qwen3.5, Qwen2-VL, LLaVA, Gemma 3, etc.
    model, processor = FastVisionModel.from_pretrained(
        "mlx-community/Qwen3.5-0.8B-bf16",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,      # Keep vision encoder frozen
        finetune_language_layers=True,      # Train language layers
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
    )

    # 2. Vision reasoning dataset — image + question + expected answer
    vision_data = [
        {
            "prompt": f"{SYSTEM_PROMPT}\n\nWhat is the main color of this image?",
            "image": create_color_image('red'),
            "answer": "red",
        },
        {
            "prompt": f"{SYSTEM_PROMPT}\n\nWhat is the main color of this image?",
            "image": create_color_image('blue'),
            "answer": "blue",
        },
        {
            "prompt": f"{SYSTEM_PROMPT}\n\nWhat color is on the left side of this image?",
            "image": create_split_image('green', 'yellow'),
            "answer": "green",
        },
        {
            "prompt": f"{SYSTEM_PROMPT}\n\nWhat color is on the right side of this image?",
            "image": create_split_image('red', 'blue'),
            "answer": "blue",
        },
        {
            "prompt": f"{SYSTEM_PROMPT}\n\nHow many circles are in this image?",
            "image": create_circles_image(3),
            "answer": "3",
        },
        {
            "prompt": f"{SYSTEM_PROMPT}\n\nHow many circles are in this image?",
            "image": create_circles_image(5),
            "answer": "5",
        },
        {
            "prompt": f"{SYSTEM_PROMPT}\n\nWhat is the main color of this image?",
            "image": create_color_image('green'),
            "answer": "green",
        },
        {
            "prompt": f"{SYSTEM_PROMPT}\n\nWhat color is on the left side of this image?",
            "image": create_split_image('purple', 'orange'),
            "answer": "purple",
        },
        {
            "prompt": f"{SYSTEM_PROMPT}\n\nHow many circles are in this image?",
            "image": create_circles_image(2),
            "answer": "2",
        },
        {
            "prompt": f"{SYSTEM_PROMPT}\n\nWhat is the main color of this image?",
            "image": create_color_image('yellow'),
            "answer": "yellow",
        },
    ]

    # 3. Configure Vision GRPO
    # Matches Unsloth's GRPO parameters for vision models.
    # For real training, use max_steps=60+ and a harder dataset like MathVista.
    config = VLMGRPOConfig(
        beta=0.04,                       # KL coefficient
        num_generations=2,               # Generate 2 completions per prompt
        temperature=0.9,                 # Sampling temperature (Unsloth uses 0.9)
        max_completion_length=256,       # Max tokens per completion
        learning_rate=5e-6,              # Unsloth uses 5e-6 for vision GRPO
        max_steps=10,                    # Quick demo (Unsloth uses 60)
        logging_steps=1,                 # Log every step
        output_dir="./vlm_grpo_output",
    )

    # 4. Create trainer with custom reward function
    trainer = VLMGRPOTrainer(
        model=model,
        train_dataset=vision_data,
        processor=processor,
        reward_fn=combined_reward,
        args=config,
    )

    print(f"\nReward function: 70% correctness + 30% format")
    print(f"Expected format: <reasoning>...</reasoning><answer>...</answer>")

    # 5. Train!
    result = trainer.train()

    print(f"\nResult: {result['status']}")
    print(f"Adapters saved to: {result['adapter_path']}")


if __name__ == "__main__":
    main()
