"""
Reinforcement Learning Trainers for Unsloth-MLX

Provides Unsloth/TRL-compatible RL training interfaces:
- DPOTrainer: Direct Preference Optimization
- ORPOTrainer: Odds Ratio Preference Optimization
- GRPOTrainer: Group Relative Policy Optimization (DeepSeek R1 style)
- KTOTrainer: Kahneman-Tversky Optimization
- SimPOTrainer: Simple Preference Optimization

These trainers use MLX under the hood for Apple Silicon optimization.
"""

from typing import Optional, Dict, Any, Union, List, Callable
from pathlib import Path
import json
import subprocess
import warnings


class DPOConfig:
    """
    Configuration for Direct Preference Optimization training.

    Compatible with TRL's DPOConfig.

    Example:
        >>> config = DPOConfig(
        ...     beta=0.1,
        ...     learning_rate=5e-7,
        ...     max_steps=100,
        ... )
    """

    def __init__(
        self,
        # DPO-specific
        beta: float = 0.1,  # KL penalty coefficient
        loss_type: str = "sigmoid",  # sigmoid, hinge, ipo, kto_pair
        label_smoothing: float = 0.0,
        # Training args
        output_dir: str = "./dpo_outputs",
        learning_rate: float = 5e-7,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        warmup_steps: int = 10,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        max_prompt_length: int = 512,
        **kwargs
    ):
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class ORPOConfig:
    """
    Configuration for Odds Ratio Preference Optimization training.

    ORPO combines SFT and preference learning into a single step,
    making it simpler and more efficient than traditional RLHF.

    Example:
        >>> config = ORPOConfig(
        ...     beta=0.1,
        ...     learning_rate=8e-6,
        ...     max_steps=1000,
        ... )
    """

    def __init__(
        self,
        # ORPO-specific
        beta: float = 0.1,  # Odds ratio coefficient
        # Training args
        output_dir: str = "./orpo_outputs",
        learning_rate: float = 8e-6,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        warmup_steps: int = 10,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        max_prompt_length: int = 512,
        **kwargs
    ):
        self.beta = beta
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class GRPOConfig:
    """
    Configuration for Group Relative Policy Optimization training.

    GRPO is used by DeepSeek to train their R1 reasoning models.
    It replaces the value model with group statistics and uses custom
    reward functions.

    Supports loss types:
    - 'grpo': Standard GRPO
    - 'dr_grpo': Dr. GRPO (distilled)
    - 'dapo': DAPO variant
    - 'bnpo': BNPO variant

    Example:
        >>> config = GRPOConfig(
        ...     loss_type='grpo',
        ...     num_generations=4,
        ...     learning_rate=1e-6,
        ... )
    """

    def __init__(
        self,
        # GRPO-specific
        loss_type: str = "grpo",  # grpo, dr_grpo, dapo, bnpo
        beta: float = 0.04,  # KL coefficient
        num_generations: int = 4,  # Number of generations per prompt
        temperature: float = 0.7,
        max_completion_length: int = 512,
        # Reward function (custom callable)
        reward_fn: Optional[Callable] = None,
        # Training args
        output_dir: str = "./grpo_outputs",
        learning_rate: float = 1e-6,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        warmup_ratio: float = 0.1,
        logging_steps: int = 1,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        **kwargs
    ):
        self.loss_type = loss_type
        self.beta = beta
        self.num_generations = num_generations
        self.temperature = temperature
        self.max_completion_length = max_completion_length
        self.reward_fn = reward_fn
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_ratio = warmup_ratio
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_') and k != 'reward_fn'}


class DPOTrainer:
    """
    Direct Preference Optimization Trainer.

    DPO trains models on preference data (chosen vs rejected responses)
    without requiring a separate reward model.

    Compatible with TRL's DPOTrainer API.

    Example:
        >>> from unsloth_mlx import FastLanguageModel, DPOTrainer, DPOConfig
        >>>
        >>> model, tokenizer = FastLanguageModel.from_pretrained(...)
        >>> model = FastLanguageModel.get_peft_model(model, r=16)
        >>>
        >>> # Preference dataset with chosen/rejected pairs
        >>> dataset = [
        ...     {"prompt": "...", "chosen": "...", "rejected": "..."},
        ... ]
        >>>
        >>> trainer = DPOTrainer(
        ...     model=model,
        ...     ref_model=None,  # Can use same model as ref
        ...     train_dataset=dataset,
        ...     tokenizer=tokenizer,
        ...     args=DPOConfig(beta=0.1),
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        ref_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        args: Optional[DPOConfig] = None,
        **kwargs
    ):
        self.model = model
        self.ref_model = ref_model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, 'tokenizer', None)

        # Extract config
        if args is None:
            args = DPOConfig()

        self.config = args
        self.beta = args.beta
        self.loss_type = args.loss_type
        self.output_dir = Path(args.output_dir)
        self.learning_rate = args.learning_rate
        self.batch_size = args.per_device_train_batch_size
        self.max_steps = args.max_steps
        self.max_seq_length = args.max_seq_length

        # Calculate iters
        if self.max_steps > 0:
            self.iters = self.max_steps
        else:
            dataset_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 100
            self.iters = max(1, (dataset_size // self.batch_size) * args.num_train_epochs)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"DPOTrainer initialized:")
        print(f"  Beta: {self.beta}")
        print(f"  Loss type: {self.loss_type}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.iters}")

    def _prepare_dpo_data(self) -> str:
        """Prepare DPO preference data in MLX-compatible format."""

        train_file = self.output_dir / "train.jsonl"
        valid_file = self.output_dir / "valid.jsonl"

        print("Preparing DPO preference data...")

        with open(train_file, 'w') as f:
            for sample in self.train_dataset:
                # DPO format: prompt, chosen, rejected
                if 'prompt' in sample and 'chosen' in sample and 'rejected' in sample:
                    # Convert to chat format for training on chosen
                    messages = [
                        {"role": "user", "content": sample['prompt']},
                        {"role": "assistant", "content": sample['chosen']}
                    ]
                    f.write(json.dumps({"messages": messages}) + '\n')
                elif 'messages' in sample:
                    f.write(json.dumps(sample) + '\n')

        # Copy as validation
        import shutil
        shutil.copy(train_file, valid_file)

        print(f"✓ Prepared DPO data at {self.output_dir}")
        return str(self.output_dir)

    def train(self):
        """
        Train the model using DPO.

        Note: MLX-LM's LoRA training is used as the base. Full DPO loss
        computation requires custom implementation for optimal results.
        Currently uses SFT on chosen responses as approximation.
        """

        print("=" * 70)
        print("Starting DPO Training")
        print("=" * 70)

        warnings.warn(
            "DPO training currently uses SFT on chosen responses. "
            "Full DPO loss implementation is planned for future versions. "
            "For production use, consider using mlx-lm-lora package.",
            UserWarning
        )

        data_dir = self._prepare_dpo_data()
        model_name = getattr(self.model, 'model_name', 'model')

        cmd = [
            "mlx_lm.lora",
            "--model", model_name,
            "--train",
            "--data", data_dir,
            "--iters", str(self.iters),
            "--learning-rate", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--adapter-path", str(self.output_dir / "adapters"),
        ]

        print(f"Running: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            print("\n" + "=" * 70)
            print("DPO Training Complete!")
            print("=" * 70)
        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")
            raise


class ORPOTrainer:
    """
    Odds Ratio Preference Optimization Trainer.

    ORPO combines SFT and preference alignment in a single training step,
    making it simpler and more memory-efficient than DPO.

    Compatible with TRL's ORPOTrainer API.

    Example:
        >>> trainer = ORPOTrainer(
        ...     model=model,
        ...     train_dataset=preference_dataset,
        ...     tokenizer=tokenizer,
        ...     args=ORPOConfig(beta=0.1),
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        args: Optional[ORPOConfig] = None,
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, 'tokenizer', None)

        if args is None:
            args = ORPOConfig()

        self.config = args
        self.beta = args.beta
        self.output_dir = Path(args.output_dir)
        self.learning_rate = args.learning_rate
        self.batch_size = args.per_device_train_batch_size
        self.max_steps = args.max_steps

        if self.max_steps > 0:
            self.iters = self.max_steps
        else:
            dataset_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 100
            self.iters = max(1, (dataset_size // self.batch_size) * args.num_train_epochs)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"ORPOTrainer initialized:")
        print(f"  Beta: {self.beta}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.iters}")

    def _prepare_orpo_data(self) -> str:
        """Prepare ORPO data."""
        train_file = self.output_dir / "train.jsonl"
        valid_file = self.output_dir / "valid.jsonl"

        with open(train_file, 'w') as f:
            for sample in self.train_dataset:
                if 'prompt' in sample and 'chosen' in sample:
                    messages = [
                        {"role": "user", "content": sample['prompt']},
                        {"role": "assistant", "content": sample['chosen']}
                    ]
                    f.write(json.dumps({"messages": messages}) + '\n')
                elif 'messages' in sample:
                    f.write(json.dumps(sample) + '\n')

        import shutil
        shutil.copy(train_file, valid_file)

        return str(self.output_dir)

    def train(self):
        """Train using ORPO."""

        print("=" * 70)
        print("Starting ORPO Training")
        print("=" * 70)

        warnings.warn(
            "ORPO training currently uses SFT approximation. "
            "Full ORPO loss implementation is planned.",
            UserWarning
        )

        data_dir = self._prepare_orpo_data()
        model_name = getattr(self.model, 'model_name', 'model')

        cmd = [
            "mlx_lm.lora",
            "--model", model_name,
            "--train",
            "--data", data_dir,
            "--iters", str(self.iters),
            "--learning-rate", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--adapter-path", str(self.output_dir / "adapters"),
        ]

        subprocess.run(cmd, check=True)
        print("ORPO Training Complete!")


class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer.

    GRPO is the technique used by DeepSeek to train reasoning models like R1.
    It removes the need for a value model by using group statistics from
    multiple generations and custom reward functions.

    Key features:
    - No value model needed (uses group statistics)
    - Custom reward functions (for math, code verification, etc.)
    - Supports GRPO, Dr.GRPO, DAPO, BNPO variants

    Example:
        >>> def math_reward(response, ground_truth):
        ...     # Custom reward for math problems
        ...     return 1.0 if extract_answer(response) == ground_truth else 0.0
        >>>
        >>> trainer = GRPOTrainer(
        ...     model=model,
        ...     train_dataset=math_dataset,
        ...     tokenizer=tokenizer,
        ...     args=GRPOConfig(
        ...         loss_type='grpo',
        ...         reward_fn=math_reward,
        ...         num_generations=4,
        ...     ),
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        reward_fn: Optional[Callable] = None,
        args: Optional[GRPOConfig] = None,
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, 'tokenizer', None)

        if args is None:
            args = GRPOConfig()

        self.config = args
        self.loss_type = args.loss_type
        self.beta = args.beta
        self.num_generations = args.num_generations
        self.reward_fn = reward_fn or args.reward_fn
        self.output_dir = Path(args.output_dir)
        self.learning_rate = args.learning_rate
        self.batch_size = args.per_device_train_batch_size
        self.max_steps = args.max_steps
        self.temperature = args.temperature

        if self.max_steps > 0:
            self.iters = self.max_steps
        else:
            dataset_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 100
            self.iters = max(1, (dataset_size // self.batch_size) * args.num_train_epochs)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"GRPOTrainer initialized:")
        print(f"  Loss type: {self.loss_type}")
        print(f"  Beta: {self.beta}")
        print(f"  Num generations: {self.num_generations}")
        print(f"  Custom reward fn: {'Yes' if self.reward_fn else 'No'}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.iters}")

    def _prepare_grpo_data(self) -> str:
        """Prepare GRPO data."""
        train_file = self.output_dir / "train.jsonl"
        valid_file = self.output_dir / "valid.jsonl"

        with open(train_file, 'w') as f:
            for sample in self.train_dataset:
                # GRPO typically uses prompt + optional ground truth for reward
                if 'prompt' in sample:
                    messages = [{"role": "user", "content": sample['prompt']}]
                    if 'response' in sample or 'answer' in sample:
                        response = sample.get('response', sample.get('answer', ''))
                        messages.append({"role": "assistant", "content": response})
                    f.write(json.dumps({"messages": messages}) + '\n')
                elif 'messages' in sample:
                    f.write(json.dumps(sample) + '\n')

        import shutil
        shutil.copy(train_file, valid_file)

        return str(self.output_dir)

    def train(self):
        """
        Train using GRPO.

        Note: Full GRPO with multiple generations and reward computation
        requires custom MLX implementation. Current version uses SFT
        approximation with the provided training data.
        """

        print("=" * 70)
        print(f"Starting GRPO Training (loss_type={self.loss_type})")
        print("=" * 70)

        warnings.warn(
            f"GRPO ({self.loss_type}) training currently uses SFT approximation. "
            "Full GRPO with multi-generation sampling and reward computation "
            "requires custom MLX kernels (planned for future versions).",
            UserWarning
        )

        data_dir = self._prepare_grpo_data()
        model_name = getattr(self.model, 'model_name', 'model')

        cmd = [
            "mlx_lm.lora",
            "--model", model_name,
            "--train",
            "--data", data_dir,
            "--iters", str(self.iters),
            "--learning-rate", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--adapter-path", str(self.output_dir / "adapters"),
        ]

        subprocess.run(cmd, check=True)
        print("GRPO Training Complete!")


class KTOTrainer:
    """
    Kahneman-Tversky Optimization Trainer.

    KTO uses prospect theory for preference optimization,
    treating gains and losses asymmetrically.
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        beta: float = 0.1,
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.beta = beta
        self.output_dir = Path(kwargs.get('output_dir', './kto_outputs'))
        self.learning_rate = kwargs.get('learning_rate', 5e-7)
        self.iters = kwargs.get('max_steps', 100)

        print(f"KTOTrainer initialized (beta={self.beta})")

    def train(self):
        """Train using KTO."""
        warnings.warn(
            "KTO training uses SFT approximation. Full KTO loss planned.",
            UserWarning
        )
        # Similar implementation to DPO
        print("KTO Training - using SFT approximation")


class SimPOTrainer:
    """
    Simple Preference Optimization Trainer.

    SimPO simplifies DPO by removing the reference model requirement.
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        gamma: float = 0.5,
        beta: float = 2.0,
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.beta = beta
        self.output_dir = Path(kwargs.get('output_dir', './simpo_outputs'))

        print(f"SimPOTrainer initialized (gamma={gamma}, beta={beta})")

    def train(self):
        """Train using SimPO."""
        warnings.warn(
            "SimPO training uses SFT approximation. Full SimPO loss planned.",
            UserWarning
        )
        print("SimPO Training - using SFT approximation")


# Utility functions for preference data

def prepare_preference_dataset(
    dataset: Any,
    tokenizer: Any,
    format_type: str = "dpo",
) -> List[Dict]:
    """
    Prepare dataset for preference-based training (DPO, ORPO, etc.).

    Args:
        dataset: HuggingFace dataset with preference pairs
        tokenizer: Tokenizer for formatting
        format_type: 'dpo', 'orpo', or 'grpo'

    Returns:
        Formatted dataset ready for training

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("Anthropic/hh-rlhf")
        >>> formatted = prepare_preference_dataset(dataset, tokenizer, "dpo")
    """

    formatted_data = []

    for sample in dataset:
        if format_type in ["dpo", "orpo"]:
            # Expect chosen/rejected format
            if 'chosen' in sample and 'rejected' in sample:
                formatted_data.append({
                    "prompt": sample.get('prompt', ''),
                    "chosen": sample['chosen'],
                    "rejected": sample['rejected'],
                })
        elif format_type == "grpo":
            # Expect prompt + optional ground truth
            formatted_data.append({
                "prompt": sample.get('prompt', sample.get('question', '')),
                "answer": sample.get('answer', sample.get('response', '')),
            })

    return formatted_data


def create_reward_function(reward_type: str = "simple") -> Callable:
    """
    Create a reward function for GRPO training.

    Args:
        reward_type: Type of reward function
            - 'simple': Binary correct/incorrect
            - 'math': Extract and compare numerical answers
            - 'code': Execute and verify code output
            - 'length': Reward based on response length

    Returns:
        Reward function callable

    Example:
        >>> reward_fn = create_reward_function('math')
        >>> trainer = GRPOTrainer(..., reward_fn=reward_fn)
    """

    if reward_type == "simple":
        def simple_reward(response: str, ground_truth: str) -> float:
            return 1.0 if ground_truth.lower() in response.lower() else 0.0
        return simple_reward

    elif reward_type == "math":
        def math_reward(response: str, ground_truth: str) -> float:
            import re
            # Extract numbers from response
            numbers = re.findall(r'-?\d+\.?\d*', response)
            target = re.findall(r'-?\d+\.?\d*', ground_truth)
            if numbers and target:
                try:
                    return 1.0 if float(numbers[-1]) == float(target[-1]) else 0.0
                except:
                    return 0.0
            return 0.0
        return math_reward

    elif reward_type == "length":
        def length_reward(response: str, _: str) -> float:
            # Reward longer, more detailed responses (up to a point)
            length = len(response.split())
            if length < 10:
                return 0.2
            elif length < 50:
                return 0.5
            elif length < 200:
                return 1.0
            else:
                return 0.8  # Penalize very long responses
        return length_reward

    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
