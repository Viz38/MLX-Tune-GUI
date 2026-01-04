"""
Vision Language Model (VLM) Support for Unsloth-MLX

Provides support for Vision-Language models like:
- Qwen2-VL / Qwen2.5-VL
- LLaVA
- Pixtral
- And other VLMs supported by MLX-VLM

Uses mlx-vlm package under the hood for Apple Silicon optimization.
"""

from typing import Optional, Any, List, Dict, Union
from pathlib import Path
import warnings


class FastVisionModel:
    """
    Unsloth-compatible API for Vision Language Models.

    This class provides the same API patterns as FastLanguageModel but
    for vision-language models, using MLX-VLM under the hood.

    Supported models:
    - Qwen2-VL / Qwen2.5-VL (recommended)
    - LLaVA 1.5 / 1.6
    - Pixtral
    - Idefics 2/3
    - PaliGemma

    Example:
        >>> from unsloth_mlx import FastVisionModel
        >>>
        >>> model, processor = FastVisionModel.from_pretrained(
        ...     model_name="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        ...     max_seq_length=2048,
        ... )
        >>>
        >>> # Generate with image
        >>> response = model.generate(
        ...     prompt="Describe this image",
        ...     image_path="photo.jpg",
        ... )
    """

    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: Optional[int] = None,
        dtype: Optional[Any] = None,
        load_in_4bit: bool = False,
        **kwargs
    ):
        """
        Load a pretrained Vision Language Model.

        Args:
            model_name: Model identifier from HuggingFace Hub
                       (e.g., "mlx-community/Qwen2-VL-2B-Instruct-4bit")
            max_seq_length: Maximum sequence length
            dtype: Data type (auto-selected by MLX)
            load_in_4bit: Whether model is 4-bit quantized
            **kwargs: Additional arguments

        Returns:
            Tuple of (model, processor)

        Example:
            >>> model, processor = FastVisionModel.from_pretrained(
            ...     "mlx-community/Qwen2-VL-2B-Instruct-4bit"
            ... )
        """

        try:
            from mlx_vlm import load as vlm_load
        except ImportError:
            raise ImportError(
                "mlx-vlm package is required for vision models. "
                "Install it with: pip install mlx-vlm"
            )

        print(f"Loading VLM: {model_name}")

        try:
            model, processor = vlm_load(model_name, **kwargs)

            wrapped_model = VLMModelWrapper(
                model=model,
                processor=processor,
                max_seq_length=max_seq_length,
                model_name=model_name,
            )

            return wrapped_model, processor

        except Exception as e:
            raise RuntimeError(
                f"Failed to load VLM '{model_name}'. "
                f"Error: {str(e)}\n\n"
                f"Tips:\n"
                f"- Ensure mlx-vlm is installed: pip install mlx-vlm\n"
                f"- Check model exists on HuggingFace Hub\n"
                f"- For quantized models, use mlx-community versions"
            ) from e

    @staticmethod
    def get_peft_model(
        model: Any,
        r: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        **kwargs
    ) -> Any:
        """
        Add LoRA adapters to the VLM for fine-tuning.

        Args:
            model: The VLM to add LoRA adapters to
            r: LoRA rank
            target_modules: Target modules for LoRA
            lora_alpha: LoRA scaling parameter
            lora_dropout: LoRA dropout
            **kwargs: Additional LoRA configuration

        Returns:
            Model with LoRA configured
        """

        if target_modules is None:
            # Default for vision-language models
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        if hasattr(model, 'configure_lora'):
            model.configure_lora(
                r=r,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                **kwargs
            )
        else:
            warnings.warn(
                "Model does not support LoRA configuration directly. "
                "Use mlx-vlm's fine-tuning utilities instead."
            )

        return model

    @staticmethod
    def for_inference(model: Any) -> Any:
        """Enable inference mode for the VLM."""
        if hasattr(model, 'enable_inference_mode'):
            model.enable_inference_mode()
        return model


class VLMModelWrapper:
    """
    Wrapper around MLX-VLM models providing Unsloth-compatible interface.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        max_seq_length: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.model_name = model_name

        # LoRA config
        self.lora_config = None
        self.lora_enabled = False
        self.inference_mode = False

    def configure_lora(self, **kwargs):
        """Configure LoRA parameters."""
        self.lora_config = kwargs
        self.lora_enabled = True
        print(f"LoRA configured for VLM: {kwargs}")

    def enable_inference_mode(self, use_cache: bool = True):
        """Enable inference mode."""
        self.inference_mode = True
        print("VLM inference mode enabled")

    def generate(
        self,
        prompt: str,
        image: Optional[Any] = None,
        image_path: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate response for image+text input.

        Args:
            prompt: Text prompt
            image: PIL Image or numpy array
            image_path: Path to image file
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation args

        Returns:
            Generated text response
        """

        try:
            from mlx_vlm import generate as vlm_generate
        except ImportError:
            raise ImportError("mlx-vlm required for generation")

        # Load image if path provided
        if image_path and image is None:
            from PIL import Image
            image = Image.open(image_path)

        return vlm_generate(
            self.model,
            self.processor,
            prompt=prompt,
            image=image,
            max_tokens=max_tokens,
            temp=temperature,
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


class VLMSFTTrainer:
    """
    Supervised Fine-Tuning Trainer for Vision Language Models.

    Fine-tune VLMs on image-text datasets using LoRA.

    Example:
        >>> from unsloth_mlx import FastVisionModel, VLMSFTTrainer
        >>>
        >>> model, processor = FastVisionModel.from_pretrained(
        ...     "mlx-community/Qwen2-VL-2B-Instruct-4bit"
        ... )
        >>> model = FastVisionModel.get_peft_model(model, r=16)
        >>>
        >>> # Image-text dataset
        >>> dataset = [
        ...     {"image": "path/to/image.jpg", "conversations": [...]},
        ... ]
        >>>
        >>> trainer = VLMSFTTrainer(
        ...     model=model,
        ...     processor=processor,
        ...     train_dataset=dataset,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        train_dataset: Any,
        output_dir: str = "./vlm_outputs",
        learning_rate: float = 2e-4,
        num_train_epochs: int = 1,
        batch_size: int = 1,
        **kwargs
    ):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"VLMSFTTrainer initialized:")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs: {self.num_train_epochs}")

    def train(self):
        """
        Train the VLM.

        Note: Uses mlx-vlm's fine-tuning capabilities.
        """

        print("=" * 70)
        print("Starting VLM Fine-Tuning")
        print("=" * 70)

        warnings.warn(
            "VLM fine-tuning requires mlx-vlm's training utilities. "
            "For full VLM training, use: mlx-vlm fine-tune command. "
            "See: https://github.com/Blaizzy/mlx-vlm",
            UserWarning
        )

        # Prepare data
        data_file = self.output_dir / "train.jsonl"
        with open(data_file, 'w') as f:
            import json
            for sample in self.train_dataset:
                f.write(json.dumps(sample) + '\n')

        print(f"Training data saved to: {data_file}")
        print("\nTo train, run:")
        print(f"  mlx_vlm.fine_tune --model {self.model.model_name} --data {data_file}")


def load_vlm_dataset(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    image_column: str = "image",
    text_column: str = "text",
) -> List[Dict]:
    """
    Load and prepare a VLM dataset.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_path: Local dataset path
        image_column: Column name for images
        text_column: Column name for text

    Returns:
        List of formatted samples
    """

    if dataset_name:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name)
    elif dataset_path:
        import json
        with open(dataset_path) as f:
            dataset = [json.loads(line) for line in f]
    else:
        raise ValueError("Provide dataset_name or dataset_path")

    return dataset
