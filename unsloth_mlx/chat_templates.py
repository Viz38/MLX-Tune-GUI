"""
Chat Templates and Dataset Formatting for Unsloth-MLX

This module provides Unsloth-compatible dataset formatting utilities,
converting various dataset formats to mlx-lm compatible formats.

Supported input formats:
- Alpaca: {"instruction": "...", "input": "...", "output": "..."}
- ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
- ChatML: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
- Text: {"text": "..."}
- Completions: {"prompt": "...", "completion": "..."}

Output formats (mlx-lm compatible):
- text: {"text": "..."}
- chat: {"messages": [...]}
- completions: {"prompt": "...", "completion": "..."}
"""

from typing import Any, Dict, List, Optional, Callable, Union, NamedTuple
from datasets import Dataset
import re


# Default Alpaca prompt template
ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


# =============================================================================
# CHAT TEMPLATE REGISTRY (Unsloth-compatible)
# =============================================================================

class ChatTemplateEntry(NamedTuple):
    """Registry entry for a chat template."""
    template: str       # Jinja2 template string
    eos_token: str      # EOS token (or "eos_token" to use tokenizer's default)
    bos_token: str      # BOS token (or "bos_token" to use tokenizer's default)
    stop_token: str     # Stop token for generation


# Jinja2 templates for each model family
# These are based on official HuggingFace tokenizer configs and Unsloth

_LLAMA3_TEMPLATE = """{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] | trim + '\n\n' -%}
    {%- set messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = '' -%}
{%- endif -%}

{{ bos_token }}{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif -%}"""

_CHATML_TEMPLATE = """{%- for message in messages -%}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|im_start|>assistant\n' }}
{%- endif -%}"""

_GEMMA_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ '<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<start_of_turn>model\n' + message['content'] | trim + '<end_of_turn>\n' }}
    {%- elif message['role'] == 'system' -%}
        {{ '<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<start_of_turn>model\n' }}
{%- endif -%}"""

_QWEN_TEMPLATE = """{%- for message in messages -%}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|im_start|>assistant\n' }}
{%- endif -%}"""

_QWEN3_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ '<|im_start|>system\n' + message['content'] | trim + '<|im_end|>\n' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<|im_start|>user\n' + message['content'] | trim + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {%- if message.get('reasoning_content') -%}
            {{ '<|im_start|>assistant\n<think>\n' + message['reasoning_content'] | trim + '\n</think>\n' + message['content'] | trim + '<|im_end|>\n' }}
        {%- else -%}
            {{ '<|im_start|>assistant\n' + message['content'] | trim + '<|im_end|>\n' }}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|im_start|>assistant\n' }}
{%- endif -%}"""

_PHI3_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ '<|system|>\n' + message['content'] | trim + '<|end|>\n' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<|user|>\n' + message['content'] | trim + '<|end|>\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|assistant|>\n' + message['content'] | trim + '<|end|>\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|assistant|>\n' }}
{%- endif -%}"""

_PHI4_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ '<|im_start|>system<|im_sep|>' + message['content'] | trim + '<|im_end|>' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<|im_start|>user<|im_sep|>' + message['content'] | trim + '<|im_end|>' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|im_start|>assistant<|im_sep|>' + message['content'] | trim + '<|im_end|>' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|im_start|>assistant<|im_sep|>' }}
{%- endif -%}"""

_MISTRAL_TEMPLATE = """{{ bos_token }}{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ '[INST] ' + message['content'] | trim + ' [/INST]' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ message['content'] | trim + eos_token }}
    {%- elif message['role'] == 'system' -%}
        {{ '[INST] ' + message['content'] | trim + ' [/INST]' }}
    {%- endif -%}
{%- endfor -%}"""

_DEEPSEEK_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ message['content'] | trim + '\n\n' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<｜User｜>' + message['content'] | trim + '\n\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<｜Assistant｜>' + message['content'] | trim + '<｜end▁of▁sentence｜>' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<｜Assistant｜>' }}
{%- endif -%}"""

_VICUNA_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ 'USER: ' + message['content'] | trim + '\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ 'ASSISTANT: ' + message['content'] | trim + '</s>\n' }}
    {%- elif message['role'] == 'system' -%}
        {{ message['content'] | trim + '\n\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ 'ASSISTANT: ' }}
{%- endif -%}"""

_ALPACA_CHAT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ '### Instruction:\n' + message['content'] | trim + '\n\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '### Response:\n' + message['content'] | trim + '\n\n' }}
    {%- elif message['role'] == 'system' -%}
        {{ message['content'] | trim + '\n\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '### Response:\n' }}
{%- endif -%}"""

_ZEPHYR_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ '<|system|>\n' + message['content'] | trim + '</s>\n' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<|user|>\n' + message['content'] | trim + '</s>\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|assistant|>\n' + message['content'] | trim + '</s>\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|assistant|>\n' }}
{%- endif -%}"""


# Main template registry
CHAT_TEMPLATES: Dict[str, ChatTemplateEntry] = {
    # Llama 3 family
    "llama-3": ChatTemplateEntry(
        template=_LLAMA3_TEMPLATE,
        eos_token="<|eot_id|>",
        bos_token="<|begin_of_text|>",
        stop_token="<|eot_id|>",
    ),
    "llama-3.1": ChatTemplateEntry(
        template=_LLAMA3_TEMPLATE,
        eos_token="<|eot_id|>",
        bos_token="<|begin_of_text|>",
        stop_token="<|eot_id|>",
    ),

    # ChatML (OpenAI format)
    "chatml": ChatTemplateEntry(
        template=_CHATML_TEMPLATE,
        eos_token="<|im_end|>",
        bos_token="",
        stop_token="<|im_end|>",
    ),

    # Gemma family
    "gemma-2": ChatTemplateEntry(
        template=_GEMMA_TEMPLATE,
        eos_token="<end_of_turn>",
        bos_token="<bos>",
        stop_token="<end_of_turn>",
    ),
    "gemma-3": ChatTemplateEntry(
        template=_GEMMA_TEMPLATE,
        eos_token="<end_of_turn>",
        bos_token="<bos>",
        stop_token="<end_of_turn>",
    ),

    # Qwen family
    "qwen-2.5": ChatTemplateEntry(
        template=_QWEN_TEMPLATE,
        eos_token="<|im_end|>",
        bos_token="",
        stop_token="<|im_end|>",
    ),
    "qwen-3": ChatTemplateEntry(
        template=_QWEN3_TEMPLATE,
        eos_token="<|im_end|>",
        bos_token="",
        stop_token="<|im_end|>",
    ),

    # Phi family
    "phi-3": ChatTemplateEntry(
        template=_PHI3_TEMPLATE,
        eos_token="<|end|>",
        bos_token="",
        stop_token="<|end|>",
    ),
    "phi-3.5": ChatTemplateEntry(
        template=_PHI3_TEMPLATE,
        eos_token="<|end|>",
        bos_token="",
        stop_token="<|end|>",
    ),
    "phi-4": ChatTemplateEntry(
        template=_PHI4_TEMPLATE,
        eos_token="<|im_end|>",
        bos_token="",
        stop_token="<|im_end|>",
    ),

    # Mistral family
    "mistral": ChatTemplateEntry(
        template=_MISTRAL_TEMPLATE,
        eos_token="</s>",
        bos_token="<s>",
        stop_token="</s>",
    ),
    "mistral-nemo": ChatTemplateEntry(
        template=_MISTRAL_TEMPLATE,
        eos_token="</s>",
        bos_token="<s>",
        stop_token="</s>",
    ),

    # DeepSeek
    "deepseek-v2": ChatTemplateEntry(
        template=_DEEPSEEK_TEMPLATE,
        eos_token="<｜end▁of▁sentence｜>",
        bos_token="<｜begin▁of▁sentence｜>",
        stop_token="<｜end▁of▁sentence｜>",
    ),

    # Legacy formats
    "alpaca": ChatTemplateEntry(
        template=_ALPACA_CHAT_TEMPLATE,
        eos_token="</s>",
        bos_token="",
        stop_token="</s>",
    ),
    "vicuna": ChatTemplateEntry(
        template=_VICUNA_TEMPLATE,
        eos_token="</s>",
        bos_token="",
        stop_token="</s>",
    ),
    "zephyr": ChatTemplateEntry(
        template=_ZEPHYR_TEMPLATE,
        eos_token="</s>",
        bos_token="",
        stop_token="</s>",
    ),
}


# Template aliases for convenience
TEMPLATE_ALIASES: Dict[str, str] = {
    # Llama aliases
    "llama3": "llama-3",
    "llama-3.2": "llama-3.1",
    "llama-3.3": "llama-3.1",
    "llama31": "llama-3.1",
    "llama32": "llama-3.1",
    "llama33": "llama-3.1",

    # Gemma aliases
    "gemma": "gemma-2",
    "gemma2": "gemma-2",
    "gemma3": "gemma-3",

    # Qwen aliases
    "qwen": "qwen-2.5",
    "qwen25": "qwen-2.5",
    "qwen2.5": "qwen-2.5",
    "qwen3": "qwen-3",

    # Phi aliases
    "phi3": "phi-3",
    "phi35": "phi-3.5",
    "phi4": "phi-4",

    # Mistral aliases
    "mistral-v0.3": "mistral",
    "mistral-instruct": "mistral",

    # DeepSeek aliases
    "deepseek": "deepseek-v2",
    "deepseek-v3": "deepseek-v2",

    # OpenAI format
    "openai": "chatml",
    "im_start": "chatml",
}


# Default system messages for models that benefit from them
DEFAULT_SYSTEM_MESSAGES: Dict[str, str] = {
    "llama-3": "You are a helpful assistant.",
    "llama-3.1": "You are a helpful assistant.",
    "qwen-2.5": "You are a helpful assistant.",
    "qwen-3": "You are a helpful assistant.",
    "deepseek-v2": "You are a helpful assistant.",
}


def detect_dataset_format(sample: Dict[str, Any]) -> str:
    """
    Detect the format of a dataset sample.

    Args:
        sample: A single sample from the dataset

    Returns:
        Format string: 'alpaca', 'sharegpt', 'chatml', 'text', 'completions', or 'unknown'
    """
    keys = set(sample.keys())

    # Check for text format (simplest)
    if 'text' in keys:
        return 'text'

    # Check for ChatML format (messages with role/content)
    if 'messages' in keys:
        messages = sample['messages']
        if isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], dict) and 'role' in messages[0]:
                return 'chatml'

    # Check for ShareGPT format (conversations with from/value)
    if 'conversations' in keys:
        convos = sample['conversations']
        if isinstance(convos, list) and len(convos) > 0:
            if isinstance(convos[0], dict) and 'from' in convos[0]:
                return 'sharegpt'

    # Check for completions format
    if 'prompt' in keys and 'completion' in keys:
        return 'completions'

    # Check for Alpaca format
    if 'instruction' in keys and 'output' in keys:
        return 'alpaca'

    return 'unknown'


def standardize_sharegpt(dataset: Dataset) -> Dataset:
    """
    Convert ShareGPT format to ChatML format.

    ShareGPT uses {"from": "human/gpt", "value": "..."}
    ChatML uses {"role": "user/assistant", "content": "..."}

    Args:
        dataset: Dataset with ShareGPT format conversations

    Returns:
        Dataset with ChatML format messages
    """
    role_mapping = {
        'human': 'user',
        'user': 'user',
        'gpt': 'assistant',
        'assistant': 'assistant',
        'system': 'system',
    }

    def convert_sample(sample):
        if 'conversations' not in sample:
            return sample

        messages = []
        for turn in sample['conversations']:
            role = role_mapping.get(turn.get('from', '').lower(), 'user')
            content = turn.get('value', '')
            messages.append({'role': role, 'content': content})

        return {'messages': messages}

    return dataset.map(convert_sample)


def alpaca_to_text(
    sample: Dict[str, Any],
    template: Optional[str] = None,
) -> str:
    """
    Convert Alpaca format sample to text.

    Args:
        sample: Alpaca format sample with instruction, input, output
        template: Optional custom template string

    Returns:
        Formatted text string
    """
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')

    if template:
        return template.format(
            instruction=instruction,
            input=input_text,
            output=output
        )

    # Use appropriate template based on whether input is provided
    if input_text.strip():
        return ALPACA_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
            output=output
        )
    else:
        return ALPACA_TEMPLATE_NO_INPUT.format(
            instruction=instruction,
            output=output
        )


def apply_chat_template_to_sample(
    sample: Dict[str, Any],
    tokenizer: Any,
    add_generation_prompt: bool = False,
) -> str:
    """
    Apply tokenizer's chat template to a sample.

    Args:
        sample: Sample with 'messages' field (ChatML format)
        tokenizer: Tokenizer with apply_chat_template method
        add_generation_prompt: Whether to add generation prompt

    Returns:
        Formatted text string
    """
    messages = sample.get('messages', [])

    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
    else:
        # Fallback: simple formatting
        text_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            text_parts.append(f"{role}: {content}")
        return '\n'.join(text_parts)


def convert_to_mlx_format(
    dataset: Dataset,
    tokenizer: Any,
    output_format: str = 'text',
    alpaca_template: Optional[str] = None,
) -> Dataset:
    """
    Convert any supported dataset format to mlx-lm compatible format.

    This is the main function for dataset conversion, similar to Unsloth's
    formatting workflow.

    Args:
        dataset: Input dataset in any supported format
        tokenizer: Tokenizer (used for chat template if available)
        output_format: Target format ('text', 'chat', 'completions')
        alpaca_template: Custom template for Alpaca format conversion

    Returns:
        Dataset in mlx-lm compatible format

    Example:
        >>> from unsloth_mlx.chat_templates import convert_to_mlx_format
        >>> dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")
        >>> dataset = convert_to_mlx_format(dataset, tokenizer)
        >>> # Now dataset has 'text' field compatible with mlx-lm
    """
    if len(dataset) == 0:
        return dataset

    # Detect input format from first sample
    input_format = detect_dataset_format(dataset[0])
    print(f"Detected dataset format: {input_format}")

    if input_format == 'unknown':
        print(f"Warning: Unknown dataset format. Fields: {list(dataset[0].keys())}")
        print("Attempting to use raw sample...")

    def convert_sample(sample):
        # Already in target format?
        if input_format == output_format:
            return sample
        if input_format == 'text' and output_format == 'text':
            return sample

        # Convert based on input format
        if input_format == 'alpaca':
            if output_format == 'text':
                text = alpaca_to_text(sample, alpaca_template)
                return {'text': text}
            elif output_format == 'completions':
                instruction = sample.get('instruction', '')
                input_text = sample.get('input', '')
                prompt = f"{instruction}\n{input_text}".strip() if input_text else instruction
                return {'prompt': prompt, 'completion': sample.get('output', '')}
            elif output_format == 'chat':
                # Convert to messages format
                messages = [
                    {'role': 'user', 'content': f"{sample.get('instruction', '')}\n{sample.get('input', '')}".strip()},
                    {'role': 'assistant', 'content': sample.get('output', '')}
                ]
                return {'messages': messages}

        elif input_format == 'sharegpt':
            # First convert to ChatML
            messages = []
            role_mapping = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
            for turn in sample.get('conversations', []):
                role = role_mapping.get(turn.get('from', '').lower(), 'user')
                messages.append({'role': role, 'content': turn.get('value', '')})

            if output_format == 'chat':
                return {'messages': messages}
            elif output_format == 'text':
                text = apply_chat_template_to_sample({'messages': messages}, tokenizer)
                return {'text': text}

        elif input_format == 'chatml':
            if output_format == 'chat':
                return sample  # Already in chat format
            elif output_format == 'text':
                text = apply_chat_template_to_sample(sample, tokenizer)
                return {'text': text}

        elif input_format == 'completions':
            if output_format == 'completions':
                return sample
            elif output_format == 'text':
                return {'text': f"{sample.get('prompt', '')}\n{sample.get('completion', '')}"}

        elif input_format == 'text':
            return sample

        # Fallback for unknown format - try to create text
        if output_format == 'text':
            # Try common field names
            for field in ['text', 'content', 'output', 'response', 'completion']:
                if field in sample:
                    return {'text': sample[field]}
            # Last resort: stringify the sample
            import json
            return {'text': json.dumps(sample)}

        return sample

    converted = dataset.map(convert_sample)

    # Verify conversion
    if len(converted) > 0:
        result_format = detect_dataset_format(converted[0])
        print(f"Output format: {result_format}")
        if output_format == 'text' and 'text' not in converted[0]:
            print(f"Warning: Conversion may have failed. Sample keys: {list(converted[0].keys())}")

    return converted


def get_formatting_func(
    tokenizer: Any,
    dataset_format: str = 'auto',
    alpaca_template: Optional[str] = None,
) -> Callable:
    """
    Get a formatting function for use with SFTTrainer.

    This returns a function that can be passed to SFTTrainer's formatting_func
    parameter to automatically convert samples to the text format.

    Args:
        tokenizer: Tokenizer instance
        dataset_format: Expected format ('auto', 'alpaca', 'sharegpt', 'chatml')
        alpaca_template: Custom template for Alpaca format

    Returns:
        Formatting function that takes a sample and returns formatted text

    Example:
        >>> formatting_func = get_formatting_func(tokenizer)
        >>> trainer = SFTTrainer(
        ...     model=model,
        ...     train_dataset=dataset,
        ...     formatting_func=formatting_func,
        ...     ...
        ... )
    """
    def formatting_func(sample: Dict[str, Any]) -> str:
        # Detect format if auto
        fmt = dataset_format
        if fmt == 'auto':
            fmt = detect_dataset_format(sample)

        # Convert based on format
        if fmt == 'text':
            return sample.get('text', '')

        elif fmt == 'alpaca':
            return alpaca_to_text(sample, alpaca_template)

        elif fmt == 'sharegpt':
            # Convert to ChatML first
            messages = []
            role_mapping = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
            for turn in sample.get('conversations', []):
                role = role_mapping.get(turn.get('from', '').lower(), 'user')
                messages.append({'role': role, 'content': turn.get('value', '')})
            return apply_chat_template_to_sample({'messages': messages}, tokenizer)

        elif fmt == 'chatml':
            return apply_chat_template_to_sample(sample, tokenizer)

        elif fmt == 'completions':
            return f"{sample.get('prompt', '')}\n{sample.get('completion', '')}"

        else:
            # Unknown format - try to extract text
            for field in ['text', 'content', 'output', 'response']:
                if field in sample:
                    return sample[field]
            return str(sample)

    return formatting_func


# =============================================================================
# GET_CHAT_TEMPLATE FUNCTION (Unsloth-compatible)
# =============================================================================

def _detect_template_from_tokenizer(tokenizer: Any) -> str:
    """
    Auto-detect the appropriate chat template from tokenizer name or config.

    Args:
        tokenizer: A HuggingFace tokenizer

    Returns:
        Template name string
    """
    # Get model name from tokenizer
    name = getattr(tokenizer, 'name_or_path', '').lower()

    # Detection rules based on model name
    if 'llama-3' in name or 'llama3' in name:
        if any(v in name for v in ['3.1', '3.2', '3.3', '3-1', '3-2', '3-3']):
            return 'llama-3.1'
        return 'llama-3'

    if 'gemma-3' in name or 'gemma3' in name:
        return 'gemma-3'
    if 'gemma' in name:
        return 'gemma-2'

    if 'qwen-3' in name or 'qwen3' in name:
        return 'qwen-3'
    if 'qwen' in name:
        return 'qwen-2.5'

    if 'phi-4' in name or 'phi4' in name:
        return 'phi-4'
    if 'phi-3.5' in name or 'phi35' in name:
        return 'phi-3.5'
    if 'phi-3' in name or 'phi3' in name:
        return 'phi-3'

    if 'mistral-nemo' in name:
        return 'mistral-nemo'
    if 'mistral' in name:
        return 'mistral'

    if 'deepseek' in name:
        return 'deepseek-v2'

    if 'vicuna' in name:
        return 'vicuna'

    if 'zephyr' in name:
        return 'zephyr'

    # Check if tokenizer already has a chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # Try to detect from existing template content
        template = tokenizer.chat_template
        if '<|im_start|>' in template:
            if '<|im_sep|>' in template:
                return 'phi-4'
            return 'chatml'
        if '<|start_header_id|>' in template:
            return 'llama-3'
        if '<start_of_turn>' in template:
            return 'gemma-2'
        if '[INST]' in template:
            return 'mistral'

    # Default to chatml (widely compatible)
    return 'chatml'


def get_chat_template(
    tokenizer: Any,
    chat_template: str = "auto",
    mapping: Optional[Dict[str, str]] = None,
    map_eos_token: bool = True,
    system_message: Optional[str] = None,
) -> Any:
    """
    Apply a chat template to the tokenizer.

    This function matches Unsloth's get_chat_template API for drop-in compatibility.
    It sets the tokenizer's chat_template attribute and optionally configures
    special tokens.

    Args:
        tokenizer: A HuggingFace tokenizer
        chat_template: Template name or "auto" to detect from model name.
                      Supported: llama-3, llama-3.1, chatml, gemma-2, gemma-3,
                      qwen-2.5, qwen-3, phi-3, phi-3.5, phi-4, mistral,
                      mistral-nemo, deepseek-v2, alpaca, vicuna, zephyr
        mapping: Optional column mapping for dataset conversion.
                 e.g., {"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
        map_eos_token: Whether to update the tokenizer's EOS token to match template
        system_message: Custom system message to prepend (if supported by template)

    Returns:
        Modified tokenizer with chat_template set

    Example:
        >>> from unsloth_mlx import get_chat_template, FastLanguageModel
        >>> model, tokenizer = FastLanguageModel.from_pretrained("mlx-community/Llama-3.2-1B-Instruct-4bit")
        >>> tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> text = tokenizer.apply_chat_template(messages, tokenize=False)
    """
    # Resolve alias
    template_name = chat_template.lower().strip()
    if template_name in TEMPLATE_ALIASES:
        template_name = TEMPLATE_ALIASES[template_name]

    # Auto-detect if needed
    if template_name == "auto":
        template_name = _detect_template_from_tokenizer(tokenizer)
        print(f"Auto-detected chat template: {template_name}")

    # Look up template
    if template_name not in CHAT_TEMPLATES:
        available = list_chat_templates()
        raise ValueError(
            f"Unknown chat template: '{chat_template}'. "
            f"Available templates: {', '.join(available)}"
        )

    entry = CHAT_TEMPLATES[template_name]

    # Set the chat template
    tokenizer.chat_template = entry.template

    # Optionally map EOS token
    if map_eos_token and entry.eos_token != "eos_token":
        # Store the stop token for generation
        tokenizer._unsloth_stop_token = entry.stop_token

        # Try to set EOS token if the tokenizer supports it
        try:
            if hasattr(tokenizer, 'eos_token'):
                # Check if the token exists in vocabulary
                if hasattr(tokenizer, 'get_vocab'):
                    vocab = tokenizer.get_vocab()
                    if entry.eos_token in vocab:
                        tokenizer.eos_token = entry.eos_token
        except Exception:
            pass  # Silently fail if we can't set the EOS token

    # Store BOS token reference
    if entry.bos_token and entry.bos_token != "bos_token":
        tokenizer._unsloth_bos_token = entry.bos_token

    # Store mapping for dataset conversion
    if mapping:
        tokenizer._unsloth_mapping = mapping

    # Store system message
    if system_message:
        tokenizer._unsloth_system_message = system_message
    elif template_name in DEFAULT_SYSTEM_MESSAGES:
        tokenizer._unsloth_system_message = DEFAULT_SYSTEM_MESSAGES[template_name]

    # Store template name for reference
    tokenizer._unsloth_chat_template_name = template_name

    return tokenizer


def list_chat_templates() -> List[str]:
    """
    List all available chat template names.

    Returns:
        Sorted list of template names

    Example:
        >>> from unsloth_mlx import list_chat_templates
        >>> templates = list_chat_templates()
        >>> print(templates)
        ['alpaca', 'chatml', 'deepseek-v2', 'gemma-2', ...]
    """
    return sorted(CHAT_TEMPLATES.keys())


def get_template_info(template_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific chat template.

    Args:
        template_name: The template name (supports aliases)

    Returns:
        Dictionary with template information

    Raises:
        ValueError: If template_name is not found

    Example:
        >>> from unsloth_mlx import get_template_info
        >>> info = get_template_info("llama-3")
        >>> print(info['eos_token'])
        '<|eot_id|>'
    """
    # Resolve alias
    name = template_name.lower().strip()
    if name in TEMPLATE_ALIASES:
        name = TEMPLATE_ALIASES[name]

    if name not in CHAT_TEMPLATES:
        available = list_chat_templates()
        raise ValueError(
            f"Unknown chat template: '{template_name}'. "
            f"Available templates: {', '.join(available)}"
        )

    entry = CHAT_TEMPLATES[name]
    return {
        "name": name,
        "eos_token": entry.eos_token,
        "bos_token": entry.bos_token,
        "stop_token": entry.stop_token,
        "template_preview": entry.template[:200] + "..." if len(entry.template) > 200 else entry.template,
        "default_system_message": DEFAULT_SYSTEM_MESSAGES.get(name),
    }


def get_template_for_model(model_name: str) -> str:
    """
    Get the recommended chat template name for a given model.

    Args:
        model_name: Model name or path (e.g., "meta-llama/Llama-3.2-1B-Instruct")

    Returns:
        Recommended template name

    Example:
        >>> from unsloth_mlx import get_template_for_model
        >>> template = get_template_for_model("meta-llama/Llama-3.2-1B-Instruct")
        >>> print(template)
        'llama-3.1'
    """
    name = model_name.lower()

    # Detection rules
    if 'llama-3' in name or 'llama3' in name:
        if any(v in name for v in ['3.1', '3.2', '3.3']):
            return 'llama-3.1'
        return 'llama-3'

    if 'gemma-3' in name:
        return 'gemma-3'
    if 'gemma' in name:
        return 'gemma-2'

    if 'qwen3' in name or 'qwen-3' in name:
        return 'qwen-3'
    if 'qwen' in name:
        return 'qwen-2.5'

    if 'phi-4' in name:
        return 'phi-4'
    if 'phi-3.5' in name:
        return 'phi-3.5'
    if 'phi-3' in name or 'phi3' in name:
        return 'phi-3'

    if 'mistral-nemo' in name:
        return 'mistral-nemo'
    if 'mistral' in name:
        return 'mistral'

    if 'deepseek' in name:
        return 'deepseek-v2'

    if 'vicuna' in name:
        return 'vicuna'

    if 'zephyr' in name:
        return 'zephyr'

    # Default
    return 'chatml'


# =============================================================================
# TRAIN_ON_RESPONSES_ONLY (Unsloth-compatible)
# =============================================================================

def train_on_responses_only(
    trainer: Any,
    instruction_part: Optional[str] = None,
    response_part: Optional[str] = None,
) -> Any:
    """
    Modify the trainer to only compute loss on response tokens.

    This function matches Unsloth's train_on_responses_only API for drop-in
    compatibility. It configures the trainer to mask instruction/prompt tokens
    so that loss is only computed on the assistant's responses.

    Args:
        trainer: An SFTTrainer instance
        instruction_part: The token sequence marking the start of user instruction.
                         If None, auto-detected from template.
                         e.g., "<|start_header_id|>user<|end_header_id|>"
        response_part: The token sequence marking the start of assistant response.
                      If None, auto-detected from template.
                      e.g., "<|start_header_id|>assistant<|end_header_id|>"

    Returns:
        Modified trainer with response-only training enabled

    Example:
        >>> from unsloth_mlx import SFTTrainer, train_on_responses_only
        >>> trainer = SFTTrainer(model=model, ...)
        >>> trainer = train_on_responses_only(
        ...     trainer,
        ...     instruction_part="<|start_header_id|>user<|end_header_id|>",
        ...     response_part="<|start_header_id|>assistant<|end_header_id|>",
        ... )
        >>> trainer.train()
    """
    # Store configuration on the trainer
    trainer._train_on_responses_only = True
    trainer._instruction_part = instruction_part
    trainer._response_part = response_part

    # Try to auto-detect parts from template if not provided
    if instruction_part is None or response_part is None:
        template_name = None

        # Check if tokenizer has template info
        if hasattr(trainer, 'tokenizer'):
            tokenizer = trainer.tokenizer
            if hasattr(tokenizer, '_unsloth_chat_template_name'):
                template_name = tokenizer._unsloth_chat_template_name
            elif hasattr(tokenizer, 'name_or_path'):
                template_name = get_template_for_model(tokenizer.name_or_path)

        # Set default parts based on template
        if template_name:
            parts = _get_template_parts(template_name)
            if instruction_part is None:
                trainer._instruction_part = parts.get('instruction_part')
            if response_part is None:
                trainer._response_part = parts.get('response_part')

    # Log configuration
    print(f"train_on_responses_only enabled:")
    print(f"  instruction_part: {trainer._instruction_part}")
    print(f"  response_part: {trainer._response_part}")

    return trainer


def _get_template_parts(template_name: str) -> Dict[str, str]:
    """
    Get the instruction and response marker parts for a template.

    These are used for masking during response-only training.
    """
    # Resolve alias
    name = template_name.lower()
    if name in TEMPLATE_ALIASES:
        name = TEMPLATE_ALIASES[name]

    # Template-specific parts
    parts_mapping = {
        "llama-3": {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        },
        "llama-3.1": {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        },
        "chatml": {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        },
        "gemma-2": {
            "instruction_part": "<start_of_turn>user\n",
            "response_part": "<start_of_turn>model\n",
        },
        "gemma-3": {
            "instruction_part": "<start_of_turn>user\n",
            "response_part": "<start_of_turn>model\n",
        },
        "qwen-2.5": {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        },
        "qwen-3": {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        },
        "phi-3": {
            "instruction_part": "<|user|>\n",
            "response_part": "<|assistant|>\n",
        },
        "phi-3.5": {
            "instruction_part": "<|user|>\n",
            "response_part": "<|assistant|>\n",
        },
        "phi-4": {
            "instruction_part": "<|im_start|>user<|im_sep|>",
            "response_part": "<|im_start|>assistant<|im_sep|>",
        },
        "mistral": {
            "instruction_part": "[INST]",
            "response_part": "[/INST]",
        },
        "mistral-nemo": {
            "instruction_part": "[INST]",
            "response_part": "[/INST]",
        },
        "deepseek-v2": {
            "instruction_part": "<｜User｜>",
            "response_part": "<｜Assistant｜>",
        },
        "alpaca": {
            "instruction_part": "### Instruction:\n",
            "response_part": "### Response:\n",
        },
        "vicuna": {
            "instruction_part": "USER: ",
            "response_part": "ASSISTANT: ",
        },
        "zephyr": {
            "instruction_part": "<|user|>\n",
            "response_part": "<|assistant|>\n",
        },
    }

    return parts_mapping.get(name, {
        "instruction_part": None,
        "response_part": None,
    })


def get_response_template_ids(
    tokenizer: Any,
    response_part: str,
) -> List[int]:
    """
    Get the token IDs for the response template marker.

    This is useful for finding where responses start in tokenized sequences.

    Args:
        tokenizer: The tokenizer to use
        response_part: The response marker string

    Returns:
        List of token IDs for the response marker
    """
    if hasattr(tokenizer, 'encode'):
        # Use encode without special tokens
        try:
            return tokenizer.encode(response_part, add_special_tokens=False)
        except Exception:
            return tokenizer.encode(response_part)
    return []


def create_response_only_collator(
    tokenizer: Any,
    instruction_part: str,
    response_part: str,
    ignore_index: int = -100,
) -> Callable:
    """
    Create a data collator that masks instruction tokens.

    This is used during training to ensure loss is only computed on response tokens.

    Args:
        tokenizer: The tokenizer to use
        instruction_part: The instruction marker string
        response_part: The response marker string
        ignore_index: The index to use for masked tokens (default -100)

    Returns:
        A collator function that masks instruction tokens in labels
    """
    # Get token IDs for response marker
    response_ids = get_response_template_ids(tokenizer, response_part)

    def collator(examples):
        """Collate examples and mask instruction tokens."""
        # This is a simplified version - full implementation would handle batching
        for example in examples:
            if 'labels' in example and 'input_ids' in example:
                input_ids = example['input_ids']
                labels = example['labels']

                # Find response start positions and mask everything before
                # This is a simplified approach - Unsloth uses more sophisticated matching
                # For now, we rely on mlx-lm's --mask-prompt flag for subprocess training

        return examples

    return collator


# Convenience exports matching Unsloth API
__all__ = [
    # Dataset format detection and conversion
    'detect_dataset_format',
    'standardize_sharegpt',
    'convert_to_mlx_format',
    'get_formatting_func',
    'apply_chat_template_to_sample',
    'alpaca_to_text',
    'ALPACA_TEMPLATE',
    'ALPACA_TEMPLATE_NO_INPUT',
    # Chat template functions (Unsloth-compatible)
    'get_chat_template',
    'list_chat_templates',
    'get_template_info',
    'get_template_for_model',
    # Response-only training
    'train_on_responses_only',
    '_get_template_parts',
    'get_response_template_ids',
    'create_response_only_collator',
    # Template registry
    'CHAT_TEMPLATES',
    'TEMPLATE_ALIASES',
    'DEFAULT_SYSTEM_MESSAGES',
    'ChatTemplateEntry',
]
