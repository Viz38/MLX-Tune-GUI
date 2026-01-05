"""
Loss functions for Unsloth-MLX RL training.

Provides proper loss implementations for:
- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- KTO (Kahneman-Tversky Optimization)
- SimPO (Simple Preference Optimization)
"""

from typing import Optional, Tuple, Callable, List, Any
import mlx.core as mx
import mlx.nn as nn


def compute_log_probs(
    model: Any,
    input_ids: mx.array,
    attention_mask: Optional[mx.array] = None,
) -> mx.array:
    """
    Compute per-token log probabilities for a batch of sequences.

    Args:
        model: The language model.
        input_ids: Token IDs of shape [batch_size, seq_len].
        attention_mask: Optional mask of shape [batch_size, seq_len].

    Returns:
        Log probabilities of shape [batch_size] (sum over sequence).
    """
    # Get inputs (all tokens except last) and targets (all tokens except first)
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    # Forward pass to get logits
    logits = model(inputs)  # [batch_size, seq_len-1, vocab_size]

    # Compute log softmax to get log probabilities
    log_probs = nn.log_softmax(logits, axis=-1)  # [batch_size, seq_len-1, vocab_size]

    # Gather log probs for the actual target tokens
    # targets: [batch_size, seq_len-1]
    # We need to get log_probs[b, t, targets[b, t]] for each position
    batch_size, seq_len = targets.shape

    # Use advanced indexing to gather target log probs
    target_log_probs = mx.take_along_axis(
        log_probs,
        targets[:, :, None],  # [batch_size, seq_len-1, 1]
        axis=-1
    ).squeeze(-1)  # [batch_size, seq_len-1]

    # Apply attention mask if provided
    if attention_mask is not None:
        # Shift mask to match targets
        mask = attention_mask[:, 1:]
        target_log_probs = target_log_probs * mask

    # Sum log probs over sequence to get sequence log probability
    sequence_log_probs = target_log_probs.sum(axis=-1)  # [batch_size]

    return sequence_log_probs


def compute_log_probs_with_lengths(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
) -> mx.array:
    """
    Compute per-token log probabilities with explicit length masking.

    Args:
        model: The language model.
        input_ids: Token IDs of shape [batch_size, seq_len].
        lengths: Sequence lengths of shape [batch_size].

    Returns:
        Log probabilities of shape [batch_size] (sum over valid tokens).
    """
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits = model(inputs)
    log_probs = nn.log_softmax(logits, axis=-1)

    target_log_probs = mx.take_along_axis(
        log_probs,
        targets[:, :, None],
        axis=-1
    ).squeeze(-1)

    # Create mask from lengths
    seq_len = targets.shape[1]
    positions = mx.arange(seq_len)[None, :]  # [1, seq_len]
    mask = positions < lengths[:, None]  # [batch_size, seq_len]

    # Apply mask and sum
    masked_log_probs = target_log_probs * mask.astype(target_log_probs.dtype)
    sequence_log_probs = masked_log_probs.sum(axis=-1)

    return sequence_log_probs


def dpo_loss(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
    beta: float = 0.1,
    reference_chosen_logprobs: Optional[mx.array] = None,
    reference_rejected_logprobs: Optional[mx.array] = None,
    label_smoothing: float = 0.0,
) -> Tuple[mx.array, mx.array]:
    """
    Compute DPO (Direct Preference Optimization) loss.

    DPO Loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    Where:
        log_ratio = log_pi(y|x) - log_ref(y|x)

    Args:
        model: The policy model being trained.
        chosen_ids: Token IDs for chosen responses [batch_size, seq_len].
        rejected_ids: Token IDs for rejected responses [batch_size, seq_len].
        chosen_lengths: Lengths of chosen sequences [batch_size].
        rejected_lengths: Lengths of rejected sequences [batch_size].
        beta: KL penalty coefficient (temperature).
        reference_chosen_logprobs: Pre-computed reference log probs for chosen.
        reference_rejected_logprobs: Pre-computed reference log probs for rejected.
        label_smoothing: Label smoothing coefficient.

    Returns:
        Tuple of (loss, num_tokens).
    """
    # Compute policy model log probabilities
    log_pi_chosen = compute_log_probs_with_lengths(model, chosen_ids, chosen_lengths)
    log_pi_rejected = compute_log_probs_with_lengths(model, rejected_ids, rejected_lengths)

    # Handle reference model log probabilities
    if reference_chosen_logprobs is None or reference_rejected_logprobs is None:
        # Use current model with stop_gradient as reference (memory efficient)
        log_ref_chosen = mx.stop_gradient(log_pi_chosen)
        log_ref_rejected = mx.stop_gradient(log_pi_rejected)
    else:
        log_ref_chosen = reference_chosen_logprobs
        log_ref_rejected = reference_rejected_logprobs

    # Compute log ratios
    log_ratio_chosen = log_pi_chosen - log_ref_chosen
    log_ratio_rejected = log_pi_rejected - log_ref_rejected

    # DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    logits = beta * (log_ratio_chosen - log_ratio_rejected)

    if label_smoothing > 0:
        # Smooth the labels
        losses = (
            -nn.log_sigmoid(logits) * (1 - label_smoothing)
            - nn.log_sigmoid(-logits) * label_smoothing
        )
    else:
        losses = -nn.log_sigmoid(logits)

    loss = mx.mean(losses)
    ntoks = chosen_lengths.sum() + rejected_lengths.sum()

    return loss, ntoks


def orpo_loss(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
    beta: float = 0.1,
) -> Tuple[mx.array, mx.array]:
    """
    Compute ORPO (Odds Ratio Preference Optimization) loss.

    ORPO combines SFT loss with odds ratio preference loss:
        L = L_SFT + beta * L_OR

    Where:
        L_SFT = -log P(chosen)
        L_OR = -log(sigmoid(log(odds_ratio)))
        odds_ratio = P(chosen) / P(rejected)

    Args:
        model: The model being trained.
        chosen_ids: Token IDs for chosen responses.
        rejected_ids: Token IDs for rejected responses.
        chosen_lengths: Lengths of chosen sequences.
        rejected_lengths: Lengths of rejected sequences.
        beta: Weight for odds ratio loss.

    Returns:
        Tuple of (loss, num_tokens).
    """
    # Compute log probabilities
    log_pi_chosen = compute_log_probs_with_lengths(model, chosen_ids, chosen_lengths)
    log_pi_rejected = compute_log_probs_with_lengths(model, rejected_ids, rejected_lengths)

    # SFT loss on chosen (negative log likelihood)
    # Normalize by length for fair comparison
    avg_log_pi_chosen = log_pi_chosen / chosen_lengths.astype(log_pi_chosen.dtype)
    sft_loss = -mx.mean(avg_log_pi_chosen)

    # Odds ratio loss
    # log(odds_ratio) = log(P_chosen) - log(P_rejected)
    log_odds = log_pi_chosen - log_pi_rejected
    or_loss = -mx.mean(nn.log_sigmoid(log_odds))

    # Combined loss
    loss = sft_loss + beta * or_loss

    ntoks = chosen_lengths.sum() + rejected_lengths.sum()
    return loss, ntoks


def kto_loss(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
    labels: mx.array,  # 1 for positive, 0 for negative
    beta: float = 0.1,
    reference_logprobs: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Compute KTO (Kahneman-Tversky Optimization) loss.

    KTO uses prospect theory with asymmetric treatment of gains and losses:
        L = -E[w(y) * log(sigmoid(beta * log_ratio))]

    Where w(y) = lambda if y is positive, 1 if y is negative.

    Args:
        model: The model being trained.
        input_ids: Token IDs [batch_size, seq_len].
        lengths: Sequence lengths [batch_size].
        labels: Binary labels (1=positive, 0=negative) [batch_size].
        beta: Temperature coefficient.
        reference_logprobs: Pre-computed reference log probs.

    Returns:
        Tuple of (loss, num_tokens).
    """
    # Compute policy log probs
    log_pi = compute_log_probs_with_lengths(model, input_ids, lengths)

    # Handle reference
    if reference_logprobs is None:
        log_ref = mx.stop_gradient(log_pi)
    else:
        log_ref = reference_logprobs

    log_ratio = log_pi - log_ref

    # KTO weights (lambda for positive, 1 for negative)
    lambda_weight = 1.0  # Can be tuned
    weights = mx.where(labels > 0.5, lambda_weight, 1.0)

    # Loss with asymmetric weights
    positive_mask = labels > 0.5
    negative_mask = ~positive_mask

    positive_loss = -nn.log_sigmoid(beta * log_ratio) * positive_mask
    negative_loss = -nn.log_sigmoid(-beta * log_ratio) * negative_mask

    loss = mx.mean(weights * (positive_loss + negative_loss))
    ntoks = lengths.sum()

    return loss, ntoks


def simpo_loss(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> Tuple[mx.array, mx.array]:
    """
    Compute SimPO (Simple Preference Optimization) loss.

    SimPO simplifies DPO by removing the need for a reference model:
        L = -log(sigmoid(beta * (r_chosen - r_rejected - gamma)))

    Where r = log P(y|x) / |y| (length-normalized log prob).

    Args:
        model: The model being trained.
        chosen_ids: Token IDs for chosen responses.
        rejected_ids: Token IDs for rejected responses.
        chosen_lengths: Lengths of chosen sequences.
        rejected_lengths: Lengths of rejected sequences.
        beta: Temperature coefficient.
        gamma: Target reward margin.

    Returns:
        Tuple of (loss, num_tokens).
    """
    # Compute log probabilities
    log_pi_chosen = compute_log_probs_with_lengths(model, chosen_ids, chosen_lengths)
    log_pi_rejected = compute_log_probs_with_lengths(model, rejected_ids, rejected_lengths)

    # Length-normalize to get "reward"
    r_chosen = log_pi_chosen / chosen_lengths.astype(log_pi_chosen.dtype)
    r_rejected = log_pi_rejected / rejected_lengths.astype(log_pi_rejected.dtype)

    # SimPO loss
    logits = beta * (r_chosen - r_rejected - gamma)
    loss = -mx.mean(nn.log_sigmoid(logits))

    ntoks = chosen_lengths.sum() + rejected_lengths.sum()
    return loss, ntoks


def sft_loss(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
) -> Tuple[mx.array, mx.array]:
    """
    Standard Supervised Fine-Tuning (cross-entropy) loss.

    Args:
        model: The model being trained.
        input_ids: Token IDs [batch_size, seq_len].
        lengths: Sequence lengths [batch_size].

    Returns:
        Tuple of (loss, num_tokens).
    """
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits = model(inputs)

    # Create length mask
    seq_len = targets.shape[1]
    positions = mx.arange(seq_len)[None, :]
    mask = positions < lengths[:, None]

    # Cross entropy loss
    ce = nn.losses.cross_entropy(logits, targets, reduction='none')
    masked_ce = ce * mask.astype(ce.dtype)

    ntoks = mask.sum()
    loss = masked_ce.sum() / ntoks

    return loss, ntoks


# GRPO-specific functions

def generate_with_log_probs(
    model: Any,
    tokenizer: Any,
    prompt_ids: mx.array,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> Tuple[mx.array, mx.array]:
    """
    Generate a completion and return token IDs with their log probabilities.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt_ids: Prompt token IDs [seq_len].
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Tuple of (generated_ids, log_probs) where:
            generated_ids: [prompt_len + gen_len]
            log_probs: [gen_len] log probability of each generated token
    """
    generated_ids = list(prompt_ids.tolist()) if hasattr(prompt_ids, 'tolist') else list(prompt_ids)
    log_probs = []

    # Current sequence
    x = mx.array([generated_ids])

    for _ in range(max_tokens):
        # Get logits for next token
        logits = model(x)[:, -1, :]  # [1, vocab_size]

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
            probs = mx.softmax(logits, axis=-1)
            # Sample from categorical distribution
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
        else:
            # Greedy decoding
            next_token = mx.argmax(logits, axis=-1)

        next_token_id = next_token.item()

        # Get log probability of sampled token
        log_prob = nn.log_softmax(logits, axis=-1)[0, next_token_id]
        log_probs.append(log_prob)

        # Append to sequence
        generated_ids.append(next_token_id)

        # Check for EOS
        if hasattr(tokenizer, 'eos_token_id') and next_token_id == tokenizer.eos_token_id:
            break

        # Update input sequence
        x = mx.array([generated_ids])

    return mx.array(generated_ids), mx.stack(log_probs)


def grpo_loss(
    model: Any,
    tokenizer: Any,
    prompt_ids: mx.array,
    reward_fn: Callable[[str, str], float],
    prompt_text: str,
    num_generations: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 256,
    beta: float = 0.04,
) -> Tuple[mx.array, int]:
    """
    Compute GRPO (Group Relative Policy Optimization) loss for a single prompt.

    GRPO:
    1. Generates multiple completions for each prompt
    2. Computes rewards for each completion
    3. Uses group statistics for advantage estimation
    4. Computes policy gradient loss

    Args:
        model: The policy model.
        tokenizer: The tokenizer.
        prompt_ids: Prompt token IDs.
        reward_fn: Function(completion, prompt) -> reward.
        prompt_text: Original prompt text for reward computation.
        num_generations: Number of completions to generate per prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per completion.
        beta: KL penalty coefficient.

    Returns:
        Tuple of (loss, num_completions).
    """
    completions = []
    all_log_probs = []

    # Generate multiple completions
    for _ in range(num_generations):
        gen_ids, log_probs = generate_with_log_probs(
            model, tokenizer, prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Decode completion (skip prompt)
        prompt_len = len(prompt_ids)
        completion_ids = gen_ids[prompt_len:]
        completion_text = tokenizer.decode(completion_ids.tolist())

        completions.append(completion_text)
        all_log_probs.append(log_probs.sum())  # Sum log probs

    # Compute rewards
    rewards = []
    for completion in completions:
        reward = reward_fn(completion, prompt_text)
        rewards.append(reward)

    rewards = mx.array(rewards)
    log_probs_tensor = mx.stack(all_log_probs)

    # Compute advantages using group statistics
    mean_reward = mx.mean(rewards)
    std_reward = mx.std(rewards) + 1e-8
    advantages = (rewards - mean_reward) / std_reward

    # Policy gradient loss: -E[advantage * log_prob]
    # We want to increase prob of high-advantage completions
    pg_loss = -mx.mean(advantages * log_probs_tensor)

    return pg_loss, num_generations


def grpo_batch_loss(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    reward_fn: Callable[[str, str], float],
    num_generations: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 256,
    beta: float = 0.04,
) -> Tuple[mx.array, int]:
    """
    Compute GRPO loss for a batch of prompts.

    Args:
        model: The policy model.
        tokenizer: The tokenizer.
        prompts: List of prompt strings.
        reward_fn: Reward function.
        num_generations: Completions per prompt.
        temperature: Sampling temperature.
        max_tokens: Max tokens per completion.
        beta: KL coefficient.

    Returns:
        Tuple of (average_loss, total_completions).
    """
    losses = []
    total_completions = 0

    for prompt in prompts:
        prompt_ids = mx.array(tokenizer.encode(prompt))

        loss, n_comp = grpo_loss(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            reward_fn=reward_fn,
            prompt_text=prompt,
            num_generations=num_generations,
            temperature=temperature,
            max_tokens=max_tokens,
            beta=beta,
        )

        losses.append(loss)
        total_completions += n_comp

    avg_loss = mx.mean(mx.stack(losses))
    return avg_loss, total_completions


# Utility function for batched DPO

def compute_reference_logprobs(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
) -> Tuple[mx.array, mx.array]:
    """
    Compute reference log probabilities (for frozen reference model).

    Call this once before training to get reference logprobs,
    then pass them to dpo_loss to avoid recomputation.

    Args:
        model: The reference model (should be frozen/not updated).
        chosen_ids: Chosen sequence token IDs.
        rejected_ids: Rejected sequence token IDs.
        chosen_lengths: Chosen sequence lengths.
        rejected_lengths: Rejected sequence lengths.

    Returns:
        Tuple of (ref_chosen_logprobs, ref_rejected_logprobs).
    """
    ref_chosen = compute_log_probs_with_lengths(model, chosen_ids, chosen_lengths)
    ref_rejected = compute_log_probs_with_lengths(model, rejected_ids, rejected_lengths)

    return mx.stop_gradient(ref_chosen), mx.stop_gradient(ref_rejected)
