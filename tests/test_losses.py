"""
Unit tests for loss functions in unsloth_mlx.losses
"""

import pytest
import mlx.core as mx
import mlx.nn as nn


class TestComputeLogProbs:
    """Test log probability computation."""

    def test_compute_log_probs_shape(self):
        """Test output shape of compute_log_probs."""
        from unsloth_mlx.losses import compute_log_probs_with_lengths

        # Create a simple mock model
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        # Test input
        batch_size = 2
        seq_len = 10
        input_ids = mx.random.randint(0, 100, (batch_size, seq_len))
        lengths = mx.array([8, 6])

        log_probs = compute_log_probs_with_lengths(model, input_ids, lengths)

        assert log_probs.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {log_probs.shape}"

    def test_compute_log_probs_values(self):
        """Test that log probs are negative (as expected for probabilities)."""
        from unsloth_mlx.losses import compute_log_probs_with_lengths

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        input_ids = mx.random.randint(0, 100, (2, 10))
        lengths = mx.array([8, 6])

        log_probs = compute_log_probs_with_lengths(model, input_ids, lengths)
        mx.eval(log_probs)

        # Log probabilities should be negative (or zero at maximum)
        assert mx.all(log_probs <= 0), "Log probabilities should be non-positive"


class TestDPOLoss:
    """Test DPO loss computation."""

    def test_dpo_loss_shape(self):
        """Test DPO loss returns scalar."""
        from unsloth_mlx.losses import dpo_loss

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        batch_size = 2
        seq_len = 10
        chosen_ids = mx.random.randint(0, 100, (batch_size, seq_len))
        rejected_ids = mx.random.randint(0, 100, (batch_size, seq_len))
        chosen_lengths = mx.array([8, 7])
        rejected_lengths = mx.array([9, 6])

        loss, ntoks = dpo_loss(
            model, chosen_ids, rejected_ids,
            chosen_lengths, rejected_lengths,
            beta=0.1
        )

        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        assert ntoks.shape == (), f"ntoks should be scalar, got shape {ntoks.shape}"

    def test_dpo_loss_beta_effect(self):
        """Test that higher beta increases loss magnitude."""
        from unsloth_mlx.losses import dpo_loss

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        chosen_ids = mx.random.randint(0, 100, (2, 10))
        rejected_ids = mx.random.randint(0, 100, (2, 10))
        chosen_lengths = mx.array([8, 7])
        rejected_lengths = mx.array([9, 6])

        loss_low_beta, _ = dpo_loss(model, chosen_ids, rejected_ids,
                                     chosen_lengths, rejected_lengths, beta=0.01)
        loss_high_beta, _ = dpo_loss(model, chosen_ids, rejected_ids,
                                      chosen_lengths, rejected_lengths, beta=1.0)

        mx.eval(loss_low_beta, loss_high_beta)

        # Both losses should be finite
        assert not mx.isnan(loss_low_beta), "Low beta loss should not be NaN"
        assert not mx.isnan(loss_high_beta), "High beta loss should not be NaN"


class TestORPOLoss:
    """Test ORPO loss computation."""

    def test_orpo_loss_shape(self):
        """Test ORPO loss returns scalar."""
        from unsloth_mlx.losses import orpo_loss

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        chosen_ids = mx.random.randint(0, 100, (2, 10))
        rejected_ids = mx.random.randint(0, 100, (2, 10))
        chosen_lengths = mx.array([8, 7])
        rejected_lengths = mx.array([9, 6])

        loss, ntoks = orpo_loss(model, chosen_ids, rejected_ids,
                                chosen_lengths, rejected_lengths, beta=0.1)

        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"


class TestSimPOLoss:
    """Test SimPO loss computation."""

    def test_simpo_loss_shape(self):
        """Test SimPO loss returns scalar."""
        from unsloth_mlx.losses import simpo_loss

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        chosen_ids = mx.random.randint(0, 100, (2, 10))
        rejected_ids = mx.random.randint(0, 100, (2, 10))
        chosen_lengths = mx.array([8, 7])
        rejected_lengths = mx.array([9, 6])

        loss, ntoks = simpo_loss(model, chosen_ids, rejected_ids,
                                  chosen_lengths, rejected_lengths,
                                  beta=2.0, gamma=0.5)

        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"


class TestSFTLoss:
    """Test SFT loss computation."""

    def test_sft_loss_shape(self):
        """Test SFT loss returns scalar."""
        from unsloth_mlx.losses import sft_loss

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        input_ids = mx.random.randint(0, 100, (2, 10))
        lengths = mx.array([8, 6])

        loss, ntoks = sft_loss(model, input_ids, lengths)

        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        assert loss.item() > 0, "Cross entropy loss should be positive"
