"""Uncertainty Estimation Branch (UEB) from UnSFlow.

The implementation is extracted from the active server training code in
``models/block/BHTN.py`` and renamed to match the terminology in the paper.
"""

import torch
import torch.nn as nn


class MonteCarloDropoutUncertaintyEstimator(nn.Module):
    """Estimate temporal response mean, variance, and entropy with MC Dropout."""

    def __init__(self, hidden_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=1),
            nn.Softmax(dim=1),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(
        self,
        x: torch.Tensor,
        mc_samples: int = 5,
        force_mc_dropout: bool = False,
    ):
        if mc_samples <= 1:
            mean_map = self.forward_once(x)
            variance_map = torch.zeros_like(mean_map)
            entropy_map = self._binary_entropy(mean_map)
            return mean_map, variance_map, entropy_map

        dropout_states = []
        if force_mc_dropout:
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    dropout_states.append((module, module.training))
                    module.train()

        samples = [self.forward_once(x) for _ in range(mc_samples)]

        for module, previous_state in dropout_states:
            module.training = previous_state

        samples = torch.stack(samples, dim=0)
        mean_map = samples.mean(dim=0)
        variance_map = samples.var(dim=0, unbiased=False)
        entropy_map = self._binary_entropy(mean_map)
        return mean_map, variance_map, entropy_map

    @staticmethod
    def _binary_entropy(probability: torch.Tensor, eps: float = 1e-6):
        probability = probability.clamp(min=eps, max=1.0 - eps)
        return -(
            probability * torch.log(probability)
            + (1.0 - probability) * torch.log(1.0 - probability)
        )


class UncertaintyEstimationBranch(nn.Module):
    """UEB: reliability-aware short- and long-term temporal feature modeling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        period: int,
        mc_samples: int = 5,
        dropout_p: float = 0.1,
        beta: float = 0.001,
        mc_inference: bool = False,
        use_uncertainty_suppression: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.mc_samples = mc_samples
        self.dropout_p = dropout_p
        self.beta = beta
        self.mc_inference = mc_inference
        self.use_uncertainty_suppression = use_uncertainty_suppression

        self.short_term_extractor = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=period,
            stride=1,
            padding=period // 2,
            groups=input_dim,
        )
        self.long_term_extractor = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=seq_len,
            stride=1,
            padding=seq_len // 2,
            groups=input_dim,
        )

        self.uncertainty_branch_short = MonteCarloDropoutUncertaintyEstimator(
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
        )
        self.uncertainty_branch_long = MonteCarloDropoutUncertaintyEstimator(
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
        )

        self.weak_signal_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )
        self.bi_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.output_layer = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, x: torch.Tensor, return_uncertainty: bool = False):
        """Process input features shaped ``[batch, time, channels]``."""
        x_short = self.short_term_extractor(x.permute(0, 2, 1))
        x_long = self.long_term_extractor(x.permute(0, 2, 1))

        use_mc = self.training or self.mc_inference
        mc_samples = self.mc_samples if use_mc else 1

        short_mean, short_variance, short_entropy = self.uncertainty_branch_short(
            x_short,
            mc_samples=mc_samples,
            force_mc_dropout=use_mc,
        )
        long_mean, long_variance, long_entropy = self.uncertainty_branch_long(
            x_long,
            mc_samples=mc_samples,
            force_mc_dropout=use_mc,
        )

        if self.use_uncertainty_suppression:
            short_uncertainty = short_variance + short_entropy
            long_uncertainty = long_variance + long_entropy
            short_reliability = torch.exp(-short_uncertainty)
            long_reliability = torch.exp(-long_uncertainty)
            x_short = x_short * (1.0 + self.beta * short_reliability)
            x_long = x_long * (1.0 + self.beta * long_reliability)
        else:
            x_short = x_short * (1.0 + self.beta * short_mean)
            x_long = x_long * (1.0 + self.beta * long_mean)

        combined = (x_short + x_long).permute(0, 2, 1)
        gated = self.weak_signal_gate(combined) * combined
        attended, _ = self.attention(gated, gated, gated)
        temporal_features, _ = self.bi_lstm(attended)
        output = self.output_layer(temporal_features)

        if return_uncertainty:
            uncertainty = {
                "short_mean": short_mean,
                "short_var": short_variance,
                "short_entropy": short_entropy,
                "long_mean": long_mean,
                "long_var": long_variance,
                "long_entropy": long_entropy,
            }
            return output, uncertainty
        return output


# Compatibility aliases for the original server implementation.
MCDropoutUncertaintyBranch = MonteCarloDropoutUncertaintyEstimator
HTemporalNet = UncertaintyEstimationBranch


__all__ = [
    "MonteCarloDropoutUncertaintyEstimator",
    "UncertaintyEstimationBranch",
    "MCDropoutUncertaintyBranch",
    "HTemporalNet",
]
