"""Uncertainty Sharpness-Aware Minimization (USAM) from UnSFlow.

The implementation is extracted from the active server training code in
``solver_b2SAM.py`` and renamed to match the terminology in the paper.
"""

from collections import defaultdict

import torch


class UncertaintySharpnessAwareMinimization:
    """USAM optimizer wrapper with predictive-entropy adaptive perturbations."""

    def __init__(
        self,
        optimizer,
        model,
        rho: float = 0.05,
        uncertainty_weight: float = 0.1,
    ):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.uncertainty_weight = uncertainty_weight
        self.state = defaultdict(dict)

    @staticmethod
    def compute_uncertainty(logits: torch.Tensor) -> float:
        """Return mean predictive entropy from unnormalized model logits."""
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(
            probabilities * torch.log(probabilities + 1e-16),
            dim=-1,
        )
        return entropy.mean().item()

    @torch.no_grad()
    def first_step(self, images, logits: torch.Tensor):
        """Apply the uncertainty-adaptive ascent perturbation."""
        del images  # Kept in the signature for compatibility with the training loop.
        uncertainty = self.compute_uncertainty(logits)

        gradients = [
            torch.norm(parameter.grad, p=2)
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        gradient_norm = torch.norm(torch.stack(gradients), p=2) + 1e-16
        dynamic_rho = self.rho + self.uncertainty_weight * uncertainty

        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            perturbation = self.state[parameter].get("eps")
            if perturbation is None:
                perturbation = torch.clone(parameter).detach()
                self.state[parameter]["eps"] = perturbation
            perturbation.copy_(parameter.grad)
            perturbation.mul_(dynamic_rho / gradient_norm)
            parameter.add_(perturbation)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self):
        """Remove the perturbation and perform the base-optimizer update."""
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            parameter.sub_(self.state[parameter]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


# Compatibility alias for the original server implementation.
PredictionUncertaintySAM = UncertaintySharpnessAwareMinimization


__all__ = [
    "UncertaintySharpnessAwareMinimization",
    "PredictionUncertaintySAM",
]
