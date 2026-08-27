"""Uncertainty-Aware Routed Attention (UARA) from UnSFlow.

The implementation is extracted from the active server training code in
``models/block/UAAHTN.py`` and renamed to match the terminology in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import use_fused_attn


class UncertaintyAwareRoutedAttention(nn.Module):
    """UARA with token uncertainty, reliability routing, and shared heads."""

    LOAD_BALANCING_LOSSES = []

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
        shared_head: int = 0,
        routed_head: int = 0,
        head_dim=None,
        mc_samples: int = 5,
        mc_dropout_p: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.shared_head = shared_head
        self.routed_head = routed_head
        self.mc_samples = mc_samples
        self.eps = eps
        self.head_dim = dim // num_heads if head_dim is None else head_dim
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(
            dim,
            (self.head_dim * self.num_heads) * 3,
            bias=qkv_bias,
        )
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.routed_head > 0:
            self.wg = nn.Linear(dim, num_heads - shared_head, bias=False)
            if self.shared_head > 0:
                self.wg_0 = nn.Linear(dim, 2, bias=False)
        if self.shared_head > 0:
            self.wg_1 = nn.Linear(dim, shared_head, bias=False)

        self.uncertainty_branch = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mc_dropout_p),
            nn.Linear(dim // 2, dim // 2),
            nn.Softmax(dim=1),
            nn.Dropout(p=mc_dropout_p),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )
        self.uncertainty_proj = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def _estimate_uncertainty(
        self,
        x_flat: torch.Tensor,
        batch_size: int,
        token_count: int,
    ) -> torch.Tensor:
        if self.training and self.mc_samples > 1:
            responses = torch.stack(
                [self.uncertainty_branch(x_flat) for _ in range(self.mc_samples)],
                dim=0,
            )
            mean_response = responses.mean(dim=0)
            variance_response = responses.var(dim=0, unbiased=True)
            probability = mean_response.clamp(min=self.eps, max=1.0 - self.eps)
            entropy = -probability * torch.log(probability) - (
                1.0 - probability
            ) * torch.log(1.0 - probability)
            uncertainty = self.uncertainty_proj(variance_response + entropy)
        else:
            uncertainty = self.uncertainty_branch(x_flat)
        return uncertainty.reshape(batch_size, token_count, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process token features shaped ``[batch, tokens, channels]``."""
        batch_size, token_count, channels = x.shape
        x_flat = x.reshape(batch_size * token_count, channels)

        uncertainty = self._estimate_uncertainty(
            x_flat,
            batch_size,
            token_count,
        )
        reliability = 1.0 - uncertainty

        if self.routed_head > 0:
            routing_activation = F.softmax(self.wg(x_flat), dim=1)
            routing_score = routing_activation * reliability.reshape(
                batch_size * token_count,
                1,
            )
            num_tokens, num_experts = routing_score.shape
            del num_tokens
            _, indices = torch.topk(routing_score, k=self.routed_head, dim=1)
            mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)
            mask = mask.to(routing_score.dtype)

            if self.training:
                mean_routing_score = routing_score.mean(dim=0)
                mean_selection = mask.float().mean(dim=0)
                auxiliary_loss = (
                    torch.mean(mean_routing_score * mean_selection)
                    * num_experts
                    * num_experts
                )
                UncertaintyAwareRoutedAttention.LOAD_BALANCING_LOSSES.append(
                    auxiliary_loss
                )

            routed_gates = routing_score * mask
            denominator = routed_gates.sum(dim=1, keepdim=True)
            denominator = torch.clamp(
                denominator,
                min=torch.finfo(denominator.dtype).eps,
            )
            routed_gates = routed_gates / denominator
            routed_gates = routed_gates.reshape(batch_size, token_count, -1)
            routed_gates = routed_gates * self.routed_head

        qkv = self.qkv(x).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        attention_reliability = reliability.unsqueeze(1)

        if self.fused_attn:
            output = F.scaled_dot_product_attention(
                query * attention_reliability,
                key,
                value,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            query = query * self.scale * attention_reliability
            attention = query @ key.transpose(-2, -1)
            attention = self.attn_drop(attention.softmax(dim=-1))
            output = attention @ value

        output = output.transpose(1, 2)
        if self.routed_head > 0:
            if self.shared_head > 0:
                shared_gates = F.softmax(self.wg_1(x_flat), dim=1)
                shared_gates = shared_gates.reshape(batch_size, token_count, -1)
                shared_gates = shared_gates * self.shared_head

                balance = F.softmax(self.wg_0(x_flat), dim=1)
                balance = balance.reshape(batch_size, token_count, 2) * 2
                shared_gates = torch.einsum(
                    "bn,bne->bne",
                    balance[:, :, 0],
                    shared_gates,
                )
                routed_gates = torch.einsum(
                    "bn,bne->bne",
                    balance[:, :, 1],
                    routed_gates,
                )
                masked_gates = torch.cat([shared_gates, routed_gates], dim=2)
            else:
                masked_gates = routed_gates
        else:
            if self.shared_head <= 0:
                raise ValueError(
                    "Either routed_head or shared_head must be greater than 0."
                )
            masked_gates = F.softmax(self.wg_1(x_flat), dim=1)
            masked_gates = masked_gates.reshape(batch_size, token_count, -1)
            masked_gates = masked_gates * self.shared_head

        output = torch.einsum("bne,bned->bned", masked_gates, output)
        output = output.reshape(
            batch_size,
            token_count,
            self.head_dim * self.num_heads,
        )
        return self.proj_drop(self.proj(output))


# Compatibility alias for the original server implementation.
BMoHAttention = UncertaintyAwareRoutedAttention


__all__ = ["UncertaintyAwareRoutedAttention", "BMoHAttention"]
