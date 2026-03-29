"""Set Transformer — SectorSetRanker for cross-stock ranking within sector cohorts.

Architecture: stock features → linear projection → N × SetAttentionBlock → per-stock score.
Each stock's score is informed by all other stocks in its sector set (cross-stock attention).

Reference: Lee et al. (2019) "Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core attention primitives
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional padding mask."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: [B, N, d_model]
            key_mask: [B, N] bool — True = real, False = padding.
                      Padding positions in key are masked to -inf before softmax.
        Returns:
            [B, N, d_model]
        """
        B, N, _ = query.shape

        Q = self.W_q(query).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, N, N]

        if key_mask is not None:
            # Expand mask to [B, 1, 1, N] — masks key positions (columns)
            mask = key_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # If an entire key sequence is padding, softmax produces NaN — replace with 0
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)                              # [B, H, N, d_k]
        context = context.transpose(1, 2).contiguous().view(B, N, -1)  # [B, N, d_model]
        return self.W_o(context)


class SetAttentionBlock(nn.Module):
    """SAB: permutation-equivariant self-attention block with residual + LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    [B, N, d_model]
            mask: [B, N] bool — True = real stock, False = padding
        Returns:
            [B, N, d_model] — padding positions are updated but unused (masked in loss)
        """
        x = self.norm1(x + self.attn(x, x, x, key_mask=mask))
        x = self.norm2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Top-level ranking model
# ---------------------------------------------------------------------------

class SectorSetRanker(nn.Module):
    """
    Scores each stock relative to its sector cohort.

    Input:  [B, N, D]  — batch of sector sets, padded to N with mask
    Output: [B, N]     — score per stock (real stocks only; padding is garbage)

    The key property: each stock's score is conditioned on all other stocks in
    its sector set via cross-stock self-attention — the model can detect outliers.
    """

    def __init__(
        self,
        dim_input: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(dim_input, d_model),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([
            SetAttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        # Per-stock output: residual path from projected input + attended features
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    [B, N, D]  raw stock features (preprocessed, padded)
            mask: [B, N]     bool — True = real stock
        Returns:
            scores: [B, N]   per-stock ranking score
        """
        h = self.input_proj(x)           # [B, N, d_model]
        residual = h                     # skip connection from before attention

        for block in self.blocks:
            h = block(h, mask)           # [B, N, d_model]

        # Concatenate pre-attention and post-attention representations
        out = self.output_head(torch.cat([residual, h], dim=-1))  # [B, N, 1]
        return out.squeeze(-1)           # [B, N]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
