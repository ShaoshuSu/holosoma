"""Vision Transformer encoder for fusing CNN perception with state observations.

Adapted from FAR-FALCON-Terrains ``humanoidverse.agents.modules.vision_transformer``.
"""

from __future__ import annotations

import torch
from torch import nn


class SelfAttentionBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_prob: float = 0.0,
        use_ffn: bool = True,
        ffn_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.use_ffn = use_ffn

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        if use_ffn:
            if ffn_hidden_dim is None:
                ffn_hidden_dim = embed_dim * 4
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(ffn_hidden_dim, embed_dim),
                nn.Dropout(dropout_prob),
            )
            self.norm2 = nn.LayerNorm(embed_dim)
        else:
            self.ffn = None
            self.norm2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed_x = self.norm1(x)
        attn_output, _ = self.self_attention(
            normed_x, normed_x, normed_x, need_weights=False
        )
        x = x + attn_output

        if self.use_ffn and self.ffn is not None:
            x = x + self.ffn(self.norm2(x))

        return x


class CrossAttentionBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_prob: float = 0.0,
        use_ffn: bool = True,
        ffn_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.use_ffn = use_ffn

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        if use_ffn:
            if ffn_hidden_dim is None:
                ffn_hidden_dim = embed_dim * 4
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(ffn_hidden_dim, embed_dim),
                nn.Dropout(dropout_prob),
            )
            self.norm2 = nn.LayerNorm(embed_dim)
        else:
            self.ffn = None
            self.norm2 = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        normed_query = self.norm1(query)
        attn_output, _ = self.cross_attention(
            normed_query, key, value, need_weights=False
        )
        x = query + attn_output

        if self.use_ffn and self.ffn is not None:
            x = x + self.ffn(self.norm2(x))

        return x


class VisionTransformerEncoder(nn.Module):
    """Fuses CNN perception features with state observations via transformer attention.

    CNN features are projected to tokens; state observations (across history) become
    query tokens.  Supports self-attention (concatenated tokens) or cross-attention
    (state queries attend to vision keys/values).

    Output shape: ``[B, embed_dim]``.
    """

    def __init__(
        self,
        cnn_encoder: nn.Module,
        obs_dim: int,
        history_length: int,
        embed_dim: int,
        num_heads: int,
        layer_config,
    ):
        super().__init__()
        self.history_length = history_length
        self.embed_dim = embed_dim
        self.output_size = embed_dim

        self.cnn_encoder = cnn_encoder

        # Derive vision token shape from a dummy forward pass.
        # Temporarily disable vision mask so the dummy pass doesn't require v_cmd.
        with torch.no_grad():
            saved_mask = getattr(cnn_encoder, "vision_mask", None)
            if saved_mask is not None:
                cnn_encoder.vision_mask = None
            dummy = torch.zeros(1, cnn_encoder.expected_input_size)
            dummy_out = cnn_encoder(dummy)  # [1, num_tokens, feat_dim]
            self.num_vision_tokens = dummy_out.shape[1]
            self.vision_feat_dim = dummy_out.shape[2]
            if saved_mask is not None:
                cnn_encoder.vision_mask = saved_mask

        self.obs_proj = nn.Linear(obs_dim // history_length, embed_dim)
        self.vision_proj = nn.Linear(self.vision_feat_dim, embed_dim)

        self.use_pos_encoding = layer_config.use_pos_encoding
        if self.use_pos_encoding:
            self.state_pos_encoding = nn.Parameter(
                torch.randn(1, history_length, embed_dim) * 0.02
            )
            self.vision_pos_encoding = nn.Parameter(
                torch.randn(1, self.num_vision_tokens, embed_dim) * 0.02
            )

        self.attention_type = layer_config.attention_type
        num_blocks = layer_config.num_blocks
        use_ffn = layer_config.use_ffn
        ffn_hidden_dim = layer_config.ffn_hidden_dim
        dropout_prob = layer_config.dropout_prob

        if self.attention_type == "self":
            self.transformer_blocks = nn.ModuleList([
                SelfAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                    use_ffn=use_ffn,
                    ffn_hidden_dim=ffn_hidden_dim,
                )
                for _ in range(num_blocks)
            ])
        elif self.attention_type == "cross":
            self.transformer_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                    use_ffn=use_ffn,
                    ffn_hidden_dim=ffn_hidden_dim,
                )
                for _ in range(num_blocks)
            ])
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}. Must be 'self' or 'cross'.")

        self.use_final_ln = layer_config.use_final_ln
        if self.use_final_ln:
            self.final_ln = nn.LayerNorm(embed_dim)
        else:
            self.final_ln = None

        self.aggregation_method = layer_config.aggregation_method
        if self.aggregation_method == "attention":
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=1,
                batch_first=True,
            )
            self.query_vector = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.use_vision_mask = (
            hasattr(cnn_encoder, "vision_mask") and cnn_encoder.vision_mask is not None
        )

    def forward(
        self,
        perception_obs: torch.Tensor,
        obs: torch.Tensor,
        v_cmd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        perception_obs : torch.Tensor
            Flattened vision observations ``[B, C*H*W]``.
        obs : torch.Tensor
            State observations ``[B, history_length * D_obs]``.
        v_cmd : torch.Tensor | None
            Velocity command ``[B, D]`` for vision masking (only used when
            the CNN encoder has a vision mask).

        Returns
        -------
        torch.Tensor
            Fused features ``[B, embed_dim]``.
        """
        batch_size = obs.shape[0]

        # CNN encoder: [B, num_tokens, feat_dim]
        if self.use_vision_mask:
            perception_x = self.vision_proj(self.cnn_encoder(perception_obs, v_cmd=v_cmd))
        else:
            perception_x = self.vision_proj(self.cnn_encoder(perception_obs))  # [B, num_tokens, embed_dim]

        if self.use_pos_encoding:
            perception_x = perception_x + self.vision_pos_encoding

        obs = obs.view(batch_size, self.history_length, -1)  # [B, T, D_obs]
        obs = self.obs_proj(obs)  # [B, T, embed_dim]

        if self.use_pos_encoding:
            obs = obs + self.state_pos_encoding

        if self.attention_type == "self":
            combined = torch.cat([perception_x, obs], dim=1)  # [B, V+T, embed_dim]
            for block in self.transformer_blocks:
                combined = block(combined)
            x = combined
        elif self.attention_type == "cross":
            x = obs
            for block in self.transformer_blocks:
                x = block(query=x, key=perception_x, value=perception_x)

        if self.final_ln is not None:
            x = self.final_ln(x)

        if self.aggregation_method == "last":
            x = x[:, -1, :]
        elif self.aggregation_method == "mean":
            x = x.mean(dim=1)
        elif self.aggregation_method == "max":
            x = x.max(dim=1)[0]
        elif self.aggregation_method == "attention":
            query = self.query_vector.expand(batch_size, -1, -1)
            x, _ = self.attention_pool(query, x, x)
            x = x.squeeze(1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        return x
