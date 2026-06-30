"""Backbone-preserving confidence-gated EEG residual adapter for ETEA-CGRA.

V1.1 keeps the same conservative text side as V1: text features are only used
to build an emotion context and are never modified. The EEG residual is weaker
by default, starts from zero, and reports shift statistics for regularization.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceGatedEEGResidualAdapter(nn.Module):
    """EEG-side residual refinement with backbone-preserving initialization."""

    def __init__(
        self,
        feature_dim=512,
        num_classes=4,
        num_text_aug=16,
        alpha=0.02,
        hidden_ratio=0.25,
        dropout=0.0,
        gate_type="scalar",
        detach_confidence=True,
        normalize_features=True,
        residual_zero_init=True,
        gate_init_bias=-2.0,
        eps=1e-8,
        debug_shapes=False,
    ):
        super().__init__()
        if gate_type not in {"scalar", "feature"}:
            raise ValueError("gate_type must be 'scalar' or 'feature', got {}".format(gate_type))
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.num_text_aug = int(num_text_aug)
        self.alpha = float(alpha)
        self.gate_type = gate_type
        self.detach_confidence = bool(detach_confidence)
        self.normalize_features = bool(normalize_features)
        self.residual_zero_init = bool(residual_zero_init)
        self.gate_init_bias = float(gate_init_bias)
        self.eps = float(eps)
        self.debug_shapes = bool(debug_shapes)
        self._debug_printed = False

        input_dim = self.feature_dim * 2 + 3
        hidden_dim = max(1, int(self.feature_dim * float(hidden_ratio)))
        gate_dim = 1 if gate_type == "scalar" else self.feature_dim

        self.residual_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.feature_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, gate_dim),
        )
        self._init_backbone_preserving_heads()

    def _init_backbone_preserving_heads(self):
        residual_last = self.residual_mlp[-1]
        if self.residual_zero_init:
            nn.init.zeros_(residual_last.weight)
            nn.init.zeros_(residual_last.bias)

        gate_last = self.gate_mlp[-1]
        if gate_last.bias is not None:
            nn.init.constant_(gate_last.bias, self.gate_init_bias)

    def _normalize(self, x):
        if not self.normalize_features:
            return x
        return F.normalize(x, dim=-1)

    def compute_class_prototypes(self, text_features):
        """Return fixed class prototypes [K, D] from prompt features [A*K, D]."""
        if text_features.dim() != 2:
            raise ValueError("text_features must be [A*K, D], got {}".format(tuple(text_features.shape)))
        expected = self.num_text_aug * self.num_classes
        if text_features.shape[0] != expected:
            raise ValueError(
                "text_features first dim must equal num_text_aug*num_classes={}*{}={}, got {}".format(
                    self.num_text_aug,
                    self.num_classes,
                    expected,
                    text_features.shape[0],
                )
            )
        if text_features.shape[1] != self.feature_dim:
            raise ValueError("feature dim mismatch: expected {}, got {}".format(self.feature_dim, text_features.shape[1]))
        prototypes = text_features.view(self.num_text_aug, self.num_classes, self.feature_dim).mean(dim=0)
        return self._normalize(prototypes)

    def compute_confidence_stats(self, logits0):
        """Return softmax and confidence statistics derived from baseline logits."""
        if logits0.dim() != 2:
            raise ValueError("logits0 must be [B, K], got {}".format(tuple(logits0.shape)))
        if logits0.shape[1] != self.num_classes:
            raise ValueError("logits0 class dim mismatch: expected {}, got {}".format(self.num_classes, logits0.shape[1]))

        logits_for_stats = logits0.detach() if self.detach_confidence else logits0
        q = torch.softmax(logits_for_stats, dim=-1)
        max_prob = q.max(dim=-1, keepdim=True).values
        entropy = -(q * (q + self.eps).log()).sum(dim=-1, keepdim=True)
        entropy_norm = entropy / math.log(max(self.num_classes, 2))
        if self.num_classes >= 2:
            top2 = torch.topk(q, k=2, dim=-1).values
            margin = (top2[:, :1] - top2[:, 1:2]).clamp(min=0.0)
        else:
            margin = torch.zeros_like(max_prob)

        if self.detach_confidence:
            q = q.detach()
            max_prob = max_prob.detach()
            entropy_norm = entropy_norm.detach()
            margin = margin.detach()

        return {
            "q": q,
            "max_prob": max_prob,
            "entropy": entropy_norm,
            "margin": margin,
        }

    def forward(self, x0, text_features, logits0):
        if x0.dim() != 2:
            raise ValueError("x0 must be [B, D], got {}".format(tuple(x0.shape)))
        if x0.shape[1] != self.feature_dim:
            raise ValueError("x0 feature dim mismatch: expected {}, got {}".format(self.feature_dim, x0.shape[1]))

        x0_norm = self._normalize(x0)
        t0 = self.compute_class_prototypes(text_features)
        stats = self.compute_confidence_stats(logits0)
        context = stats["q"] @ t0
        scalar_stats = torch.cat([stats["max_prob"], stats["entropy"], stats["margin"]], dim=-1)
        input_vec = torch.cat([x0_norm, context, scalar_stats.to(dtype=x0_norm.dtype)], dim=-1)

        delta_x = self.residual_mlp(input_vec)
        gate = torch.sigmoid(self.gate_mlp(input_vec))
        x1 = self._normalize(x0_norm + self.alpha * gate * delta_x)
        x_shift_norm = (x1 - x0_norm).norm(dim=-1).mean()

        if self.debug_shapes and not self._debug_printed:
            print(
                "ETEA_CGRA_DEBUG_SHAPES x0={} text_features={} logits0={} context={} gate={} x1={}".format(
                    tuple(x0.shape),
                    tuple(text_features.shape),
                    tuple(logits0.shape),
                    tuple(context.shape),
                    tuple(gate.shape),
                    tuple(x1.shape),
                )
            )
            self._debug_printed = True

        return {
            "x": x1,
            "x0": x0_norm,
            "gate": gate,
            "gate_mean": gate.mean(),
            "delta_norm": delta_x.norm(dim=-1).mean(),
            "x_shift_norm": x_shift_norm,
            "confidence": {
                "max_prob": stats["max_prob"].mean(),
                "entropy": stats["entropy"].mean(),
                "margin": stats["margin"].mean(),
            },
        }
