"""Cacheless varlen prefill through FA3 ``flash_attn_varlen`` (AITER on ROCm).

The FA3 side of ``wrappers.py``: the same base, layouts and padding as the
FlashInfer wrappers there, with no workspace and no head-dim padding.
"""

import torch

from mstar.engine.resources.attn.dense import flash_attn_varlen_func
from mstar.engine.resources.attn.ragged.wrappers import (
    RaggedCrossPrefillWrapper,
    RaggedPrefillWrapper,
    _RaggedPrefillBase,
)


def _max_span(cu_seqlens: torch.Tensor) -> int:
    """Longest segment in a host ``cu_seqlens``. Zero if there are no tokens."""
    if cu_seqlens.numel() < 2:
        return 0
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max())


class _DenseRaggedPrefillBase(_RaggedPrefillBase):
    """FA3 varlen over packed q/k/v. In graph mode it reads the layout from the
    base's static buffers, and its launch geometry is the bucket's ceiling, so
    every replay of the bucket runs the kernel that was captured."""

    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int, **kwargs):
        super().__init__(num_qo_heads, num_kv_heads, head_dim, **kwargs)
        self._causal = False
        self._cu_q: torch.Tensor | None = None
        self._cu_k: torch.Tensor | None = None
        self._max_q = 0
        self._max_k = 0

        if not self.accelerator_graph:
            self._n_valid = None
            self._row_idx = None
            return
        # varlen writes only the planned rows, so in a replay the rest hold
        # whatever the graph pool had there (a previous replay's, possibly NaN).
        # ``run`` selects zeros over rows past ``_n_valid``, the planned total.
        # Select, not multiply: inf*0 is nan.
        self._n_valid = torch.zeros((), dtype=torch.int32, device=self.device)
        self._row_idx = torch.arange(self.max_total_tokens, device=self.device)

    def _plan_kernel(
        self, cu: torch.Tensor, kv_cu: torch.Tensor, total_tokens: int, causal: bool,
    ) -> None:
        self._causal = causal
        if self.accelerator_graph:
            self._qo_indptr_buf.copy_(cu, non_blocking=True)
            self._cu_q = self._qo_indptr_buf
            if kv_cu is cu:
                self._cu_k = self._cu_q
            else:
                self._kv_indptr_buf.copy_(kv_cu, non_blocking=True)
                self._cu_k = self._kv_indptr_buf
            self._max_q, self._max_k = self.max_total_tokens, self.max_total_kv_tokens
            self._n_valid.fill_(total_tokens)
            return
        self._cu_q = cu.to(self.device, non_blocking=True)
        self._cu_k = self._cu_q if kv_cu is cu else kv_cu.to(self.device, non_blocking=True)
        self._max_q, self._max_k = _max_span(cu.cpu()), _max_span(kv_cu.cpu())

    @torch.compiler.disable
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q, k, v = (t.to(self.q_data_type).contiguous() for t in (q, k, v))
        out = flash_attn_varlen_func(
            q, k, v, self._cu_q, self._cu_k, self._max_q, self._max_k,
            causal=self._causal, softmax_scale=self.sm_scale,
        )
        out = out[0] if isinstance(out, tuple) else out
        if self._n_valid is None:
            return out
        n = q.shape[0]
        self._check_rows(n)
        return out.masked_fill((self._row_idx[:n] >= self._n_valid).view(n, 1, 1), 0)


class DenseRaggedPrefillWrapper(_DenseRaggedPrefillBase, RaggedPrefillWrapper):
    """Ragged self-attention through FA3 varlen."""


class DenseRaggedCrossPrefillWrapper(_DenseRaggedPrefillBase, RaggedCrossPrefillWrapper):
    """Ragged cross-attention through FA3 varlen."""
