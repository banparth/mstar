"""Cacheless varlen attention through FlashAttention-3 (AITER on ROCm).

Sibling of ``ragged/flashinfer.py``: the same plan / preplan / CUDA-graph slot
machinery, with ``flash_attn_varlen_func`` in place of FlashInfer's ragged
prefill wrapper. Nothing is paged and nothing is scattered; q/k/v are packed
by segment the same way the FlashInfer path takes them.
"""

import functools
import logging

import torch

from mstar.engine.resources.attn.dense import flash_attn_varlen_func
from mstar.engine.resources.attn.ragged.config import RaggedAttnBackend
from mstar.engine.resources.attn.ragged.flashinfer import (
    FlashInferRaggedCrossManager,
    FlashInferRaggedManager,
)

logger = logging.getLogger(__name__)


@functools.cache
def _warn_ragged_dense_fallback(reason: str, fallback: RaggedAttnBackend) -> None:
    logger.warning(
        "Ragged attention backend 'dense' requested but no varlen kernel is "
        "available (%s); using the ragged %r backend instead.",
        reason, fallback.value,
    )


def _max_span(cu_seqlens: torch.Tensor) -> int:
    """Longest segment in a host ``cu_seqlens``. Zero if there are no tokens."""
    if cu_seqlens.numel() < 2:
        return 0
    spans = cu_seqlens[1:] - cu_seqlens[:-1]
    return int(spans.max()) if spans.numel() else 0


def _varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    out = flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, max_q, max_k,
        causal=causal,
        softmax_scale=sm_scale,
    )
    return out[0] if isinstance(out, tuple) else out


class _DenseRaggedPrefillBase:
    """FA3 varlen over packed q/k/v, laid out by ``cu_seqlens``.

    Same constructor kwargs as the FlashInfer ragged wrappers so the shared
    manager base can build either. Graph mode copies layouts into stable
    device buffers (zero-length padding segments) and owns the output so a
    padded tail cannot leak the previous replay.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_num_segments: int | None = None,
        max_total_tokens: int | None = None,
        max_total_kv_tokens: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
        sm_scale: float | None = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_layout: str = "NHD",
        backend: str = "auto",
    ):
        del workspace_buffer, kv_layout, backend
        self.device = device
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = float(sm_scale) if sm_scale is not None else head_dim ** -0.5
        self.q_data_type = q_data_type
        self.use_cuda_graph = use_cuda_graph
        self.max_num_segments = max_num_segments
        self.max_total_tokens = max_total_tokens
        self.max_total_kv_tokens = (
            max_total_kv_tokens if max_total_kv_tokens is not None else max_total_tokens
        )
        self._num_segments = 0
        self._causal = False
        self._cu_q: torch.Tensor | None = None
        self._cu_k: torch.Tensor | None = None
        self._max_q = 0
        self._max_k = 0

        if use_cuda_graph:
            assert max_num_segments is not None, "max_num_segments required for CUDA graph mode"
            assert max_total_tokens is not None, "max_total_tokens required for CUDA graph mode"
            assert max_num_segments > 0, "max_num_segments must be positive"

            self._qo_indptr_buf = torch.zeros(
                max_num_segments + 1, dtype=torch.int32, device=device,
            )
            self._kv_indptr_buf = torch.zeros(
                max_num_segments + 1, dtype=torch.int32, device=device,
            )
            # Own the output: varlen writes only the planned rows, and a
            # captured graph's pool keeps the previous replay's values in the
            # tail. ``plan`` zeros it from the host count; ``run`` also
            # selects zeros over rows past ``_n_valid`` so a kernel that
            # allocates ``empty`` cannot leak. Select, not multiply: inf*0 is nan.
            self._out_buf = torch.zeros(
                max_total_tokens, num_qo_heads, head_dim,
                dtype=q_data_type, device=device,
            )
            self._n_valid = torch.zeros((), dtype=torch.int32, device=device)
            self._row_idx = torch.arange(max_total_tokens, device=device)
        else:
            self._qo_indptr_buf = None
            self._kv_indptr_buf = None
            self._out_buf = None
            self._n_valid = None
            self._row_idx = None

    @property
    def num_segments(self) -> int:
        """Real (unpadded) segment count from the most recent ``plan``."""
        return self._num_segments

    def _prepare_cu_seqlens(
        self, cu_seqlens: torch.Tensor, max_tokens: int, side: str = "",
    ) -> tuple[torch.Tensor, int]:
        """Host layout to stage, and its real token count (graph) or -1 (eager)."""
        n_seg = int(cu_seqlens.numel()) - 1
        if not self.use_cuda_graph:
            return cu_seqlens.to(torch.int32), -1

        if n_seg > self.max_num_segments:
            raise ValueError(
                f"{type(self).__name__}: {n_seg} segments exceeds the "
                f"{self.max_num_segments} this graph-mode wrapper was built for"
            )
        host = cu_seqlens.to(device="cpu", dtype=torch.int32)
        total_tokens = int(host[-1])
        if total_tokens > max_tokens:
            raise ValueError(
                f"{type(self).__name__}: {total_tokens} {side}tokens exceeds the "
                f"{max_tokens} this graph-mode wrapper was built for"
            )
        if n_seg == self.max_num_segments:
            return host, total_tokens
        cu = torch.empty(
            self.max_num_segments + 1, dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )
        cu[: n_seg + 1].copy_(host)
        # Repeating the final offset appends zero-length segments — pads the
        # segment count to the fixed size without adding tokens.
        cu[n_seg + 1:] = total_tokens
        return cu, total_tokens

    def _stage(
        self,
        cu_seqlens: torch.Tensor,
        buf: torch.Tensor | None,
        max_tokens: int,
        side: str = "",
    ) -> tuple[torch.Tensor, int, int]:
        """Device ``cu_seqlens``, real token count, and longest live span."""
        host, total_tokens = self._prepare_cu_seqlens(cu_seqlens, max_tokens, side)
        max_span = _max_span(host if total_tokens >= 0 else host.cpu())
        if self.use_cuda_graph:
            buf.copy_(host, non_blocking=True)
            return buf, total_tokens, max_span
        return host.to(device=self.device, dtype=torch.int32, non_blocking=True), -1, max_span

    def _plan(
        self, cu_seqlens: torch.Tensor, kv_cu_seqlens: torch.Tensor, causal: bool,
    ) -> None:
        cu_q, total_q, max_q = self._stage(
            cu_seqlens, self._qo_indptr_buf, self.max_total_tokens or 0,
        )
        if kv_cu_seqlens is cu_seqlens:
            cu_k, total_k, max_k = cu_q, total_q, max_q
        else:
            cu_k, total_k, max_k = self._stage(
                kv_cu_seqlens, self._kv_indptr_buf,
                self.max_total_kv_tokens or 0, "key ",
            )
        self._num_segments = int(cu_seqlens.numel()) - 1
        self._cu_q = cu_q
        self._cu_k = cu_k
        self._causal = causal
        # Graph launch geometry is sized to the bucket so it stays constant
        # across replays; eager uses this layout's longest segment.
        if self.use_cuda_graph:
            self._max_q = self.max_total_tokens
            self._max_k = self.max_total_kv_tokens
            self._n_valid.fill_(total_q)
            self._out_buf[total_q:].zero_()
        else:
            self._max_q = max_q
            self._max_k = max_k
        del total_k

    @torch.compiler.disable
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run the planned attention.

        Args:
            q: [total_tokens, num_qo_heads, head_dim], packed by the query layout
            k, v: [total_kv_tokens, num_kv_heads, head_dim], packed by the key layout
        Returns:
            output: [total_tokens, num_qo_heads, head_dim]
        """
        q, k, v = (t.to(self.q_data_type).contiguous() for t in (q, k, v))
        out = _varlen(
            q, k, v, self._cu_q, self._cu_k, self._max_q, self._max_k,
            self._causal, self.sm_scale,
        )
        if self._out_buf is None:
            return out
        n = q.shape[0]
        assert n <= self.max_total_tokens, (
            f"{type(self).__name__}: {n} rows exceeds the "
            f"{self.max_total_tokens} this graph-mode wrapper was built for"
        )
        self._out_buf[:n].copy_(out[:n])
        tail = (self._row_idx[:n] >= self._n_valid).view(n, 1, 1)
        self._out_buf[:n].masked_fill_(tail, 0)
        return self._out_buf[:n]


class DenseRaggedPrefillWrapper(_DenseRaggedPrefillBase):
    """Varlen self-attention: each segment attends within itself."""

    @torch.compiler.disable
    def plan(self, cu_seqlens: torch.Tensor, causal: bool = False) -> None:
        """Plan one packed layout. ``cu_seqlens``: ``[num_segments + 1]``, [0] == 0.

        CPU tensor preferred; a GPU one costs a sync. Safe to call before every
        replay — values are copied through the static buffers, not rebound.
        """
        self._plan(cu_seqlens, cu_seqlens, causal)


class DenseRaggedCrossPrefillWrapper(_DenseRaggedPrefillBase):
    """Varlen cross-attention between two packed layouts. Never causal."""

    @torch.compiler.disable
    def plan(self, q_cu_seqlens: torch.Tensor, kv_cu_seqlens: torch.Tensor) -> None:
        """Plan one query layout against one key layout, each ``[num_segments + 1]``
        with [0] == 0 and the same segment count. CPU tensors preferred."""
        n_q, n_kv = int(q_cu_seqlens.numel()) - 1, int(kv_cu_seqlens.numel()) - 1
        if n_q != n_kv:
            raise ValueError(
                f"DenseRaggedCrossPrefillWrapper: {n_q} query segments but {n_kv} key "
                "segments; a cross-attention plan pairs them one to one"
            )
        self._plan(q_cu_seqlens, kv_cu_seqlens, causal=False)


class DenseRaggedManager(FlashInferRaggedManager):
    """Self-attention through FA3 varlen; same plan protocol as FlashInfer."""

    wrapper_class = DenseRaggedPrefillWrapper


class DenseRaggedCrossManager(FlashInferRaggedCrossManager):
    """Cross-attention through FA3 varlen; same plan protocol as FlashInfer."""

    wrapper_class = DenseRaggedCrossPrefillWrapper
