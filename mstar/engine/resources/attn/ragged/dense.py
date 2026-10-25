"""Cacheless varlen attention through FA3 (AITER on ROCm).

Sibling of ``ragged/flashinfer.py``: the same plan / preplan / CUDA-graph slot
machinery from ``ragged/base.py``, with the wrappers in ``dense_wrappers.py``.
FA3 takes no workspace, so unlike FlashInfer's managers these allocate none.
"""

import functools
import logging

from mstar.engine.resources.attn.ragged.base import _RaggedCrossManagerBase, _RaggedSelfManagerBase
from mstar.engine.resources.attn.ragged.config import RaggedAttnBackend
from mstar.engine.resources.attn.ragged.dense_wrappers import (
    DenseRaggedCrossPrefillWrapper,
    DenseRaggedPrefillWrapper,
)

logger = logging.getLogger(__name__)


@functools.cache
def _warn_ragged_dense_fallback(reason: str, fallback: RaggedAttnBackend) -> None:
    logger.warning(
        "Ragged attention backend 'dense' requested, but %s; using the ragged %r backend instead.",
        reason, fallback.value,
    )


class DenseRaggedManager(_RaggedSelfManagerBase):
    wrapper_class = DenseRaggedPrefillWrapper


class DenseRaggedCrossManager(_RaggedCrossManagerBase):
    wrapper_class = DenseRaggedCrossPrefillWrapper
