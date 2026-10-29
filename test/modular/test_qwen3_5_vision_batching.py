"""Qwen3.5's vision encoder packs several prompts into one step: which batches a
packed replay holds, what the replay declares, that each prompt gets back
exactly the embeddings it would get alone, and that a new batch does not
recompile the forward."""
from types import SimpleNamespace

import torch

from mstar.engine.engine import Engine
from mstar.model.qwen3_5.submodules import VISION_BLOCK_LOOP, VisionEncoderSubmodule
from mstar.model.submodule_base import BatchedModelOutput, ModelInputsFromEngine, NodeInputs

MERGE = 2
HIDDEN = 8


class _FakeTower:
    """Per-patch embed and encode, and a merger folding each 4 rows into one,
    like the real tower's 2x2 spatial merge."""

    def embed(self, pixel_values, indices, weights, position_ids):
        hidden = pixel_values * 2 + indices[:, :1] + weights[:, :1] + position_ids[:, :1]
        return hidden, hidden.clone(), hidden.clone()

    def encode(self, hidden, cos, sin):
        return hidden + 1

    def merger(self, encoded):
        return encoded.view(-1, MERGE * MERGE, HIDDEN).sum(dim=1)


def _encoder(max_segments=64):
    enc = VisionEncoderSubmodule.__new__(VisionEncoderSubmodule)
    torch.nn.Module.__init__(enc)
    enc.model = _FakeTower()
    enc.config = SimpleNamespace(spatial_merge_size=MERGE, hidden_size=HIDDEN, head_dim=HIDDEN)
    enc.node_resources = {"vision_attn": SimpleNamespace(max_segments_per_request=max_segments)}
    enc.get_device = lambda: torch.device("cpu")
    return enc


def _prompt(seed, spans):
    """A prompt of ``spans`` patches per image (each a multiple of 4)."""
    g = torch.Generator().manual_seed(seed)
    n = sum(spans)
    return NodeInputs(
        input_seq_len=n,
        tensor_inputs={
            "pixel_values": torch.randn(n, HIDDEN, generator=g),
            "indices": torch.randint(0, 10, (n, 4), generator=g).float(),
            "weights": torch.rand(n, 4, generator=g),
            "position_ids": torch.randint(0, 10, (n, 2), generator=g).float(),
            "seq_lengths": tuple(spans),
        },
    )


def _run(enc, rids, prompts, forward=None):
    """The step as the engine runs it: the forward, then the per-prompt cut."""
    kwargs = enc.preprocess("encode", None, prompts)
    engine_inputs = ModelInputsFromEngine(request_ids=list(rids), per_request_info={})
    raw = BatchedModelOutput.coerce((forward or enc.forward_batched)("encode", engine_inputs, **kwargs))
    outputs = {}
    Engine.__new__(Engine)._merge_unpacked(outputs, raw, list(rids), enc, prompts, {})
    return outputs


def test_packed_prompts_match_each_prompt_alone():
    enc = _encoder()
    prompts = {"a": _prompt(0, [16]), "b": _prompt(1, [64]), "c": _prompt(2, [4])}
    packed = _run(enc, prompts, list(prompts.values()))
    assert list(packed) == list(prompts)
    for rid, inp in prompts.items():
        alone = _run(enc, [rid], [inp])[rid]["vision_embeds"][0]
        got = packed[rid]["vision_embeds"][0]
        assert got.shape == (inp.input_seq_len // (MERGE * MERGE), HIDDEN)
        assert torch.equal(got, alone)


def test_new_batches_do_not_recompile():
    """Request handles and per-prompt patch counts change every step; once each
    batch size has run, neither may be a guard of the compiled forward."""
    torch._dynamo.reset()
    enc = _encoder()
    forward = torch.compile(enc.forward_batched, backend="eager", dynamic=True)
    try:
        for size in (1, 2, 3, 4):
            _run(enc, range(size), [_prompt(i, [16]) for i in range(size)], forward)
        with torch._dynamo.config.patch(error_on_recompile=True):
            for size in (4, 1, 3, 2):
                rids = range(10 * size, 11 * size)
                prompts = [_prompt(i, [8 + 4 * i]) for i in range(size)]
                assert list(_run(enc, rids, prompts, forward)) == list(rids)
    finally:
        torch._dynamo.reset()


def test_which_batches_a_replay_holds():
    enc = _encoder(max_segments=3)
    one, two, four = _prompt(0, [8]), _prompt(1, [8, 8]), _prompt(2, [4, 4, 4, 4])
    assert enc._fits_capture([two])           # one prompt: up to the per-request ceiling
    assert not enc._fits_capture([four])      # past it: eager
    assert enc._fits_capture([one, one, one])  # packed one-image prompts
    assert not enc._fits_capture([one, two])  # a packed replay declares one segment a prompt


def test_replay_declares_segments_per_image_alone_and_per_prompt_packed():
    enc = _encoder()
    config = enc.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)[VISION_BLOCK_LOOP]
    assert config.capture_batch_sizes == enc.BLOCK_LOOP_CAPTURE_BATCH_SIZES
    alone = config.declare_step(["a"], [16, 32])
    assert [(s.request_id, s.span) for s in alone.segments] == [("a", 16), ("a", 32)]
    # packed, padded to a 4-row bucket with zero-length rows
    packed = config.declare_step(["a", "b", "pad0", "pad1"], [16, 32, 0, 0])
    assert [(s.request_id, s.span) for s in packed.segments] == [("a", 16), ("b", 32), ("pad0", 0), ("pad1", 0)]
