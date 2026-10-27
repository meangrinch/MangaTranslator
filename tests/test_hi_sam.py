import numpy as np
import pytest
import torch

from core.ml import hi_sam
from core.ml.hi_sam import (
    autocast_dtype,
    build_text_sam_ts_l,
    image_encoder,
    predict_text_probability,
    store_matmul_weights,
    to_inference_device,
)
from core.ml.hi_sam.image_encoder import (
    Attention,
    add_decomposed_rel_pos,
    fused_rel_pos_attention,
    uses_fused_attention,
)


def test_text_sam_matches_koharu_checkpoint_layout():
    # mayocream/koharu-text-sam-ts-l config.json: 728 tensors, 338,934,897 elements
    with torch.device("meta"):
        model = build_text_sam_ts_l()
    state = model.state_dict()
    assert len(state) == 728
    assert sum(t.numel() for t in state.values()) == 338_934_897


class _FakeTextSam(torch.nn.Module):
    """Stands in for SAM-TS: logits are +8 where the input is dark, -8 elsewhere."""

    def __init__(self):
        super().__init__()
        self.seen_shape = None
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images):
        self.seen_shape = tuple(images.shape)
        dark = images.mean(dim=1, keepdim=True) < 64
        return torch.where(dark, 8.0, -8.0)


def test_predict_pads_window_to_1024_and_crops_back():
    model = _FakeTextSam()
    window = np.full((300, 500, 3), 255, np.uint8)
    window[100:120, 200:260] = 0

    prob = predict_text_probability(model, window)

    assert model.seen_shape == (1, 3, 1024, 1024)
    assert prob.shape == (300, 500)
    assert prob.dtype == np.float32
    assert prob[110, 230] > 0.99
    assert prob[10, 10] < 0.01


def _explicit_attention(q, k, v, scale, rel_pos_h, rel_pos_w, size):
    attn = (q * scale) @ k.transpose(-2, -1)
    attn = add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, size, size)
    return attn.softmax(dim=-1) @ v


def test_fused_attention_matches_explicit_scores_with_rel_pos_bias():
    torch.manual_seed(0)
    size = (6, 8)
    tokens, dim = size[0] * size[1], 16
    q, k, v = torch.randn(3, 4, tokens, dim).unbind(0)
    rel_pos_h = torch.randn(2 * size[0] - 1, dim)
    rel_pos_w = torch.randn(2 * size[1] - 1, dim)

    fused = fused_rel_pos_attention(q, k, v, dim**-0.5, rel_pos_h, rel_pos_w, size)

    expected = _explicit_attention(q, k, v, dim**-0.5, rel_pos_h, rel_pos_w, size)
    torch.testing.assert_close(fused, expected, rtol=1e-5, atol=1e-5)


def test_only_global_attention_blocks_are_fused():
    with torch.device("meta"):
        model = build_text_sam_ts_l()

    fused = [i for i, b in enumerate(model.image_encoder.blocks) if b.attn.fused]

    assert fused == [5, 11, 17, 23]


def test_attention_on_cpu_keeps_the_explicit_scores(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("fused attention used on the CPU")

    monkeypatch.setattr(image_encoder, "fused_rel_pos_attention", unavailable)
    torch.manual_seed(0)
    attn = Attention(32, num_heads=2, use_rel_pos=True, input_size=(4, 4), fused=True)
    torch.nn.init.normal_(attn.rel_pos_h)

    out = attn(torch.randn(1, 4, 4, 32))

    assert out.shape == (1, 4, 4, 32)


def _layers():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 1),
        torch.nn.ConvTranspose2d(4, 4, 2),
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 4),
    )


def test_inference_device_keeps_fp32_weights_on_cpu():
    model = to_inference_device(_layers(), torch.device("cpu"))

    assert {p.dtype for p in model.parameters()} == {torch.float32}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_inference_device_stores_matmul_weights_in_fp16_on_cuda():
    model = to_inference_device(_layers(), torch.device("cuda"))

    # Autocast casts these to fp16 anyway; norms run in fp32 under autocast
    assert [m.weight.dtype for m in model] == [
        torch.float16,
        torch.float16,
        torch.float32,
        torch.float16,
    ]
    assert model[0].bias.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_attention_runs_on_cuda_under_autocast(monkeypatch):
    calls = []
    original = image_encoder.fused_rel_pos_attention

    def spy(*args):
        calls.append(args[0].dtype)
        return original(*args)

    monkeypatch.setattr(image_encoder, "fused_rel_pos_attention", spy)
    torch.manual_seed(0)
    attn = Attention(32, num_heads=2, use_rel_pos=True, input_size=(4, 4), fused=True)
    attn = attn.cuda()
    x = torch.randn(1, 4, 4, 32, device="cuda")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = attn(x)

    assert calls == [torch.float16]
    assert out.shape == (1, 4, 4, 32)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", None),
        ("cuda", torch.float16),
        ("mps", torch.float16),
        ("xpu", torch.float16),
    ],
)
def test_autocast_dtype_per_device(device, dtype):
    assert autocast_dtype(torch.device(device)) == dtype


def test_matmul_weights_take_the_autocast_dtype():
    model = store_matmul_weights(_layers(), torch.float16)

    assert [m.weight.dtype for m in model] == [
        torch.float16,
        torch.float16,
        torch.float32,
        torch.float16,
    ]


@pytest.mark.parametrize(
    ("device", "fused"),
    [("cuda", True), ("mps", True), ("xpu", True), ("cpu", False), ("meta", False)],
)
def test_fused_attention_runs_where_autocast_does(device, fused):
    assert uses_fused_attention(torch.device(device)) is fused


class _AutocastProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.autocast = None

    def forward(self, images):
        device = images.device.type
        self.autocast = (
            torch.is_autocast_enabled(device),
            torch.get_autocast_dtype(device),
        )
        return torch.zeros(1, 1, *images.shape[-2:], device=images.device)


def test_predict_runs_without_autocast_on_cpu():
    model = _AutocastProbe()

    predict_text_probability(model, np.zeros((8, 8, 3), np.uint8))

    assert model.autocast[0] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_predict_autocasts_to_the_device_dtype_on_cuda(monkeypatch):
    monkeypatch.setattr(hi_sam, "autocast_dtype", lambda device: torch.float16)
    model = _AutocastProbe().cuda()

    predict_text_probability(model, np.zeros((8, 8, 3), np.uint8))

    assert model.autocast == (True, torch.float16)
