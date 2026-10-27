import numpy as np
import torch

from core.ml.hi_sam import build_text_sam_ts_l, predict_text_probability


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
