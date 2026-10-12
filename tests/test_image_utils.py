import numpy as np
import pytest
from PIL import Image

from core.image.image_utils import resize_lanczos_antiring


@pytest.mark.parametrize("size", [(24, 24), (20, 11), (3, 5)])
def test_resize_lanczos_antiring_keeps_hard_edges_within_source_range(size):
    src = np.full((8, 8), 64, dtype=np.uint8)
    src[:, 4:] = 192
    image = Image.fromarray(src).convert("RGB")

    result = np.asarray(resize_lanczos_antiring(image, size))

    assert result.shape == (size[1], size[0], 3)
    assert result.min() == 64
    assert result.max() == 192
