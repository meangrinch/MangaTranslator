import numpy as np

from core.image.ocr_detection import OutsideTextDetector


def _detector() -> OutsideTextDetector:
    # The filter is pure box math; skip loading the model manager and cache
    return object.__new__(OutsideTextDetector)


def _result(box):
    return (np.array(box, dtype=np.float32), 0.9)


def test_text_free_box_inside_bubble_still_drops_the_bubbles_own_text():
    # female_knight_raw_6: the あったかい bubble's box swallows the えへへ… SFX
    bubble = [88, 190, 205, 415]
    sfx_text_free = [82, 332, 123, 418]
    bubble_text = _result([135, 204, 169, 364])
    sfx_text = _result([88, 329, 118, 415])

    kept = _detector().drop_in_bubble_detections(
        [bubble_text, sfx_text], [np.array(bubble)], [np.array(sfx_text_free)]
    )

    assert [r[0].tolist() for r in kept] == [sfx_text[0].tolist()]


def test_text_in_a_bubble_shaped_text_free_region_is_kept():
    # The bubble model mistook a caption for a bubble; its text is outside text
    caption = [400, 100, 520, 220]
    caption_text = _result([420, 120, 500, 200])

    kept = _detector().drop_in_bubble_detections(
        [caption_text], [np.array(caption)], [np.array([402, 98, 518, 222])]
    )

    assert len(kept) == 1


def test_text_in_a_plain_bubble_is_dropped():
    kept = _detector().drop_in_bubble_detections(
        [_result([120, 120, 180, 200])], [np.array([100, 100, 200, 220])], []
    )

    assert kept == []
