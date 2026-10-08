from core.image.sorting import (
    sort_bubbles_by_reading_order,
    sort_panels_by_reading_order,
)

# 2x2 page grid: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
PANELS = [
    (0, 0, 100, 100),
    (110, 0, 210, 100),
    (0, 110, 100, 210),
    (110, 110, 210, 210),
]


def _bubble(name, bbox):
    return {"name": name, "bbox": bbox}


def _names(items):
    return [item["name"] for item in items]


def test_sort_panels_rtl_and_ltr():
    assert sort_panels_by_reading_order(PANELS, "rtl") == [1, 0, 3, 2]
    assert sort_panels_by_reading_order(PANELS, "ltr") == [0, 1, 2, 3]


def test_sort_panels_empty():
    assert sort_panels_by_reading_order([], "rtl") == []


def test_sort_bubbles_rows_without_panels():
    bubbles = [
        _bubble("left", (0, 0, 100, 50)),
        _bubble("right", (200, 0, 300, 50)),
        _bubble("below", (100, 200, 200, 250)),
    ]
    assert _names(sort_bubbles_by_reading_order(bubbles, "rtl")) == [
        "right",
        "left",
        "below",
    ]
    assert _names(sort_bubbles_by_reading_order(bubbles, "ltr")) == [
        "left",
        "right",
        "below",
    ]


def test_sort_bubbles_slightly_offset_stay_in_same_row():
    bubbles = [
        _bubble("left", (0, 10, 100, 60)),
        _bubble("right", (200, 0, 300, 50)),
    ]
    assert _names(sort_bubbles_by_reading_order(bubbles, "rtl")) == ["right", "left"]


def test_sort_bubbles_prefers_original_bbox():
    bubbles = [
        {"name": "a", "bbox": (0, 0, 10, 10), "original_bbox": (200, 0, 300, 50)},
        {"name": "b", "bbox": (200, 0, 300, 50), "original_bbox": (0, 0, 100, 50)},
    ]
    assert _names(sort_bubbles_by_reading_order(bubbles, "rtl")) == ["a", "b"]


def test_sort_bubbles_follows_panel_order():
    bubbles = [
        _bubble("bl", (20, 140, 80, 180)),
        _bubble("tl", (20, 20, 80, 60)),
        _bubble("br", (130, 140, 190, 180)),
        _bubble("tr", (130, 20, 190, 60)),
    ]
    result = sort_bubbles_by_reading_order(bubbles, "rtl", panels=PANELS)
    assert _names(result) == ["tr", "tl", "br", "bl"]
    assert {b["name"]: b["panel_id"] for b in result} == {
        "tr": 1,
        "tl": 0,
        "br": 3,
        "bl": 2,
    }


def test_sort_bubbles_far_from_panels_go_last_unassigned():
    far = _bubble("far", (2000, 2000, 2050, 2050))
    near = _bubble("near", (20, 20, 80, 60))
    result = sort_bubbles_by_reading_order([far, near], "rtl", panels=PANELS)
    assert _names(result) == ["near", "far"]
    assert far["panel_id"] is None


def test_sort_bubbles_empty():
    assert sort_bubbles_by_reading_order([], "rtl") == []
