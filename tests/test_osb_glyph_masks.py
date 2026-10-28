import threading

import cv2
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

import core.image.page_glyph_map as pgm
import core.outside_text_processor as otp
from core.config import MangaTranslatorConfig
from core.image.page_glyph_map import PageGlyphMap
from core.image.tilt_detection import detect_crop_tilt_angle

BOX = (100, 100, 300, 160)  # x0, y0, x1, y1 of the OCR text box
STROKE = (slice(125, 131), slice(140, 220))  # dark text inside the box
# Text inside the box that the fake text model finds only at twice the text scale
FAINT = (slice(145, 149), slice(240, 290))


def _page():
    rng = np.random.default_rng(0)
    # Busy mid-gray texture, so the region is never treated as a solid background
    page = rng.integers(110, 190, size=(300, 400, 3), dtype=np.uint8)
    page[STROKE] = 0
    page[FAINT] = 60
    page[20:26, 140:220] = 0  # dark art outside the box
    return Image.fromarray(page)


class _InlineCoordinator:
    """Runs scheduled per-region jobs immediately, in order."""

    def map_ordered(self, jobs):
        return [job() for job in jobs]


def _work(
    method, mode="direct", preserve_art=True, device="cuda", text_segmentation=True
):
    config = MangaTranslatorConfig(yolo_model_path="unused.pt")
    config.device = torch.device(device)  # never used by the fakes
    config.outside_text.enabled = True
    config.outside_text.inpainting_method = method
    config.outside_text.flux_preserve_art = preserve_art
    config.cleaning.text_segmentation = text_segmentation
    config.outside_text.flux_group_regions = mode == "grouped"
    if mode == "waves":
        config.request_coordinator = _InlineCoordinator()
    image = _page()
    x0, y0, x1, y1 = BOX
    # The detector's group mask is wider than the OCR box
    combined = np.zeros((300, 400), bool)
    combined[y0 - 15 : y1 + 15, x0 - 20 : x1 + 20] = True
    return otp.OutsideTextWork(
        pil_image=image,
        config=config,
        image_path="page.png",
        image_format="PNG",
        verbose=False,
        outside_text_results=[],
        raw_outside_text_results=[],
        original_text_colors={},
        total_bubble_mask=np.zeros((300, 400), bool),
        outside_detector=None,
        mask_groups=[
            {
                "combined_mask": combined,
                "original_bbox": {
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                },
                "mask_indices": [],
            }
        ],
        img_w=400,
        img_h=300,
        mime_type="image/png",
        cv2_ext=".png",
    )


class _FakeInpainter:
    """Paints white over the composite area like an edit model's redraw, which the
    composite is clipped to (the box), or returns the input on failure."""

    def __init__(self, succeed=True):
        self.masks = []
        self.succeed = succeed

    def inpaint_mask(self, image, mask, composite_clip_bbox=None, **kwargs):
        mask = np.asarray(mask, bool).copy()
        self.masks.append(mask.copy())
        if not self.succeed:
            return image
        if composite_clip_bbox is not None:
            x0, y0, x1, y1 = composite_clip_bbox
            clip = np.zeros_like(mask)
            clip[y0:y1, x0:x1] = True
            mask &= clip
        out = np.asarray(image).copy()
        out[mask] = 255
        return Image.fromarray(out)


class _FakeManager:
    """Counts text model loads from disk; like the real manager, a loaded or parked
    model is reused."""

    def __init__(self, fail=False):
        self.flux_inference_lock = threading.Lock()
        self.fail = fail
        self.loaded = False
        self.loads = 0
        self.parks = 0

    def load_text_sam(self, verbose=False):
        if not self.loaded:
            self.loads += 1
        if self.fail:
            raise OSError("offline")
        self.loaded = True
        return "text-sam"

    def park_text_sam(self, verbose=False):
        self.parks += 1


@pytest.fixture
def patched(monkeypatch):
    """Installs fakes; returns the list of windows the fake text model was run on."""
    windows = []

    def predict(model, window):
        windows.append(window.shape)
        # Windows of the 300 x 400 test pages are bigger only at twice the text scale
        cutoff = 80 if window.shape[0] > 300 else 40
        return (window.mean(axis=2) < cutoff).astype(np.float32)

    def setup(inpainter, manager):
        monkeypatch.setattr(
            otp, "_create_osb_inpainter", lambda config, verbose: inpainter
        )
        monkeypatch.setattr(otp, "get_model_manager", lambda: manager)
        monkeypatch.setattr(pgm, "get_model_manager", lambda: manager)
        monkeypatch.setattr(pgm, "predict_text_probability", predict)
        return windows

    return setup


def _box_mask():
    x0, y0, x1, y1 = BOX
    mask = np.zeros((300, 400), bool)
    mask[y0:y1, x0:x1] = True
    return mask


@pytest.mark.parametrize("mode", ["direct", "grouped", "waves"])
def test_lama_gets_glyph_masks_clipped_to_the_text_box(patched, mode):
    inpainter, manager = _FakeInpainter(), _FakeManager()
    patched(inpainter, manager)
    work = _work("lama_manga", mode)

    otp.finish_outside_text_work(work)

    (mask,) = inpainter.masks
    assert mask[128, 180]  # the stroke
    assert mask[122, 180]  # its dilated edge
    assert not mask[150, 120]  # box pixels away from the text
    assert not mask[~_box_mask()].any()  # nothing outside the OCR box, art included
    assert manager.loads == 1
    assert work.glyph_map.prob is not None


@pytest.mark.parametrize("mode", ["direct", "grouped", "waves"])
def test_lama_masks_add_glyphs_found_at_twice_the_scale_inside_the_box(patched, mode):
    inpainter, manager = _FakeInpainter(), _FakeManager()
    windows = patched(inpainter, manager)
    work = _work("lama_manga", mode)

    otp.finish_outside_text_work(work)

    (mask,) = inpainter.masks
    assert mask[147, 265] and mask[128, 180]
    assert not mask[~_box_mask()].any()
    assert work.glyph_map.prob[147, 265] < 0.5  # the shared page map is left alone
    assert windows == [(300, 400, 3), (600, 800, 3)]


def test_lama_runs_the_upscaled_pass_on_the_cpu_too(patched):
    inpainter = _FakeInpainter()
    windows = patched(inpainter, _FakeManager())

    otp.finish_outside_text_work(_work("lama_manga", device="cpu"))

    (mask,) = inpainter.masks
    assert mask[147, 265] and mask[128, 180]
    assert windows == [(300, 400, 3), (600, 800, 3)]


def test_lama_keeps_the_page_map_when_the_upscaled_pass_fails(patched, monkeypatch):
    inpainter = _FakeInpainter()
    patched(inpainter, _FakeManager())
    page_predict = pgm.predict_text_probability

    def predict(model, window):
        if window.shape[0] > 300:
            raise RuntimeError("CUDA out of memory")
        return page_predict(model, window)

    monkeypatch.setattr(pgm, "predict_text_probability", predict)

    otp.finish_outside_text_work(_work("lama_manga"))

    (mask,) = inpainter.masks
    assert mask[128, 180] and not mask[147, 265]


def test_edit_model_composites_skip_the_upscaled_pass(patched):
    windows = patched(_FakeInpainter(), _FakeManager())

    result, _ = otp.finish_outside_text_work(_work("flux_klein_4b"))

    assert (
        np.asarray(result)[FAINT] == 60
    ).all()  # pasted only on the page map's glyphs
    assert windows == [(300, 400, 3)]


def test_edit_models_still_see_the_whole_box(patched):
    inpainter, manager = _FakeInpainter(), _FakeManager()
    patched(inpainter, manager)
    work = _work("flux_klein_4b")

    otp.finish_outside_text_work(work)

    (mask,) = inpainter.masks
    np.testing.assert_array_equal(mask, work.mask_groups[0]["combined_mask"])
    # The map is built before the edit model loads, and its model moved out of VRAM
    assert (manager.loads, manager.parks) == (1, 1)


@pytest.mark.parametrize("mode", ["direct", "grouped", "waves"])
def test_edit_models_composite_only_the_glyphs(patched, mode):
    patched(_FakeInpainter(), _FakeManager())
    work = _work("flux_klein_4b", mode)
    original = np.asarray(work.pil_image).copy()

    result, _ = otp.finish_outside_text_work(work)

    out = np.asarray(result)
    assert (out[STROKE] == 255).all()  # the text is replaced by the redraw
    # Art inside the box away from the text, and everything outside it, is original
    np.testing.assert_array_equal(out[140:160, 100:300], original[140:160, 100:300])
    np.testing.assert_array_equal(out[~_box_mask()], original[~_box_mask()])


@pytest.mark.parametrize("mode", ["direct", "grouped", "waves"])
def test_edit_models_paste_the_whole_box_without_preserve_art(patched, mode):
    inpainter, manager = _FakeInpainter(), _FakeManager()
    patched(inpainter, manager)
    work = _work("flux_klein_4b", mode, preserve_art=False)

    result, _ = otp.finish_outside_text_work(work)

    assert (np.asarray(result)[_box_mask()] == 255).all()
    assert manager.loads == 0  # the text model isn't needed


def test_lama_keeps_glyph_masks_without_preserve_art(patched):
    inpainter, manager = _FakeInpainter(), _FakeManager()
    patched(inpainter, manager)
    work = _work("lama_manga", preserve_art=False)

    otp.finish_outside_text_work(work)

    (mask,) = inpainter.masks
    assert mask[128, 180] and not mask[150, 120]


def test_edit_models_composite_the_whole_box_when_the_text_model_fails(patched):
    patched(_FakeInpainter(), _FakeManager(fail=True))
    work = _work("flux_klein_4b")

    result, _ = otp.finish_outside_text_work(work)

    assert (np.asarray(result)[_box_mask()] == 255).all()


def test_lama_falls_back_to_box_masks_when_the_text_model_fails(patched):
    inpainter, manager = _FakeInpainter(), _FakeManager(fail=True)
    patched(inpainter, manager)
    work = _work("lama_manga")

    otp.finish_outside_text_work(work)

    (mask,) = inpainter.masks
    np.testing.assert_array_equal(mask, _box_mask())


def test_lama_inpaints_whole_boxes_with_text_segmentation_off(patched):
    inpainter, manager = _FakeInpainter(), _FakeManager()
    patched(inpainter, manager)

    otp.finish_outside_text_work(_work("lama_manga", text_segmentation=False))

    (mask,) = inpainter.masks
    np.testing.assert_array_equal(mask, _box_mask())
    assert manager.loads == 0


def test_edit_models_paste_the_whole_box_with_text_segmentation_off(patched):
    manager = _FakeManager()
    patched(_FakeInpainter(), manager)

    result, _ = otp.finish_outside_text_work(
        _work("flux_klein_4b", text_segmentation=False)
    )

    assert (np.asarray(result)[_box_mask()] == 255).all()
    assert manager.loads == 0


@pytest.mark.parametrize("mode", ["direct", "grouped", "waves"])
def test_failed_lama_fills_the_whole_box(patched, mode):
    inpainter, manager = _FakeInpainter(succeed=False), _FakeManager()
    patched(inpainter, manager)
    work = _work("lama_manga", mode)

    result, _ = otp.finish_outside_text_work(work)

    filled = np.asarray(result)[_box_mask()]
    assert (filled == filled[0]).all()


TILT_RAW_BOX = (80, 100, 320, 200)


def _tilted_text_page():
    """Black glyphs on a +12 degree line inside TILT_RAW_BOX, crossed by gray art lines
    that the fake text model ignores but the ink heuristic sees."""
    page = np.full((300, 400, 3), 235, np.uint8)
    for x in range(-200, 400, 45):
        cv2.line(page, (x, 300), (x + 160, 0), (70, 70, 70), 7)
    t = np.radians(12)
    for k in range(7):
        offset = (k - 3) * 28
        center = (200 + offset * np.cos(t), 150 + offset * np.sin(t))
        box = cv2.boxPoints((center, (12, 17), 12)).astype(np.int32)
        cv2.fillPoly(page, [box], (0, 0, 0))
    return Image.fromarray(page)


def _tilt_data(
    image, method="lama_manga", follow_tilt=True, glyph_map=None, text_segmentation=True
):
    config = MangaTranslatorConfig(yolo_model_path="unused.pt")
    config.outside_text.enabled = True
    config.outside_text.inpainting_method = method
    config.outside_text.osb_follow_tilt = follow_tilt
    config.cleaning.text_segmentation = text_segmentation
    config.translation.upscale_method = "none"
    config.device = torch.device("cuda")  # never used by the fakes
    if glyph_map is None:
        glyph_map = PageGlyphMap.for_page(image, config, verbose=False)
    (item,) = otp._build_outside_text_data(
        pil_image=image,
        outside_text_results=[(list(TILT_RAW_BOX), 0.9)],
        raw_outside_text_results=[(list(TILT_RAW_BOX), 0.9)],
        original_text_colors={},
        extracted_text_colors={},
        none_skipped_clip_bboxes=set(),
        mime_type="image/png",
        cv2_ext=".png",
        config=config,
        verbose=False,
        glyph_map=glyph_map,
    )
    return item, glyph_map


def _heuristic_tilt(image):
    x0, y0, x1, y1 = TILT_RAW_BOX
    crop = cv2.cvtColor(np.asarray(image)[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
    return detect_crop_tilt_angle(
        crop, max_side=256, deadband_deg=3.0, max_tilt_deg=20.0
    )


def test_tilt_is_measured_on_the_glyph_map(patched):
    manager = _FakeManager()
    patched(_FakeInpainter(), manager)
    image = _tilted_text_page()

    item, _ = _tilt_data(image)

    assert item["tilt_deg"] == pytest.approx(12, abs=2)
    assert item["tilt_conf"] >= 0.5
    assert item["tilt_axis"] == "horizontal"
    assert manager.loads == 1


def test_tilt_uses_the_ink_heuristic_with_text_segmentation_off(patched):
    manager = _FakeManager()
    patched(_FakeInpainter(), manager)
    image = _tilted_text_page()

    item, _ = _tilt_data(image, text_segmentation=False)

    angle, conf, _, axis = _heuristic_tilt(image)
    assert (item["tilt_deg"], item["tilt_conf"], item["tilt_axis"]) == (
        angle,
        conf,
        axis,
    )
    assert manager.loads == 0


def test_tilt_uses_the_ink_heuristic_when_the_text_model_fails(patched):
    patched(_FakeInpainter(), _FakeManager(fail=True))
    image = _tilted_text_page()

    item, glyph_map = _tilt_data(image)

    angle, conf, _, axis = _heuristic_tilt(image)
    assert (item["tilt_deg"], item["tilt_conf"], item["tilt_axis"]) == (
        angle,
        conf,
        axis,
    )
    assert glyph_map.failed


def test_page_map_is_built_once_for_tilt_and_lama(patched):
    inpainter, manager = _FakeInpainter(), _FakeManager()
    windows = patched(inpainter, manager)
    work = _work("lama_manga")

    _tilt_data(work.pil_image, glyph_map=work.glyph_map)
    otp.finish_outside_text_work(work)

    assert len(inpainter.masks) == 1
    assert manager.loads == 1
    assert windows == [(300, 400, 3), (600, 800, 3)]  # the page map, then the box pass
    assert manager.parks == 0  # LaMa keeps the text model on the GPU for the next page


@pytest.mark.parametrize("method", ["flux_klein_4b", "qwen_image_2_1_turbo"])
def test_text_model_is_freed_after_the_map_for_diffusion_methods(patched, method):
    manager = _FakeManager()
    patched(_FakeInpainter(), manager)

    _, glyph_map = _tilt_data(_tilted_text_page(), method=method)

    assert glyph_map.prob is not None
    assert (manager.loads, manager.parks) == (1, 1)


@pytest.mark.parametrize("method", ["opencv", "none"])
def test_text_model_stays_on_the_device_without_a_diffusion_model(patched, method):
    manager = _FakeManager()
    patched(_FakeInpainter(), manager)

    _tilt_data(_tilted_text_page(), method=method)

    assert (manager.loads, manager.parks) == (1, 0)


@pytest.mark.parametrize(
    ("osb", "colored_bubbles", "parks"),
    [(False, False, 0), (False, True, 1), (True, False, 1)],
)
def test_text_model_is_freed_only_when_a_diffusion_model_will_run(
    patched, osb, colored_bubbles, parks
):
    manager = _FakeManager()
    patched(_FakeInpainter(), manager)
    config = MangaTranslatorConfig(yolo_model_path="unused.pt")
    config.outside_text.enabled = osb
    config.outside_text.inpainting_method = "flux_klein_4b"
    config.cleaning.inpaint_colored_bubbles = colored_bubbles

    PageGlyphMap.for_page(_page(), config, verbose=False).get()

    assert manager.parks == parks


def test_next_page_reuses_the_parked_text_model(patched):
    manager = _FakeManager()
    patched(_FakeInpainter(), manager)

    _tilt_data(_tilted_text_page(), method="flux_klein_4b")
    _tilt_data(_tilted_text_page(), method="flux_klein_4b")

    assert (manager.loads, manager.parks) == (1, 2)


def test_text_model_is_freed_even_when_the_map_fails(patched):
    manager = _FakeManager(fail=True)
    patched(_FakeInpainter(), manager)

    _tilt_data(_tilted_text_page(), method="flux_klein_4b")

    assert manager.parks == 1


def test_no_map_is_built_without_tilt_for_other_methods(patched):
    manager = _FakeManager()
    windows = patched(_FakeInpainter(), manager)

    item, _ = _tilt_data(_tilted_text_page(), method="flux_klein_4b", follow_tilt=False)

    assert item["tilt_deg"] == 0.0
    assert manager.loads == 0
    assert windows == []


def test_tilt_without_a_map_uses_the_ink_heuristic(patched):
    manager = _FakeManager()
    patched(_FakeInpainter(), manager)
    image = _tilted_text_page()
    config = MangaTranslatorConfig(yolo_model_path="unused.pt")
    config.translation.upscale_method = "none"

    (item,) = otp._build_outside_text_data(
        pil_image=image,
        outside_text_results=[(list(TILT_RAW_BOX), 0.9)],
        raw_outside_text_results=[(list(TILT_RAW_BOX), 0.9)],
        original_text_colors={},
        extracted_text_colors={},
        none_skipped_clip_bboxes=set(),
        mime_type="image/png",
        cv2_ext=".png",
        config=config,
        verbose=False,
        glyph_map=None,
    )

    angle, conf, _, axis = _heuristic_tilt(image)
    assert (item["tilt_deg"], item["tilt_conf"], item["tilt_axis"]) == (
        angle,
        conf,
        axis,
    )
    assert manager.loads == 0


SFX_BLUE = (60, 140, 230)


def _color_sfx_work(device, method):
    """A color page with blue outlined SFX in BOX, and a text map covering its letters."""
    rng = np.random.default_rng(0)
    page = Image.fromarray(
        np.repeat(np.repeat(rng.integers(0, 256, (30, 40, 3), np.uint8), 10, 0), 10, 1)
    )
    x0, y0, x1, y1 = BOX
    font = ImageFont.load_default(size=50)
    ImageDraw.Draw(page).text(
        (x0 + 10, y0), "ZAP", font=font, fill=SFX_BLUE, stroke_width=2, stroke_fill=0
    )
    letters = Image.new("L", page.size, 0)
    ImageDraw.Draw(letters).text(
        (x0 + 10, y0), "ZAP", font=font, fill=255, stroke_width=2, stroke_fill=255
    )
    config = MangaTranslatorConfig(yolo_model_path="unused.pt")
    config.outside_text.inpainting_method = method
    config.device = torch.device(device)
    work = otp.OutsideTextWork(
        pil_image=page,
        config=config,
        image_path="page.png",
        image_format="PNG",
        verbose=False,
        outside_text_results=[],
        raw_outside_text_results=[],
        original_text_colors={},
        total_bubble_mask=np.zeros((300, 400), bool),
        outside_detector=None,
        mask_groups=[],
        img_w=400,
        img_h=300,
        mime_type="image/png",
        cv2_ext=".png",
        is_color_page=True,
    )
    work.glyph_map.prob = np.asarray(letters, np.float32) / 255
    group = {
        "original_bbox": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
        "mask_indices": [],
    }
    return work, group


@pytest.mark.parametrize(
    ("device", "method"),
    [("cuda", "opencv"), ("cpu", "lama_manga"), ("cpu", "opencv")],
)
def test_color_page_sfx_keep_their_hue(device, method):
    work, group = _color_sfx_work(device, method)

    color = otp._decide_osb_region_fill(work, work.pil_image, group, 1).text_color

    assert np.abs(np.subtract(color, SFX_BLUE)).max() <= 10


MISSED_LETTERS = [(50, 200, 62, 220), (66, 200, 78, 220), (82, 200, 94, 220)]
BUBBLE_BOX = (240, 30, 300, 80)
YOLO_BOX = (295, 195, 365, 220)


def _missed_text_page():
    page = np.full((300, 400, 3), 255, np.uint8)
    for x0, y0, x1, y1 in MISSED_LETTERS:
        page[y0:y1, x0:x1] = 0
    page[40:60, 250:280] = 0  # dialogue inside the bubble
    page[200:215, 300:360] = 0  # text the OSB model found
    return Image.fromarray(page)


class _FakeOsbDetector:
    def __init__(self):
        self.near_calls = []

    def detect_outside_text(self, *args, **kwargs):
        return [(np.array(YOLO_BOX, np.float32), 0.8)]

    def get_text_masks(self, *args, **kwargs):
        return [], None

    def detect_text_near(self, image_cv, boxes, confidence):
        self.near_calls += [tuple(box) for box in boxes]
        return [
            [((float(b[0]), float(b[1]), float(b[2]), float(b[3])), 0.7)]
            if b[0] < 100
            else []
            for b in boxes
        ]


@pytest.fixture
def recovery(patched, monkeypatch):
    patched(_FakeInpainter(), _FakeManager())
    detector = _FakeOsbDetector()
    monkeypatch.setattr(otp, "OutsideTextDetector", lambda **kwargs: detector)

    def run(device="cuda", method="lama_manga", **settings):
        config = MangaTranslatorConfig(yolo_model_path="unused.pt")
        config.device = torch.device(device)  # never used by the fakes
        config.outside_text.enabled = True
        config.outside_text.inpainting_method = method
        config.translation.upscale_method = "none"
        for name, value in settings.items():
            section, field = name.split(".")
            setattr(getattr(config, section), field, value)
        work = otp.prepare_outside_text_work(
            _missed_text_page(),
            config,
            "page.png",
            "PNG",
            bubble_data=[{"bbox": list(BUBBLE_BOX)}],
            text_free_boxes=[],
        )
        boxes = [tuple(int(c) for c in b) for b, _ in work.raw_outside_text_results]
        return boxes, detector.near_calls

    return run


@pytest.mark.parametrize(
    ("device", "method"), [("cuda", "lama_manga"), ("cpu", "opencv")]
)
def test_text_the_osb_model_missed_is_recovered_from_the_glyph_map(
    recovery, device, method
):
    boxes, near_calls = recovery(device=device, method=method)

    # Only the letters outside every bubble and OSB box are checked, as one group
    assert near_calls == [(50, 200, 94, 220)]
    assert boxes == [YOLO_BOX, (50, 200, 94, 220)]


@pytest.mark.parametrize(
    "settings",
    [
        {"outside_text.osb_recover_missed_text": False},
        {"outside_text.osb_text_free_only": True},
        {"cleaning.text_segmentation": False},
    ],
)
def test_missed_text_recovery_is_skipped(recovery, settings):
    boxes, near_calls = recovery(**settings)

    assert near_calls == []
    assert boxes == [YOLO_BOX]
