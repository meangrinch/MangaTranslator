import re

import gradio as gr
import pytest
import torch

from ui import callbacks, layout


def _unpacked_count(fn, *args) -> int:
    """Number of values `fn` unpacks from its positional args, read from the error."""
    with pytest.raises(ValueError, match="expected") as excinfo:
        fn(*args)
    return int(re.search(r"expected (\d+)", str(excinfo.value)).group(1))


def test_positional_ui_wiring_matches_the_callbacks(tmp_path):
    app = layout.create_layout(
        models_dir=tmp_path, fonts_base_dir=tmp_path, target_device=torch.device("cpu")
    )
    events = {}
    for dep in app.fns.values():
        fn = getattr(dep.fn, "func", dep.fn)
        events.setdefault(getattr(fn, "__name__", ""), dep)

    settings_count = _unpacked_count(callbacks._build_ui_state_from_args, (), False)
    assert len(events["handle_translate_click"].inputs) == 1 + settings_count
    assert len(events["handle_batch_click"].inputs) == 2 + settings_count
    assert len(events["handle_save_config_click"].inputs) == _unpacked_count(
        callbacks.handle_save_config_click
    )
    assert len(events["handle_reset_defaults_click"].outputs) == len(
        callbacks.handle_reset_defaults_click(tmp_path)
    )


def test_reset_defaults_sends_choices_with_every_choice_radio_value(tmp_path):
    """Gradio validates a Radio against the session's last-sent choices, so a reset
    that sends a value without choices fails on the next event that reads it."""
    app = layout.create_layout(
        models_dir=tmp_path, fonts_base_dir=tmp_path, target_device=torch.device("cpu")
    )
    reset = next(
        dep
        for dep in app.fns.values()
        if getattr(getattr(dep.fn, "func", dep.fn), "__name__", "")
        == "handle_reset_defaults_click"
    )
    updates = callbacks.handle_reset_defaults_click(tmp_path)

    model_dependent_radios = {
        "reasoning_effort_dropdown",
        "effort_dropdown",
        "verbosity_dropdown",
        "image_detail_dropdown",
        "media_resolution_bubbles_dropdown",
        "media_resolution_context_dropdown",
    }
    checked = set()
    for component, update in zip(reset.outputs, updates, strict=True):
        if component.elem_id not in model_dependent_radios:
            continue
        checked.add(component.elem_id)
        if update.get("value") is None:
            continue
        assert "choices" in update, component.elem_id
        choice_values = [c[1] if isinstance(c, tuple) else c for c in update["choices"]]
        assert update["value"] in choice_values, component.elem_id
    assert checked == model_dependent_radios


INPAINTING_METHODS = (
    "flux_klein_9b",
    "flux_klein_4b",
    "flux_kontext",
    "flux3_klein",
    "fibo2_edit_turbo",
    "qwen_image_2_1_turbo",
    "lama_manga",
    "opencv",
    "none",
)


@pytest.mark.parametrize("method", INPAINTING_METHODS)
def test_inpainting_method_change_keeps_radio_values_in_their_choices(tmp_path, method):
    app = layout.create_layout(
        models_dir=tmp_path, fonts_base_dir=tmp_path, target_device=torch.device("cpu")
    )
    handler = next(
        dep.fn
        for dep in app.fns.values()
        if getattr(dep.fn, "__name__", "") == "_update_inpainting_controls"
    )

    # Start from settings another method left behind
    updates = handler(method, "nunchaku", "INT8-ConvRot", "Q4_K_XL", 4, False, True)

    backend, _, _, diffusion, text_encoder, steps, *_ = updates
    for radio in (backend, diffusion, text_encoder):
        assert radio["value"] in [value for _, value in radio["choices"]]
    assert steps["value"] <= steps["maximum"]
    has_backend = method not in ("lama_manga", "opencv", "none")
    assert backend["visible"] is has_backend
    assert steps["visible"] is has_backend
    assert steps["interactive"] is (has_backend and method != "qwen_image_2_1_turbo")


@pytest.mark.parametrize("method", INPAINTING_METHODS)
@pytest.mark.parametrize("group_regions", [True, False])
def test_luminance_correction_is_offered_for_every_model(
    tmp_path, method, group_regions
):
    app = layout.create_layout(
        models_dir=tmp_path, fonts_base_dir=tmp_path, target_device=torch.device("cpu")
    )
    handler = next(
        dep.fn
        for dep in app.fns.values()
        if getattr(dep.fn, "__name__", "") == "_update_inpainting_controls"
    )

    luminance = handler(method, "sdcpp", "", "", 4, group_regions, True)[6]

    uses_model = method not in ("opencv", "none")
    assert luminance["visible"] is uses_model
    assert luminance["interactive"] is (uses_model and not group_regions)


@pytest.mark.parametrize("method", INPAINTING_METHODS)
@pytest.mark.parametrize("text_segmentation", [True, False])
@pytest.mark.parametrize("saved_choice", [True, False])
def test_preserve_art_is_offered_for_every_model_but_lama(
    tmp_path, monkeypatch, method, text_segmentation, saved_choice
):
    app = layout.create_layout(
        models_dir=tmp_path, fonts_base_dir=tmp_path, target_device=torch.device("cpu")
    )
    handler = next(
        dep.fn
        for dep in app.fns.values()
        if getattr(dep.fn, "__name__", "") == "_update_inpainting_controls"
    )

    monkeypatch.setattr(
        layout.settings_manager,
        "get_saved_settings",
        lambda: {"outside_text_flux_preserve_art": saved_choice},
    )

    preserve = handler(method, "sdcpp", "", "", 4, False, text_segmentation)[9]
    toggled = next(
        dep.fn
        for dep in app.fns.values()
        if getattr(dep.fn, "__name__", "") == "_update_preserve_art_interactivity"
    )(text_segmentation, method)

    uses_model = method not in ("opencv", "none")
    offered = uses_model and method != "lama_manga" and text_segmentation
    assert preserve["visible"] is uses_model
    assert preserve["interactive"] is offered
    assert toggled["interactive"] is offered
    # Restored from saved config when enabled; while disabled, shows whether the
    # art is kept anyway (LaMa erases only the letters the segmentation finds)
    lama_keeps_art = method == "lama_manga" and text_segmentation
    expected = (offered and saved_choice) or lama_keeps_art
    assert preserve["value"] is expected
    assert toggled["value"] is expected


def test_a_forced_backend_does_not_replace_the_saved_choice(tmp_path):
    """Fibo forces sd.cpp; the user's SDNQ and quant choices must survive it."""
    app = layout.create_layout(
        models_dir=tmp_path, fonts_base_dir=tmp_path, target_device=torch.device("cpu")
    )
    dep = next(
        dep
        for dep in app.fns.values()
        if getattr(dep.fn, "__name__", "") == "_update_inpainting_controls"
    )
    assert not any(isinstance(block, gr.State) for block in dep.outputs)

    fibo = dep.fn("fibo2_edit_turbo", "sdnq", "Q6_K", "Q6_K_XL", 4, False, True)
    assert fibo[0]["value"] == "sdcpp"
    assert fibo[3]["value"] == "INT8-ConvRot"

    flux3 = dep.fn("flux3_klein", "sdnq", "Q6_K", "Q6_K_XL", 4, False, True)
    assert flux3[0]["value"] == "sdnq"
    assert flux3[3]["value"] == "Q6_K"


def test_backend_and_quant_radios_store_only_user_choices(tmp_path):
    """Gradio fires .change for programmatic updates too, which would store forced values."""
    app = layout.create_layout(
        models_dir=tmp_path, fonts_base_dir=tmp_path, target_device=torch.device("cpu")
    )
    radio_ids = {
        block_id
        for block_id, block in app.blocks.items()
        if getattr(block, "label", None)
        in ("Backend", "Diffusion Model Quant", "Text Encoder Model Quant")
    }
    assert len(radio_ids) == 3

    events = [
        event
        for dep in app.fns.values()
        for block_id, event in dep.targets
        if block_id in radio_ids
    ]
    assert events
    assert set(events) == {"input"}


@pytest.mark.parametrize("text_free_only", [False, True])
@pytest.mark.parametrize("text_segmentation", [True, False])
@pytest.mark.parametrize("saved_choice", [True, False])
def test_missed_text_recovery_needs_text_segmentation_and_not_caption_boxes_only(
    tmp_path, monkeypatch, text_free_only, text_segmentation, saved_choice
):
    app = layout.create_layout(
        models_dir=tmp_path, fonts_base_dir=tmp_path, target_device=torch.device("cpu")
    )
    handler = next(
        dep.fn
        for dep in app.fns.values()
        if getattr(dep.fn, "__name__", "") == "_update_recovery_interactivity"
    )

    monkeypatch.setattr(
        layout.settings_manager,
        "get_saved_settings",
        lambda: {"outside_text_osb_recover_missed_text": saved_choice},
    )

    update = handler(text_free_only, text_segmentation)
    offered = text_segmentation and not text_free_only
    assert update["interactive"] is offered
    # Forced off while disabled, restored from saved config when enabled
    assert update["value"] is (offered and saved_choice)
