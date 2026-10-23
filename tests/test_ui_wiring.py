import re

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
