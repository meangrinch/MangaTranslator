import pytest

from ui import callbacks, utils


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("Anthropic", "claude-opus-4-5-20251101"),
        ("OpenRouter", "anthropic/claude-opus-4.5"),
    ],
)
@pytest.mark.parametrize(
    ("reasoning_effort", "temp_interactive"), [("none", True), ("low", False)]
)
def test_sampling_toggle_respects_reasoning_effort(
    provider, model, reasoning_effort, temp_interactive
):
    temp = callbacks.handle_use_custom_sampling_change(
        provider, model, reasoning_effort, True
    )[0]
    assert temp["interactive"] is temp_interactive
    assert temp["visible"] is temp_interactive


def test_model_change_sliders_follow_the_effort_it_selects(monkeypatch):
    monkeypatch.setattr(
        utils,
        "get_reasoning_effort_config",
        lambda provider, model_name: (True, ["low", "none"], "none"),
    )
    temp = callbacks.handle_model_change("Anthropic", "claude-opus-4-5-20251101", 0.1)[
        0
    ]
    assert temp["interactive"] is True
