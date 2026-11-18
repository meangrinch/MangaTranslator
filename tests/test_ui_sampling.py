import pytest

from ui import callbacks, settings_manager, utils


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


@pytest.mark.parametrize("model", ["gemini-4-hydrogen", "gemini-4-neon"])
def test_gemini_4_hydrogen_and_neon_offer_xhigh_and_default_to_high(model):
    assert model in settings_manager.PROVIDER_MODELS["Google"]
    assert utils.get_reasoning_effort_config("Google", model) == (
        True,
        ["xhigh", "high", "medium", "low"],
        "high",
    )
    assert (
        utils.get_reasoning_effort_config("OpenRouter", f"google/{model}")[2] == "high"
    )
    assert (
        utils.format_thinking_status("OpenRouter", f"google/{model}", None)
        == " (thinking: high)"
    )
    assert (
        utils.get_reasoning_effort_config("OpenRouter", "google/gemini-4-argon")[2]
        == "medium"
    )


def test_qwen4_max_offers_high_default_and_no_medium():
    assert utils.get_reasoning_effort_config("QwenCloud", "qwen4-max") == (
        True,
        ["xhigh", "high", "low", "none"],
        "high",
    )
    assert utils.get_reasoning_effort_config("OpenCode", "qwen4-max") == (
        True,
        ["xhigh", "high", "low", "none"],
        "high",
    )
    assert utils.get_reasoning_effort_config("QwenCloud", "qwen3.8-max") == (
        True,
        ["xhigh", "medium", "low", "none"],
        "xhigh",
    )


@pytest.mark.parametrize(
    "model",
    ["gpt-6.2-sol", "gpt-6.2-astra", "gpt-6.2-sol-pro", "gpt-6.2-astra-pro"],
)
def test_gpt62_models_omit_none_effort_and_disable_sampling(model):
    expected_efforts = ["max", "xhigh", "high", "medium", "low"]
    for provider in ("OpenAI", "OpenCode", "OpenAI-Compatible"):
        assert utils.get_reasoning_effort_config(provider, model) == (
            True,
            expected_efforts,
            "medium",
        )
    assert utils.get_verbosity_config("OpenAI", model) == (
        True,
        ["high", "medium", "low"],
        "low",
    )
    assert utils.get_sampling_slider_interactivity("OpenAI", model, "low") == (
        False,
        False,
        False,
    )
    assert utils.get_sampling_slider_interactivity(
        "OpenRouter", f"openai/{model}", "none"
    ) == (False, False, False)
