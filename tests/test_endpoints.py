import copy
import time
from types import SimpleNamespace

import pytest
import requests

from utils.endpoints import (
    call_anthropic_endpoint,
    call_deepseek_endpoint,
    call_gemini_endpoint,
    call_meta_model_endpoint,
    call_moonshot_endpoint,
    call_openai_compatible_endpoint,
    call_openai_endpoint,
    call_openrouter_endpoint,
    call_qwencloud_endpoint,
    call_xai_endpoint,
)
from utils.endpoints._http import post_with_retries
from utils.endpoints.openai_compatible import build_openai_compatible_url
from utils.exceptions import TranslationError
from utils.model_metadata import anthropic_model_flags

IMAGE = {"inline_data": {"mime_type": "image/png", "data": "AAAA"}}

# Raw Responses API bodies carry no top-level `output_text`; that field is SDK-only
RESPONSES_BODY = {
    "error": None,
    "status": "completed",
    "output": [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "thinking..."}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "1: Hello", "annotations": []}],
        },
    ],
}


@pytest.fixture
def post(monkeypatch):
    """Fake requests.post: records the request, replies from `post.queue` then `post.body`."""

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return fake.queue.pop(0) if fake.queue else fake.body

    def fake(url, json=None, **kwargs):
        fake.url, fake.sent = url, copy.deepcopy(json)
        return FakeResponse()

    fake.body = {}
    fake.queue = []
    monkeypatch.setattr(requests, "post", fake)
    return fake


@pytest.mark.parametrize(
    ("call", "model"),
    [
        (call_openai_endpoint, "gpt-5.5"),
        (call_deepseek_endpoint, "deepseek-flash"),
        (call_qwencloud_endpoint, "qwen3.8-max"),
        (call_meta_model_endpoint, "muse-spark-1.3"),
        (call_xai_endpoint, "grok-4.7"),
    ],
)
def test_responses_api_text_is_read_from_output_items(post, call, model):
    post.body = RESPONSES_BODY
    assert call("key", model, [{"text": "go"}], {}) == "1: Hello"


@pytest.mark.parametrize(
    ("call", "model"),
    [(call_deepseek_endpoint, "deepseek-flash"), (call_xai_endpoint, "grok-4.7")],
)
def test_string_error_field_is_reported(post, call, model):
    post.body = {"error": "boom"}
    with pytest.raises(TranslationError, match="boom"):
        call("key", model, [{"text": "go"}], {})


def test_chat_completions_string_error_field_is_reported(post, monkeypatch):
    monkeypatch.setattr(
        "utils.endpoints.openrouter.openrouter_is_reasoning_model",
        lambda *args, **kwargs: False,
    )
    post.body = {"error": "boom"}
    with pytest.raises(TranslationError, match="boom"):
        call_openrouter_endpoint("key", "openai/gpt-5.5", [{"text": "go"}], {})
    with pytest.raises(TranslationError, match="boom"):
        call_openai_compatible_endpoint(
            "http://localhost:1/v1", None, "local-model", [{"text": "go"}], {}
        )


def test_responses_api_message_items_stay_on_separate_lines(post):
    def message(*texts):
        content = [{"type": "output_text", "text": t} for t in texts]
        return {"type": "message", "content": content}

    post.body = {
        "output": [
            message("I'll look that up."),
            {"type": "web_search_call", "status": "completed"},
            message("1: Hel", "lo"),
        ]
    }
    text = call_openai_endpoint("key", "gpt-5.5", [{"text": "go"}], {})
    assert text == "I'll look that up.\n1: Hello"


def test_responses_api_falls_back_to_top_level_output_text(post):
    post.body = {"output_text": " 1: Hello "}
    assert call_qwencloud_endpoint("key", "qwen3.8-max", [{"text": "go"}], {}) == (
        "1: Hello"
    )


def test_deepseek_sends_responses_style_reasoning(post):
    post.body = RESPONSES_BODY
    config = {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}
    call_deepseek_endpoint("key", "deepseek-flash", [{"text": "go"}], config)
    assert post.sent["reasoning"] == {"effort": "max"}
    assert not {"thinking", "reasoning_effort", "tools"} & post.sent.keys()

    call_deepseek_endpoint(
        "key", "deepseek-flash", [{"text": "go"}], {"thinking": {"type": "disabled"}}
    )
    assert post.sent["reasoning"] == {"effort": "none"}


def test_meta_uses_responses_api(post):
    post.body = RESPONSES_BODY
    call_meta_model_endpoint(
        "key",
        "muse-spark-1.3",
        [{"text": "Crop 1:"}, IMAGE, {"text": "go"}],
        {"max_tokens": 4096, "reasoning_effort": "high"},
        system_prompt="sys",
        enable_web_search=True,
    )
    assert post.url == "https://api.meta.ai/v1/responses"
    assert post.sent["instructions"] == "sys"
    assert post.sent["max_output_tokens"] == 4096
    assert post.sent["reasoning"] == {"effort": "high"}
    assert post.sent["tools"] == [{"type": "web_search"}]
    assert post.sent["input"][0]["content"] == [
        {"type": "input_text", "text": "Crop 1:"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        {"type": "input_text", "text": "go"},
    ]


@pytest.mark.parametrize(
    ("part_level", "request_wide", "detail"),
    [
        ("MEDIA_RESOLUTION_LOW", "high", "low"),
        ("MEDIA_RESOLUTION_MEDIUM", "low", "high"),
        (None, "low", "low"),
        (None, "medium", "high"),
    ],
)
def test_xai_image_detail_prefers_the_images_own_resolution(
    post, part_level, request_wide, detail
):
    post.body = RESPONSES_BODY
    image = {**IMAGE, "media_resolution": {"level": part_level}}
    call_xai_endpoint(
        "key", "grok-4.7", [image, {"text": "go"}], {"media_resolution": request_wide}
    )
    assert post.sent["input"][0]["content"][0]["detail"] == detail


def test_anthropic_keeps_answer_text_after_tool_blocks(post):
    post.body = {
        "content": [
            {"type": "text", "text": "I'll look that up."},
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
            {"type": "text", "text": "1: Hel"},
            {"type": "text", "text": "lo"},
        ]
    }
    text = call_anthropic_endpoint("key", "claude-sonnet-4-6", [{"text": "go"}], {})
    assert text == "I'll look that up.\n\n1: Hello"


@pytest.mark.parametrize("thinking_type", ["adaptive", "enabled"])
def test_anthropic_drops_sampling_params_while_thinking(post, thinking_type):
    post.body = {"content": [{"type": "text", "text": "ok"}]}
    config = {
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_k": 0,
        "thinking_type": thinking_type,
        "reasoning_effort": "low",
    }
    call_anthropic_endpoint("key", "claude-sonnet-4-6", [{"text": "go"}], config)
    assert post.sent["thinking"]["type"] == thinking_type
    assert not {"temperature", "top_k"} & post.sent.keys()
    if thinking_type == "enabled":
        # 20% of 4096 is below the API minimum
        assert post.sent["thinking"]["budget_tokens"] == 1024


def test_anthropic_skips_thinking_when_no_budget_fits_under_max_tokens(post):
    post.body = {"content": [{"type": "text", "text": "ok"}]}
    config = {"max_tokens": 1024, "thinking_type": "enabled", "reasoning_effort": "low"}
    call_anthropic_endpoint(
        "key", "claude-sonnet-4-5-20250929", [{"text": "go"}], config
    )
    assert "thinking" not in post.sent


def test_anthropic_resumes_paused_turn(post):
    paused = [
        {"type": "text", "text": "Searching."},
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
    ]
    post.queue = [
        {"stop_reason": "pause_turn", "content": paused},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "1: Hello"}]},
    ]
    text = call_anthropic_endpoint(
        "key", "claude-sonnet-4-6", [{"text": "go"}], {}, enable_web_search=True
    )
    assert text == "Searching.\n1: Hello"
    assert post.queue == []
    # The paused turn goes back unchanged, with the same tools
    assert post.sent["messages"][1:] == [{"role": "assistant", "content": paused}]
    assert post.sent["tools"][0]["name"] == "web_search"


def test_anthropic_keeps_paused_text_when_continuation_is_empty(post):
    post.queue = [
        {"stop_reason": "pause_turn", "content": [{"type": "text", "text": "1: Hi"}]},
        {"stop_reason": "end_turn", "content": []},
    ]
    text = call_anthropic_endpoint("key", "claude-sonnet-4-6", [{"text": "go"}], {})
    assert text == "1: Hi"


def test_anthropic_returns_partial_text_when_turn_keeps_pausing(post):
    post.body = {
        "stop_reason": "pause_turn",
        "content": [{"type": "text", "text": "part"}],
    }
    text = call_anthropic_endpoint(
        "key", "claude-sonnet-4-6", [{"text": "go"}], {}, max_retries=2
    )
    assert text == "partpartpart"


@pytest.mark.parametrize(
    ("model", "tool_type"),
    [
        ("claude-opus-5-5", "web_search_20260209"),
        ("claude-sonnet-4-6", "web_search_20260209"),
        ("claude-opus-4-5-20251101", "web_search_20250305"),
        ("claude-haiku-4-5-20251001", "web_search_20250305"),
    ],
)
def test_anthropic_web_search_tool_version(post, model, tool_type):
    post.body = {"content": [{"type": "text", "text": "ok"}]}
    config = {"_metadata": anthropic_model_flags(model)}
    call_anthropic_endpoint(
        "key", model, [{"text": "go"}], config, enable_web_search=True
    )
    assert post.sent["tools"] == [{"type": tool_type, "name": "web_search"}]


def test_anthropic_keeps_sampling_params_without_thinking(post):
    post.body = {"content": [{"type": "text", "text": "ok"}]}
    call_anthropic_endpoint(
        "key", "claude-sonnet-4-6", [{"text": "go"}], {"temperature": 0.1, "top_k": 5}
    )
    assert post.sent["temperature"] == 0.1
    assert post.sent["top_k"] == 5


def test_gemini_keeps_answer_text_after_code_execution(post):
    post.body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "hidden", "thought": True},
                        {"text": "Let me zoom in."},
                        {"executableCode": {"language": "PYTHON", "code": "..."}},
                        {"codeExecutionResult": {"outcome": "OUTCOME_OK"}},
                        {"text": "1: Hello"},
                    ]
                }
            }
        ]
    }
    text = call_gemini_endpoint("key", "gemini-3.6-flash", [{"text": "go"}], {})
    assert text == "Let me zoom in.\n\n1: Hello"


@pytest.mark.parametrize(
    ("model", "effort", "sends_temperature"),
    [
        ("anthropic/claude-sonnet-4.5", "low", False),
        ("anthropic/claude-sonnet-4.5", "none", True),
        ("anthropic/claude-sonnet-4.6", "auto", False),
        ("anthropic/claude-sonnet-4.6", "none", True),
    ],
)
def test_openrouter_drops_claude_temperature_while_thinking(
    post, monkeypatch, model, effort, sends_temperature
):
    monkeypatch.setattr(
        "utils.endpoints.openrouter.openrouter_is_reasoning_model",
        lambda *args, **kwargs: True,
    )
    post.body = {"choices": [{"message": {"content": "ok"}}]}
    config = {
        "temperature": 0.1,
        "reasoning_effort": effort,
        "_metadata": {"is_anthropic_model": True, **anthropic_model_flags(model)},
    }
    call_openrouter_endpoint("key", model, [{"text": "go"}], config)
    assert ("temperature" in post.sent) is sends_temperature


@pytest.mark.parametrize("model", ["kimi-k2.6", "kimi-k3"])
def test_moonshot_leaves_fixed_sampling_params_unset(post, model):
    post.body = {"choices": [{"message": {"content": "ok"}}]}
    call_moonshot_endpoint(
        "key", model, [{"text": "go"}], {"temperature": 0.1, "top_p": 1.0}
    )
    assert not {"temperature", "top_p", "tools"} & post.sent.keys()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"choices": [{"message": {"content": " 1: Hello "}}]}, "1: Hello"),
        ({"choices": [{"message": {"content": None}}]}, ""),
        ({"choices": [{"finish_reason": "content_filter"}]}, ""),
        ({"choices": []}, None),
    ],
)
def test_chat_reply_tells_an_empty_answer_from_no_answer(post, body, expected):
    post.body = body
    assert call_moonshot_endpoint("key", "kimi-k3", [{"text": "go"}], {}) == expected


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "https://res.openai.azure.com",
            "https://res.openai.azure.com/openai/v1/chat/completions",
        ),
        (
            "https://res.openai.azure.com/openai/v1/",
            "https://res.openai.azure.com/openai/v1/chat/completions",
        ),
        (
            "https://res.openai.azure.com/openai/deployments/dep/chat/completions?api-version=2024-06-01",
            "https://res.openai.azure.com/openai/v1/chat/completions",
        ),
        (
            "https://gw.azure-api.net/team/v1",
            "https://gw.azure-api.net/team/openai/v1/chat/completions",
        ),
        (
            "https://res.openai.azure.com/v1/chat/completions",
            "https://res.openai.azure.com/openai/v1/chat/completions",
        ),
        (
            "https://gw.azure-api.net/team?subscription-key=abc&api-version=2024-06-01",
            "https://gw.azure-api.net/team/openai/v1/chat/completions?subscription-key=abc",
        ),
        ("http://localhost:8080/v1", "http://localhost:8080/v1/chat/completions"),
    ],
)
def test_build_openai_compatible_url(base_url, expected):
    assert build_openai_compatible_url(base_url) == expected


@pytest.mark.parametrize(
    ("is_reasoning", "token_key"),
    [(True, "max_completion_tokens"), (False, "max_tokens")],
)
def test_openai_compatible_token_limit_param(post, is_reasoning, token_key):
    post.body = {"choices": [{"message": {"content": "ok"}}]}
    config = {"max_tokens": 4096, "_metadata": {"is_openai_reasoning": is_reasoning}}
    call_openai_compatible_endpoint(
        "http://localhost:1/v1", None, "gpt-5.5", [{"text": "go"}], config
    )
    assert post.sent[token_key] == 4096
    assert len({"max_tokens", "max_completion_tokens"} & post.sent.keys()) == 1


class FakeReply:
    text = "body"

    def __init__(self, reply):
        is_status = isinstance(reply, int)
        self.status_code, self.body = (reply, None) if is_status else (200, reply)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self.body


@pytest.fixture
def net(monkeypatch):
    """Fake requests.post replaying `net.queue` (status codes, JSON bodies, or exceptions); records sleeps."""
    state = SimpleNamespace(queue=[], sleeps=[])

    def fake(url, **kwargs):
        reply = state.queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeReply(reply)

    monkeypatch.setattr(requests, "post", fake)
    monkeypatch.setattr(time, "sleep", state.sleeps.append)
    return state


def retrying_post(**kwargs):
    return post_with_retries(
        "http://x", {}, {}, lambda result: result["text"], label="Test", **kwargs
    )


def test_retry_backs_off_on_429_then_returns_parsed_body(net):
    net.queue = [429, 429, {"text": "ok"}]
    assert retrying_post() == "ok"
    assert net.sleeps == [1.0, 2.0]


# core/pipeline.py matches on "429" / "rate limit" in these messages
def test_retry_gives_up_on_429_after_max_retries(net):
    net.queue = [429] * 3
    with pytest.raises(
        TranslationError,
        match="Test API HTTP Error: Rate limited after 3 attempts: body",
    ):
        retrying_post(max_retries=2)
    assert net.sleeps == [1.0, 2.0]


def test_other_http_errors_raise_at_once_with_hint(net):
    net.queue = [401]
    with pytest.raises(
        TranslationError,
        match=r"Test API HTTP Error: Status 401: body \(Check API key\)",
    ):
        retrying_post(status_hints={401: "Check API key"})
    assert net.sleeps == []


def test_connection_errors_retry_then_raise(net):
    down = requests.exceptions.ConnectionError("down")
    net.queue = [down, {"text": "ok"}]
    assert retrying_post() == "ok"

    net.queue = [down, down]
    with pytest.raises(
        TranslationError, match="Test API Connection Error after retries: down"
    ):
        retrying_post(max_retries=1)


def test_unreadable_body_raises_translation_error(net):
    net.queue = [{"other": 1}]
    with pytest.raises(
        TranslationError, match="Error processing successful Test API response"
    ):
        retrying_post()
