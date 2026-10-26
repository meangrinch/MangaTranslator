import time

import pytest

from core.ml import sdcpp_server
from utils.exceptions import ModelError


class FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = 1


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)


def _fake_requests(monkeypatch, on_poll):
    def fake_json_request(url, payload=None, timeout=None):
        if url.endswith("/img_gen"):
            return {"id": "job1"}
        return on_poll()

    monkeypatch.setattr(sdcpp_server, "_json_request", fake_json_request)


def test_job_releases_torch_cache_before_submitting(monkeypatch, no_sleep):
    events = []
    monkeypatch.setattr(sdcpp_server, "empty_cache", lambda: events.append("release"))
    monkeypatch.setattr(sdcpp_server, "_image_from_result", lambda result: "image")

    def fake_json_request(url, payload=None, timeout=None):
        events.append(url.rsplit("/", 1)[-1])
        return {"status": "completed"}

    monkeypatch.setattr(sdcpp_server, "_json_request", fake_json_request)

    assert sdcpp_server.run_image_job({"url": "http://sd"}, {}) == "image"
    assert events == ["release", "img_gen"]


def test_failed_job_reports_the_server_logged_reason(monkeypatch, tmp_path, no_sleep):
    log_path = tmp_path / "server.log"
    log_path.write_text("[W] unrelated warning from model load\n", encoding="utf-8")

    def on_poll():
        # sd.cpp logs the reason just before it marks the job failed
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                "[W] model manager cannot make enough memory available on CUDA0: "
                "need 512.00 MB device / 1535.71 MB budget, available 325.85 MB "
                "device / unlimited budget --- model_manager.cpp:2018\n"
                "[E] flux3 segment 1/1 (graph) failed during weight preparation "
                "--- ggml_runner.cpp:899\n"
            )
        return {
            "status": "failed",
            "error": {"message": "generate_image returned no results"},
        }

    _fake_requests(monkeypatch, on_poll)
    server = {"url": "http://sd", "log_path": log_path, "process": FakeProcess()}

    with pytest.raises(ModelError) as excinfo:
        sdcpp_server.run_image_job(server, {}, timeout_sec=60)
    message = str(excinfo.value)
    assert "generate_image returned no results" in message
    assert "available 325.85 MB device" in message
    assert "unrelated warning" not in message
    assert "model_manager.cpp" not in message


def test_job_fails_fast_when_server_exits(monkeypatch, tmp_path, no_sleep):
    log_path = tmp_path / "server.log"
    log_path.write_text("[I] listening\n", encoding="utf-8")
    process = FakeProcess()

    def on_poll():
        process.returncode = 3
        raise ConnectionRefusedError("refused")

    _fake_requests(monkeypatch, on_poll)
    server = {"url": "http://sd", "log_path": log_path, "process": process}

    with pytest.raises(ModelError, match="exited"):
        sdcpp_server.run_image_job(server, {}, timeout_sec=60)


def test_job_fails_fast_on_fatal_log_line_while_server_hangs(
    monkeypatch, tmp_path, no_sleep
):
    log_path = tmp_path / "server.log"
    log_path.write_text("[I] listening\n", encoding="utf-8")
    process = FakeProcess()
    polls = []

    def on_poll():
        polls.append(1)
        if len(polls) == 1:
            # Model load buffers came before the job; only new lines count
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    "[E] ggml: CUDA error: out of memory\n"
                    "ggml-cuda.cu:111: CUDA error\n"
                )
            return {"status": "generating"}
        # A crashing server on Windows stays alive while it writes a crash dump
        raise TimeoutError("timed out")

    _fake_requests(monkeypatch, on_poll)
    server = {"url": "http://sd", "log_path": log_path, "process": process}

    with pytest.raises(ModelError, match="CUDA error: out of memory"):
        sdcpp_server.run_image_job(server, {}, timeout_sec=60)
    assert process.killed
    assert len(polls) <= 2
