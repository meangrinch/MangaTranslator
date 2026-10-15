import threading
import time

import numpy as np
import pytest
import torch
from PIL import Image

from core import pipeline
from core.config import MangaTranslatorConfig, TranslationConfig
from core.ml import sdcpp_server
from core.services import translation
from utils import cancellation
from utils.cancellation import CancellationManager
from utils.exceptions import CancellationError


@pytest.fixture
def manager():
    manager = CancellationManager()
    cancellation.activate(manager)
    yield manager
    cancellation.activate(CancellationManager())


def test_cancel_before_work_raises_cancellation_not_translation_error(tmp_path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8)).save(image_path)
    cancelled = CancellationManager()
    cancelled.cancel()
    config = MangaTranslatorConfig(
        yolo_model_path="model.pt", device=torch.device("cpu")
    )

    with pytest.raises(CancellationError):
        pipeline.translate_and_render(
            image_path, config, cancellation_manager=cancelled
        )


def test_batch_cancelled_before_any_page_leaves_no_output_folder(tmp_path):
    input_dir = tmp_path / "in"
    (input_dir / "ch1").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(input_dir / "ch1" / "page.png")
    output_dir = tmp_path / "out" / "run"
    cancelled = CancellationManager()
    cancelled.cancel()
    config = MangaTranslatorConfig(
        yolo_model_path="model.pt", device=torch.device("cpu")
    )

    with pytest.raises(CancellationError):
        pipeline.batch_translate_images(
            input_dir,
            config,
            output_dir=output_dir,
            preserve_structure=True,
            cancellation_manager=cancelled,
        )
    assert not output_dir.exists()


def test_llm_call_is_abandoned_while_the_request_is_still_in_flight(
    monkeypatch, manager
):
    release = threading.Event()
    finished = threading.Event()

    def blocking_endpoint(**_):
        manager.cancel()
        release.wait(5)
        finished.set()
        return "late"

    monkeypatch.setitem(
        translation._PROVIDER_ENDPOINTS,
        "Google",
        ("google_api_key", "Google", blocking_endpoint),
    )
    config = TranslationConfig(provider="Google", google_api_key="key")

    try:
        with pytest.raises(CancellationError):
            translation._call_llm_endpoint(config, [], "prompt")
        assert not finished.is_set()
    finally:
        release.set()


def test_page_cleaning_lets_a_cancel_through_instead_of_falling_back(monkeypatch):
    def cancelled_clean(*_, **__):
        raise CancellationError("Process cancelled by user.")

    monkeypatch.setattr(pipeline, "clean_speech_bubbles", cancelled_clean)
    config = MangaTranslatorConfig(
        yolo_model_path="model.pt", device=torch.device("cpu")
    )

    with pytest.raises(CancellationError):
        pipeline._clean_speech_bubbles_for_page(
            Image.new("RGB", (8, 8)),
            [],
            config,
            torch.device("cpu"),
            1.0,
            False,
            np.zeros((8, 8, 3), dtype=np.uint8),
        )


def test_sdcpp_poll_stops_and_cancels_the_queued_job(monkeypatch, manager):
    requests = []

    def fake_json_request(url, payload=None, timeout=None):
        requests.append(url)
        if url.endswith("/img_gen"):
            return {"id": "job1"}
        if url.endswith("/cancel"):
            return {"status": "cancelled"}
        manager.cancel()
        return {"status": "queued"}

    monkeypatch.setattr(sdcpp_server, "_json_request", fake_json_request)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    with pytest.raises(CancellationError):
        sdcpp_server.run_image_job({"url": "http://sd"}, {}, timeout_sec=1)
    assert requests[-1] == "http://sd/sdcpp/v1/jobs/job1/cancel"
