import io
import urllib.request

import pytest

from utils.download import download_file


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, fail_after_first_read: bool = False):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}
        self.fail_after_first_read = fail_after_first_read
        self.reads = 0

    def read(self, size=-1):
        self.reads += 1
        if self.fail_after_first_read and self.reads > 1:
            raise KeyboardInterrupt
        return super().read(3)


def test_download_writes_the_whole_file(monkeypatch, tmp_path):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req: FakeResponse(b"abcdef"))
    target = tmp_path / "model.bin"
    download_file(target, "https://example.invalid/model.bin")
    assert target.read_bytes() == b"abcdef"


def test_interrupted_download_leaves_no_file_behind(monkeypatch, tmp_path):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req: FakeResponse(b"abcdef", fail_after_first_read=True),
    )
    target = tmp_path / "model.bin"
    with pytest.raises(KeyboardInterrupt):
        download_file(target, "https://example.invalid/model.bin")
    assert not target.exists()

    # The next run downloads again instead of trusting a half-written file
    monkeypatch.setattr(urllib.request, "urlopen", lambda req: FakeResponse(b"abcdef"))
    download_file(target, "https://example.invalid/model.bin")
    assert target.read_bytes() == b"abcdef"
