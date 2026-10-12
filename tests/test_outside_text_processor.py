import pytest

from core.outside_text_processor import _is_page_number


@pytest.mark.parametrize("text", ["１２", "Ｐａｇｅ２０", "ｐ．２０", "－３５－", "20"])
def test_page_numbers_are_recognized_in_full_width_ocr_output(text):
    assert _is_page_number(text)


@pytest.mark.parametrize("text", ["ノア", "これで３日は食えるな！", ""])
def test_ordinary_text_is_not_a_page_number(text):
    assert not _is_page_number(text)
