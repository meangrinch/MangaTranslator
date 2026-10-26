import json

import pytest

from ui import settings_manager


@pytest.mark.parametrize("old_mode", ["cache-dit", "taylorseer", "dbcache"])
def test_saved_dit_cache_modes_load_as_step_skip(tmp_path, monkeypatch, old_mode):
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"outside_text_flux_sdcpp_cache_mode": old_mode}), encoding="utf-8"
    )
    monkeypatch.setattr(settings_manager, "CONFIG_FILE", config_file)

    settings = settings_manager.get_saved_settings()

    assert settings["outside_text_flux_sdcpp_cache_mode"] == "step-skip"
