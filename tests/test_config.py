# KraperModel/tests/test_config.py                                   v002
# Тесты модели: пути из окружения и выбор устройства.

from pathlib import Path

from kraper_model import config


def test_empty_env_falls_back(monkeypatch, tmp_path) -> None:
    """Пустой env-путь подменяется значением по умолчанию."""

    monkeypatch.setenv("KRAPER_TEST_PATH", "")

    assert config._path("KRAPER_TEST_PATH", tmp_path) == (
        tmp_path.resolve()
    )


def test_env_path_wins(monkeypatch, tmp_path) -> None:
    """Непустой env-путь имеет приоритет."""

    target = tmp_path / "custom"
    monkeypatch.setenv("KRAPER_TEST_PATH", str(target))

    assert config._path("KRAPER_TEST_PATH", tmp_path) == (
        target.resolve()
    )


def test_module_paths_are_absolute() -> None:
    """Основные пути модели всегда абсолютные."""

    for value in (
        config.BASE_DIR,
        config.DATA_DIR,
        config.MODELS_DIR,
        config.LOGS_DIR,
    ):
        assert isinstance(value, Path)
        assert value.is_absolute()


def test_resolve_device_respects_cpu(monkeypatch) -> None:
    """KRAPER_DEVICE=cpu не должен включать CUDA."""

    monkeypatch.setattr(config, "DEVICE", "cpu")

    assert config.resolve_device().type == "cpu"
