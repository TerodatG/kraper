# KraperModel/tests/test_bundled_artifacts.py                         v001
# Автономность: данные и веса лежат в репозитории Kraper.
# Последнее изменение: первый вариант — проверка комплекта.

import json

from kraper_model.config import DATA_DIR, MODELS_DIR
from kraper_model.language.linear_intent import LinearIntentRouter

REQUIRED_DATA = (
    "intents.json",
    "eval_set.json",
    "personality_matrices.json",
    "dataset/classification.json",
    "dataset/pairs.json",
)
REQUIRED_MODELS = (
    "linear_intent.json",
    "tokenizer.json",
    "seq2seq.pt",
)


def test_bundled_data_and_models_exist() -> None:
    """Свежий клон содержит всё необходимое для запуска."""

    for name in REQUIRED_DATA:
        path = DATA_DIR / name
        assert path.is_file(), f"нет {path}"
        assert path.stat().st_size > 0

    for name in REQUIRED_MODELS:
        path = MODELS_DIR / name
        assert path.is_file(), f"нет {path}"
        assert path.stat().st_size > 0


def test_bundled_router_matches_intents() -> None:
    """Роутер собран ровно под 100 тем из intents.json."""

    router = LinearIntentRouter.load(
        MODELS_DIR / "linear_intent.json"
    )

    assert router is not None

    document = json.loads(
        (DATA_DIR / "intents.json").read_text(encoding="utf-8")
    )
    tags = {
        intent["tag"]
        for intent in document["intents"]
    }

    assert len(tags) == 100
    assert set(router.labels) == tags
    assert router.predict("привет") is not None
