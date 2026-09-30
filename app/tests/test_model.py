import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

pytestmark = pytest.mark.skipif(
    not os.path.exists("models/distilbert_finetuned"),
    reason="Model not found. Run train.py first.",
)

from app.model import Classifier


@pytest.fixture(scope="module")
def classifier():
    return Classifier()


def test_model_loads(classifier):
    assert classifier is not None
    assert len(classifier.labels) > 0
    assert all(isinstance(label, str) for label in classifier.labels)


def test_predict_returns_tuple(classifier):
    result = classifier.predict("play music")
    assert isinstance(result, tuple)
    assert len(result) == 2
    intent, confidence = result
    assert isinstance(intent, str)
    assert isinstance(confidence, float)


def test_predict_returns_valid_label(classifier):
    intent, _ = classifier.predict("play some music")
    assert intent in classifier.labels


def test_confidence_in_range(classifier):
    _, confidence = classifier.predict("play music")
    assert 0.0 <= confidence <= 1.0


def test_model_accuracy_on_samples(classifier):
    samples = [
        ("play some music", "play_music"),
        ("navigate to hospital", "navigate"),
        ("call mom", "make_call"),
        ("send a message", "send_message"),
        ("turn on the AC", "set_climate"),
    ]
    correct = 0
    for text, expected in samples:
        intent, _ = classifier.predict(text)
        if intent == expected:
            correct += 1
    accuracy = correct / len(samples)
    assert accuracy >= 0.8, f"Accuracy dropped to {accuracy:.0%}"


def test_high_confidence_on_clear_input(classifier):
    _, confidence = classifier.predict("play some music")
    assert confidence > 0.5


def test_empty_input_raises_error(classifier):
    with pytest.raises(ValueError):
        classifier.predict("")


def test_whitespace_only_input_raises_error(classifier):
    with pytest.raises(ValueError):
        classifier.predict("   ")


def test_very_long_input_handled(classifier):
    long_text = "please " * 200 + "play music"
    intent, confidence = classifier.predict(long_text)
    assert intent in classifier.labels


def test_special_characters_handled(classifier):
    intent, _ = classifier.predict("play @#$% music!!! 123")
    assert intent in classifier.labels


def test_mixed_case_input(classifier):
    intent, _ = classifier.predict("PLAY SOME MUSIC")
    assert intent in classifier.labels