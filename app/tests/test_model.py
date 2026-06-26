import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pytestmark = pytest.mark.skipif(
    not os.path.exists("models/distilbert_finetuned"),
    reason="Model not found. Run train.py first."
)

from model import Classifier


@pytest.fixture
def classifier():
    return Classifier()


def test_model_loads(classifier):
    assert classifier is not None
    assert len(classifier.labels) > 0


def test_predict_returns_tuple(classifier):
    intent, confidence = classifier.predict("test")
    assert isinstance(intent, str)
    assert 0.0 <= confidence <= 1.0


def test_empty_input_raises_error(classifier):
    with pytest.raises(ValueError):
        classifier.predict("")