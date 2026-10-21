import importlib.util

import pytest

from app.models.deep_learning import DEEP_LEARNING_MODELS, DeepSequenceRegressor, build_model


def test_sequence_model_builders_are_registered():
    assert DEEP_LEARNING_MODELS == ("lstm", "gru", "transformer")
    with pytest.raises(ValueError, match="Unsupported deep-learning model"):
        DeepSequenceRegressor("unknown")


def test_deep_learning_dependency_is_optional_and_reports_install_hint():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("PyTorch is installed; architecture smoke tests belong to the optional DL environment")
    with pytest.raises(RuntimeError, match="requirements-dl.txt"):
        build_model("lstm", input_size=4)
