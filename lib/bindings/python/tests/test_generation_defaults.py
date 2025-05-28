import importlib
import json
from pathlib import Path

import pytest

from dynamo.llm.generation_defaults import load_default_sampling_params


@pytest.mark.parametrize("model_path", ["some-completely-made-up-model-id-xyz", "non/existent/path"])
def test_missing_generation_config_returns_empty(model_path):
    """Helper should gracefully fall back to an empty dict when the file is missing."""
    assert load_default_sampling_params(model_path) == {}


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers not installed")
def test_local_generation_config(tmp_path: Path):
    """Loading from a local directory containing *generation_config.json* should return its contents."""
    cfg = {"temperature": 0.7, "top_p": 0.9, "top_k": 50}
    (tmp_path / "generation_config.json").write_text(json.dumps(cfg))

    result = load_default_sampling_params(str(tmp_path))

    # The helper might remove ``None`` values but must preserve the supplied keys.
    for k, v in cfg.items():
        assert result.get(k) == v 