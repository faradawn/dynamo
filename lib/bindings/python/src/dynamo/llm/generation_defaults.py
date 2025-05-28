from __future__ import annotations

"""Utility helpers to load default sampling (generation) parameters for a model.

This helper offers a single public function ``load_default_sampling_params`` which
tries to read ``generation_config.json`` from either a local model directory or a
HuggingFace Hub repository and returns a plain ``dict`` of key/value pairs.  The
resulting dictionary can be fed directly into backend-specific ``SamplingParams``
constructors (e.g. *vLLM*, *TensorRT-LLM*, *SGLang*).

If no `generation_config.json` can be located (or if the *transformers* package
is not available) an **empty dict** is returned.

Support for GGUF metadata is sketched but optional – the function will fall back
silently in environments where the ``gguf`` python package is not installed.
"""

from pathlib import Path
from typing import Any, Dict
import logging

__all__ = ["load_default_sampling_params"]


def _from_transformers(model_path: str) -> Dict[str, Any]:
    """Try to fetch generation defaults using *transformers*' GenerationConfig.

    Returns an empty dict if the model (or required file) cannot be found.
    """

    try:
        from transformers import GenerationConfig  # type: ignore
    except ModuleNotFoundError:
        logging.debug("transformers not installed – skipping GenerationConfig load")
        return {}

    try:
        # ``from_pretrained`` works for both local directories / files and hub ids.
        gen_cfg = GenerationConfig.from_pretrained(model_path)  # type: ignore[arg-type]
    except EnvironmentError as exc:
        # This is the error raised when no generation_config.json exists for the model.
        logging.debug("No generation_config.json for %s – %s", model_path, exc)
        return {}
    except Exception as exc:  # pragma: no cover – defensive
        # Any other failure, fall back to empty defaults but surface the message.
        logging.error("Failed to load generation_config for %s – %s", model_path, exc)
        return {}

    # Newer versions expose ``to_diff_dict`` which strips unchanged values.  Fall
    # back to ``to_dict`` when unavailable.
    if hasattr(gen_cfg, "to_diff_dict"):
        cfg_dict = gen_cfg.to_diff_dict()  # type: ignore[attr-defined]
    else:
        cfg_dict = gen_cfg.to_dict()  # type: ignore[attr-defined]

    # Filter out keys whose value is ``None`` so that they do not overwrite backend
    # defaults when the caller merges the dictionaries.
    return {k: v for k, v in cfg_dict.items() if v is not None}


def _from_gguf_header(model_path: str) -> Dict[str, Any]:
    """Attempt to read generation defaults from a GGUF model header.

    Currently limited by optional *gguf* dependency.  If unavailable, or the file
    extension is not ``.gguf``, the function returns an empty dict.
    """

    p = Path(model_path)
    if p.suffix.lower() != ".gguf":
        return {}

    try:
        import gguf  # type: ignore
    except ModuleNotFoundError:
        logging.debug("gguf not installed – skipping GGUF metadata load")
        return {}

    try:
        gg = gguf.GGUFReader(str(p))  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover – defensive
        logging.error("Failed to open GGUF file %s – %s", model_path, exc)
        return {}

    # Mapping of GGUF metadata keys → SamplingParams attribute names.
    key_map = {
        "temperature": "temperature",
        "top_k": "top_k",
        "top_p": "top_p",
        "repetition_penalty": "repetition_penalty",
        "logit_bias": "logit_bias",
    }

    out: Dict[str, Any] = {}
    for gguf_key, param_key in key_map.items():
        if gguf_key in gg.meta:
            out[param_key] = gg.meta[gguf_key]

    return out


def load_default_sampling_params(model_path: str) -> Dict[str, Any]:
    """Load default sampling parameters for *model_path*.

    The helper tries, in order:
    1. ``generation_config.json`` via *transformers* (works for HF hub ids).
    2. GGUF header metadata (when the path ends with ``.gguf``).

    If neither yields values an empty dict is returned.
    """

    defaults: Dict[str, Any] = {}

    # 1. transformers / generation_config.json
    defaults.update(_from_transformers(model_path))

    # 2. GGUF metadata – only fills keys not already set.
    gguf_defaults = _from_gguf_header(model_path)
    for k, v in gguf_defaults.items():
        defaults.setdefault(k, v)

    return defaults 