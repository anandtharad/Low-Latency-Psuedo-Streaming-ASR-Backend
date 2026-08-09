"""Build a :class:`StreamingASRConfig` from the environment.

Containers configure through the environment, so the whole pipeline is
reachable via ``ASR_*`` variables with no code change and no bespoke config
format. A JSON file is also supported for anything long-winded; explicit
environment variables win over it.

Only ``ASR_MODEL_PATH`` is required::

    docker run -p 8000:8000 \\
        -v /data/asr:/data/asr:ro \\
        -e ASR_MODEL_PATH=/data/asr/ams/model.onnx \\
        -e ASR_LM_PATH=/data/asr/lms/12.0/merged_lm.bin \\
        -e ASR_LEXICON_PATH=/data/asr/lms/12.0/merged_lm.lexicon \\
        streaming-asr
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from streaming_asr.config import (
    BeamDecoderConfig,
    EndpointConfig,
    SegmentationConfig,
    StabilityConfig,
    StreamingASRConfig,
    load_vocabulary,
)

logger = logging.getLogger(__name__)

ENV_PREFIX = "ASR_"


def _get(name: str, default: Any = None) -> Any:
    return os.environ.get(ENV_PREFIX + name, default)


def _get_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = _get(name)
    return default if raw is None else float(raw)


def _get_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = _get(name)
    return default if raw is None else int(raw)


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _require_readable(path: Optional[str], label: str) -> Optional[str]:
    """Fail at startup, not on the first request.

    A container that boots "successfully" and then 500s on every call because a
    volume was not mounted is much harder to diagnose than one that refuses to
    start and says which path is missing.
    """
    if path is None:
        return None
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{label} not found: {path}. If running in Docker, check that the "
            f"volume containing it is mounted and the path is the in-container one."
        )
    return path


def config_from_env() -> StreamingASRConfig:
    """Assemble the configuration from ``ASR_*`` variables (and optional JSON)."""
    base: dict[str, Any] = {}

    config_file = _get("CONFIG")
    if config_file:
        _require_readable(config_file, "ASR_CONFIG")
        base = json.loads(Path(config_file).read_text(encoding="utf-8"))
        base.pop("derived", None)
        logger.info("Loaded base configuration from %s", config_file)

    model_path = _get("MODEL_PATH", base.get("onnx_model_path"))
    if not model_path:
        raise ValueError(
            "ASR_MODEL_PATH is required (path to the exported Conformer-CTC .onnx)"
        )
    _require_readable(model_path, "ASR_MODEL_PATH")

    lm_path = _require_readable(_get("LM_PATH", base.get("lm_path")), "ASR_LM_PATH")
    lexicon_path = _require_readable(
        _get("LEXICON_PATH", base.get("lexicon_path")), "ASR_LEXICON_PATH"
    )

    vocab_path = _get("VOCAB_PATH")
    vocabulary = base.get("vocabulary")
    blank_id = base.get("blank_id")
    if vocab_path:
        _require_readable(vocab_path, "ASR_VOCAB_PATH")
        vocabulary = load_vocabulary(vocab_path)
        # A vocabulary file is expected to already contain the blank symbol.
        blank_id = len(vocabulary) - 1
        logger.info("Loaded %d vocabulary tokens from %s", len(vocabulary), vocab_path)

    sub = {
        "beam": BeamDecoderConfig(
            beam_size=_get_int("BEAM_SIZE", 50),
            beam_size_token=_get_int("BEAM_SIZE_TOKEN", 50),
            beam_threshold=_get_float("BEAM_THRESHOLD", 20.0),
            lm_weight=_get_float("LM_WEIGHT", 2.0),
            word_score=_get_float("WORD_SCORE", 0.0),
            backend=_get("BEAM_BACKEND", "auto"),
        ),
        "stability": StabilityConfig(
            stability_window=_get_float("STABILITY_WINDOW", 0.6),
            min_stable_updates=_get_int("MIN_STABLE_UPDATES", 2),
            aligner=_get("ALIGNER", "time"),
            time_tolerance=_get_float("TIME_TOLERANCE", 0.12),
        ),
        "endpoint": EndpointConfig(
            detector=_get("ENDPOINT", "explicit"),
            silence_duration=_get_float("SILENCE_SEC", 0.8),
            energy_threshold=_get_float("ENERGY_THRESHOLD", 0.005),
            min_speech_duration=_get_float("MIN_SPEECH_SEC", 0.5),
        ),
        "segmentation": SegmentationConfig(
            segment_silence=_get_float("SEGMENT_SILENCE", 0.5),
            turn_silence=_get_float("TURN_SILENCE", 1.5),
            max_segment_duration=_get_float("MAX_SEGMENT_SEC", 10.0),
            min_segment_speech=_get_float("MIN_SEGMENT_SPEECH", 0.2),
            energy_threshold=_get_float("ENERGY_THRESHOLD", 0.005),
        ),
    }

    kwargs: dict[str, Any] = {
        "sample_rate": _get_int("SAMPLE_RATE", 16000),
        "chunk_duration": _get_float("CHUNK_MS", 160.0) / 1000.0,
        "context_duration": _get_float("CONTEXT_SEC", 3.84),
        "onnx_model_path": model_path,
        "lm_path": lm_path,
        "lexicon_path": lexicon_path,
        "pipeline": _get("PIPELINE", "segmented"),
        "runtime": _get("RUNTIME", "lite"),
        "frontend_path": _get("FRONTEND_PATH"),
        "greedy_decode": _get_bool("GREEDY", True),
        "final_beam_decode": _get_bool("FINAL_BEAM", True),
        "device": _get("DEVICE", "auto"),
        "providers": _get("PROVIDERS", "auto"),
        "intra_op_threads": _get_int("INTRA_OP_THREADS", 0),
        "max_history": _get_float("MAX_HISTORY_SEC", 120.0),
        "final_segment_duration": _get_float("FINAL_SEGMENT_SEC", 10.0),
        "pad_warmup_window": _get_bool("PAD_WARMUP", False),
        **sub,
    }
    if vocabulary is not None:
        kwargs["vocabulary"] = list(vocabulary)
        kwargs["blank_id"] = blank_id

    return StreamingASRConfig(**kwargs)


def server_settings() -> dict[str, Any]:
    """Server-level knobs that are not part of the ASR config."""
    return {
        "host": _get("HOST", "0.0.0.0"),
        "port": _get_int("PORT", 8000),
        "max_concurrent_streams": _get_int("MAX_CONCURRENT_STREAMS", 4),
        "max_upload_seconds": _get_float("MAX_UPLOAD_SEC", 300.0),
        "log_level": _get("LOG_LEVEL", "info"),
    }
