"""Production service: model resident in memory, requests over HTTP/WebSocket.

``app`` is imported lazily so that ``import streaming_asr.server`` does not
require FastAPI to be installed for CLI-only or library-only use.
"""

from streaming_asr.server.model_pool import ModelPool, PoolStatus
from streaming_asr.server.settings import config_from_env, server_settings

__all__ = ["ModelPool", "PoolStatus", "config_from_env", "server_settings", "app"]


def __getattr__(name: str):
    if name in ("app", "main"):
        from streaming_asr.server import app as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
