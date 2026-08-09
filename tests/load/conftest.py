"""Fixtures for the load-harness tests.

The repository root goes on ``sys.path`` explicitly so ``tests.load...`` imports
resolve however pytest was invoked. ``tests/`` has no ``__init__.py``, so under
pytest's default import mode only ``tests/`` itself is prepended, and the
absolute imports used throughout this package would otherwise depend on the
working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi", reason="the fake server is a FastAPI app")
pytest.importorskip("uvicorn", reason="the fake server needs an ASGI server")
pytest.importorskip("websockets", reason="the load client speaks WebSocket")

from tests.load.fake_server import FakeConfig, FakeServer  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers locally.

    Done here rather than in a project-wide ``pytest.ini`` so this package
    stays self-contained: the repository has no pytest configuration file, and
    adding one to register a single marker would change collection behaviour
    for every existing test.
    """
    config.addinivalue_line(
        "markers",
        "slow: starts a real ASR service; deselect with -m 'not slow'",
    )


@pytest.fixture
def fake_config() -> FakeConfig:
    """Fast defaults: short segments so a two-second clip still produces several."""
    return FakeConfig(segment_seconds=0.5, segment_silence=0.2)


@pytest.fixture
def fake_server(fake_config: FakeConfig):
    """A protocol-compatible server with no model behind it.

    Function-scoped: the failure tests mutate the config, and a shared server
    would leak one test's injected fault into the next.
    """
    with FakeServer(fake_config) as server:
        yield server
