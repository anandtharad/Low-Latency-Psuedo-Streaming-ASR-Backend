"""Start and stop the ASR service, so a sweep can vary its configuration.

Only needed for the questions that cannot be asked of an already-running
service -- principally "does raising ``ASR_MAX_CONCURRENT_STREAMS`` buy
anything?", which requires a restart because the limit is fixed at startup.

Everything else in this package talks to whatever is already listening at
``--url``, which is the normal and safer way to benchmark: the service under
test should be the one you actually deploy, launched the way you actually
launch it.

Configuration is inherited from the current environment (``ASR_MODEL_PATH`` and
friends) with only the swept variables overridden, so a spawned server is the
same service with one knob moved -- not a second, subtly different one.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import subprocess
import sys
import time
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class ServerStartupError(RuntimeError):
    """The service did not become ready in time."""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_until_ready(url: str, timeout: float, process: Optional[subprocess.Popen] = None
                     ) -> None:
    """Poll ``/health`` until the model is loaded.

    Model load is seconds to tens of seconds -- benchmarking before it finishes
    would attribute startup cost to the first concurrency level. If the process
    dies while waiting, its exit code is reported immediately rather than after
    the full timeout, because a misconfigured service fails in the first second
    and waiting sixty more helps nobody.
    """
    import httpx

    deadline = time.perf_counter() + timeout
    last_error = "no response"
    while time.perf_counter() < deadline:
        if process is not None and process.poll() is not None:
            raise ServerStartupError(
                f"server exited with code {process.returncode} before becoming "
                f"ready. Check ASR_MODEL_PATH / ASR_VOCAB_PATH."
            )
        try:
            response = httpx.get(f"{url.rstrip('/')}/health", timeout=5.0)
            if response.status_code == 200 and response.json().get("ready"):
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001 - not up yet is the common case
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)
    raise ServerStartupError(f"{url} not ready within {timeout:.0f}s ({last_error})")


@contextlib.contextmanager
def spawned_server(
    max_concurrent_streams: int,
    port: Optional[int] = None,
    env_overrides: Optional[dict[str, str]] = None,
    startup_timeout: float = 300.0,
    log_path: Optional[str] = None,
) -> Iterator[str]:
    """Run the service for the duration of the block; yield its base URL."""
    port = port or free_port()
    env = dict(os.environ)
    env["ASR_MAX_CONCURRENT_STREAMS"] = str(max_concurrent_streams)
    env["ASR_PORT"] = str(port)
    env["ASR_HOST"] = "127.0.0.1"
    env.update(env_overrides or {})

    if not env.get("ASR_MODEL_PATH"):
        raise ServerStartupError(
            "ASR_MODEL_PATH is not set. A spawned server is configured from the "
            "environment; export the same variables you would use to start it "
            "by hand."
        )

    url = f"http://127.0.0.1:{port}"
    sink = open(log_path, "w", encoding="utf-8") if log_path else subprocess.DEVNULL
    logger.info("starting server on %s with max_concurrent_streams=%d",
                url, max_concurrent_streams)
    process = subprocess.Popen(
        [sys.executable, "-m", "streaming_asr.server.app"],
        env=env, stdout=sink, stderr=subprocess.STDOUT,
    )
    try:
        wait_until_ready(url, startup_timeout, process)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn process
            process.kill()
            process.wait(timeout=10)
        if sink is not subprocess.DEVNULL:
            sink.close()
