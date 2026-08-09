"""The shared ONNX session must be safe under concurrent use.

The service loads one engine and shares it across every stream, so
``ONNXASREngine.run`` is called from many threads at once. ONNX Runtime's own
``Run()`` is thread-safe; the risk is our bookkeeping around it.

The specific defect these tests exist to prevent: a single reused scratch
buffer for the ``length`` input. Thread A writes its length, thread B
overwrites it, then thread A hands the mutated array to ORT. NeMo's Conformer
uses ``length`` to build the padding mask, so the result is a stream silently
masked to the wrong number of frames -- a truncated or garbage transcript, with
no exception raised anywhere.

The assertion is equality against a serially-computed reference. Shape alone
would not catch it: output width follows the *features* tensor, so a corrupted
length changes the masking, not the dimensions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from streaming_asr.inference.onnx_engine import ONNXASREngine

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
MODEL = FIXTURES / "synthetic_model.onnx"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason="synthetic fixture not built; run tools/build_synthetic_fixture.py",
)

# Deliberately varied. Identical lengths across threads would make a corrupted
# length indistinguishable from a correct one.
LENGTHS = [40, 100, 200, 401, 60, 300, 150, 401]
FEATURE_FRAMES = 401


@pytest.fixture(scope="module")
def engine() -> ONNXASREngine:
    return ONNXASREngine(str(MODEL), providers="auto")


def _features(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 80, FEATURE_FRAMES)).astype(np.float32)


def test_concurrent_runs_match_serial_results(engine):
    """Same inputs, run concurrently, must give byte-identical outputs."""
    inputs = [(_features(i), length) for i, length in enumerate(LENGTHS)]
    expected = [engine.run(f, n).logits for f, n in inputs]

    # Repeat so the interleaving has many chances to go wrong.
    for _ in range(12):
        with ThreadPoolExecutor(max_workers=len(inputs)) as pool:
            futures = [pool.submit(engine.run, f, n) for f, n in inputs]
            results = [future.result().logits for future in futures]

        for index, (actual, reference) in enumerate(zip(results, expected)):
            np.testing.assert_array_equal(
                actual, reference,
                err_msg=(
                    f"stream {index} (length={LENGTHS[index]}) diverged under "
                    f"concurrency -- its length input was corrupted by another thread"
                ),
            )


def test_length_controls_masking_so_corruption_is_observable(engine):
    """Establish that a wrong length really does change the output.

    Without this, the test above could pass vacuously on a model that ignores
    the length input entirely.
    """
    features = _features(99)
    short = engine.run(features, 100).logits
    long = engine.run(features, 400).logits

    assert short.shape == long.shape, "shape follows features, not length"
    assert not np.array_equal(short, long), (
        "length does not affect this model's output, so the concurrency test "
        "cannot detect a corrupted length"
    )


def test_call_counters_are_not_lost_under_concurrency(engine):
    """Metrics use non-atomic increments; they need the lock."""
    before = engine.call_count
    calls = 64

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(engine.run, _features(i % 4), LENGTHS[i % len(LENGTHS)])
            for i in range(calls)
        ]
        for future in futures:
            future.result()

    assert engine.call_count == before + calls
