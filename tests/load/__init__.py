"""Concurrency and load testing for the streaming ASR service.

Separate from the unit suite on purpose. The tests in ``tests/`` prove the
pipeline is *correct*; these measure what it does when several people talk to
it at once, which is a different question with different failure modes --
admission refusals, queueing behind a shared ONNX session, latency that only
degrades past some concurrency, streams that die without taking the run down.

Two things live here and must not be confused:

* **Harness correctness** -- ``test_load_harness.py`` and
  ``test_failure_modes.py``. Fast, deterministic, run by ``pytest`` against a
  scripted fake server with no model. They prove the measurement apparatus
  works. They say nothing whatsoever about ASR performance.
* **The benchmark** -- ``load_test.py`` and ``run_load_sweep.py``. Run by hand
  against a real service with a real model, on the hardware you intend to
  deploy on. These produce the numbers.

Nothing here is CTC-specific: the service is treated as something that emits
``partial`` / ``segment`` / ``final`` / ``error`` events, so an RNNT or
cache-aware model behind the same protocol is measured by the same code.
"""
