"""Charts from a completed sweep. Entirely optional.

Kept apart from the load generator on purpose: nothing that produces a
measurement should import a plotting library, and a missing ``matplotlib`` must
never be the reason a benchmark did not run. Every entry point here degrades to
"no plots written" and says so.

Can also be run against a results file after the fact, which is the usual case
-- you rarely know you wanted a chart until you have seen the table::

    python -m tests.load.plots results/load_test_20260809_143210.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from tests.load.metrics import LevelSummary


def _matplotlib() -> Optional[Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display on a benchmark host
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _series(summaries: Sequence[LevelSummary], metric: str, key: str) -> list[Any]:
    return [summary.latencies.get(metric, {}).get(key) for summary in summaries]


def _grouped(summaries: Sequence[LevelSummary]) -> dict[Optional[int], list[LevelSummary]]:
    """Split by pool size, so a pool sweep draws one line per configuration."""
    groups: dict[Optional[int], list[LevelSummary]] = {}
    for summary in summaries:
        groups.setdefault(summary.pool_size, []).append(summary)
    for entries in groups.values():
        entries.sort(key=lambda s: s.concurrency)
    return groups


def plot_sweep(
    summaries: Sequence[LevelSummary],
    directory: str | Path,
    run_id: str,
    latency_metric: str = "segment_response_latencies",
) -> list[Path]:
    """Write the sweep charts. Returns the files written (empty if unavailable)."""
    plt = _matplotlib()
    if plt is None or not summaries:
        return []

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    groups = _grouped(summaries)
    written: list[Path] = []

    panels = [
        ("rtf", "Server RTF (compute / audio)",
         lambda group: [s.rtf.get("server_p50") for s in group],
         lambda group: [s.rtf.get("server_p95") for s in group], ("p50", "p95")),
        ("latency", f"{latency_metric} (ms)",
         lambda group: _series(group, latency_metric, "p50_ms"),
         lambda group: _series(group, latency_metric, "p95_ms"), ("p50", "p95")),
    ]

    for name, ylabel, first, second, labels in panels:
        figure, axes = plt.subplots(figsize=(7, 4.2))
        for pool, group in groups.items():
            suffix = f" (pool {pool})" if pool is not None and len(groups) > 1 else ""
            x = [s.concurrency for s in group]
            axes.plot(x, first(group), marker="o", label=f"{labels[0]}{suffix}")
            axes.plot(x, second(group), marker="s", linestyle="--",
                      label=f"{labels[1]}{suffix}")
        axes.set_xlabel("concurrent streams")
        axes.set_ylabel(ylabel)
        axes.grid(alpha=0.3)
        axes.legend()
        if name == "rtf":
            # RTF 1.0 is where compute stops keeping up with audio. Marking it
            # is the difference between a curve and a curve you can read.
            axes.axhline(1.0, color="crimson", linewidth=1, alpha=0.6)
            axes.annotate("RTF = 1 (real time)", xy=(x[0], 1.0),
                          xytext=(0, 4), textcoords="offset points",
                          color="crimson", fontsize=8)
        figure.tight_layout()
        path = directory / f"load_test_{run_id}_{name}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    # Percentile fan for the chosen latency metric.
    figure, axes = plt.subplots(figsize=(7, 4.2))
    for pool, group in groups.items():
        suffix = f" (pool {pool})" if pool is not None and len(groups) > 1 else ""
        x = [s.concurrency for s in group]
        for key, style in (("p50_ms", "-"), ("p95_ms", "--"), ("p99_ms", ":")):
            axes.plot(x, _series(group, latency_metric, key), linestyle=style,
                      marker="o", markersize=3, label=f"{key[:-3]}{suffix}")
    axes.set_xlabel("concurrent streams")
    axes.set_ylabel(f"{latency_metric} (ms)")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    path = directory / f"load_test_{run_id}_percentiles.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path)

    # Success rate and GPU utilisation share an axis pair: the interesting
    # reading is whether streams start failing while the device still has
    # headroom, which points away from the GPU as the constraint.
    figure, axes = plt.subplots(figsize=(7, 4.2))
    twin = axes.twinx()
    plotted_gpu = False
    for pool, group in groups.items():
        suffix = f" (pool {pool})" if pool is not None and len(groups) > 1 else ""
        x = [s.concurrency for s in group]
        axes.plot(x, [100 * s.success_rate for s in group], marker="o",
                  label=f"success %{suffix}")
        gpu = [s.resources.get("gpu_percent_mean") for s in group]
        if any(value is not None for value in gpu):
            twin.plot(x, gpu, marker="^", linestyle="--", color="tab:orange",
                      label=f"GPU %{suffix}")
            plotted_gpu = True
    axes.set_xlabel("concurrent streams")
    axes.set_ylabel("successful streams (%)")
    axes.set_ylim(0, 105)
    twin.set_ylabel("GPU utilisation (%)" if plotted_gpu
                    else "GPU utilisation (unavailable)")
    axes.grid(alpha=0.3)
    axes.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    path = directory / f"load_test_{run_id}_success_gpu.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path)

    return written


def plot_from_results(path: str | Path) -> list[Path]:
    """Redraw charts from a results JSON written by an earlier run."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    summaries = []
    for level in payload.get("levels", []):
        raw = dict(level["summary"])
        for derived in ("success_rate", "error_rate", "aggregate_audio_throughput"):
            raw.pop(derived, None)
        summaries.append(LevelSummary(**raw))
    return plot_sweep(summaries, Path(path).parent, payload.get("run_id", "replot"))


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m tests.load.plots <results.json>", file=sys.stderr)
        return 2
    if _matplotlib() is None:
        print("matplotlib is not installed; nothing to draw "
              "(pip install matplotlib)", file=sys.stderr)
        return 0
    for written in plot_from_results(argv[0]):
        print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
