#!/usr/bin/env bash
#
# Full measurement run: per-duration profile + concurrency sweep, on every
# device the machine has.
#
# Produces the numbers in docs/PROJECT_REPORT.md sections 6.2-6.8. Everything
# lands in results/<timestamp>/ with the metadata needed to read it later --
# active providers, thread counts, GPU model, host.
#
#   ./scripts/benchmark_all.sh
#   MODEL=/path/model.onnx VOCAB=/path/vocab.txt ./scripts/benchmark_all.sh
#   LM=/path/lm.bin LEXICON=/path/lexicon.txt ./scripts/benchmark_all.sh
#   DEVICES=cpu ./scripts/benchmark_all.sh          # skip the GPU pass
#
# Runtime is roughly 20-40 min on a GPU box, longer CPU-only. Nothing here is
# interactive; it is safe to run under nohup.

set -euo pipefail

# ---------------------------------------------------------------------------
# configuration -- override any of these from the environment
# ---------------------------------------------------------------------------

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-stt_en_cconformer_ctc_large-averaged.onnx}"
VOCAB="${VOCAB:-vocab.txt}"
FRONTEND="${FRONTEND:-fixtures/frontend.onnx}"
CLIPS="${CLIPS:-common_voice_en/en_train_28}"
MANIFEST="${MANIFEST:-common_voice_en/train.tsv}"
LOAD_AUDIO="${LOAD_AUDIO:-fixtures/load_sample.wav}"
RUNTIME="${RUNTIME:-lite}"

# Duration bins, seconds. The top sits under the ~150 s single-pass failure
# wall (PROJECT_REPORT.md 2.3); the segmented path is flat through it, but a
# machine with less VRAM will hit the wall sooner, so raise this deliberately.
BINS="${BINS:-5,10,15,20,30,45,60,90,120}"

# Concurrency levels per device. A GPU box can take more; CPU saturates fast.
GPU_LEVELS="${GPU_LEVELS:-1,2,4,8,12,16}"
CPU_LEVELS="${CPU_LEVELS:-1,2,4,8}"

# Admission cap used for the sweep. Also feeds the derived intra-op thread
# count on CPU, so it is not a cosmetic setting -- see
# streaming_asr_lite/execution.py.
MAX_STREAMS="${MAX_STREAMS:-8}"

# Beam + LM. Without both of these the beam_lm variant is skipped and the
# 'beam' column measures an LM-free beam, which is a different thing.
LM="${LM:-}"
LEXICON="${LEXICON:-}"

DEVICES="${DEVICES:-auto}"   # auto | cpu | gpu | "cpu gpu"

# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/${STAMP}"
mkdir -p "$OUT"

log() { printf '\n\033[1m== %s\033[0m\n' "$*" | tee -a "$OUT/run.log"; }
note() { printf '   %s\n' "$*" | tee -a "$OUT/run.log"; }

# ---- preflight ------------------------------------------------------------

log "Preflight"

missing=0
for f in "$MODEL" "$VOCAB"; do
    if [ ! -f "$f" ]; then echo "   MISSING: $f" >&2; missing=1; fi
done
if [ "$missing" -ne 0 ]; then
    cat >&2 <<'EOF'

Set MODEL and VOCAB to your copies. The checkpoint is not in the repository:

    MODEL=/path/to/model.onnx VOCAB=/path/to/vocab.txt ./scripts/benchmark_all.sh

Regenerate the vocabulary from a .nemo with:

    python tools/extract_vocabulary.py --nemo model.nemo --out vocab.txt
EOF
    exit 1
fi

if [ "$RUNTIME" = "lite" ] && [ ! -f "$FRONTEND" ]; then
    note "frontend missing, exporting it once (needs torch; nothing after does)"
    "$PYTHON" -m streaming_asr_lite.export_frontend --out "$FRONTEND" \
        2>&1 | tee -a "$OUT/run.log"
fi

HAVE_CORPUS=1
if [ ! -d "$CLIPS" ] || [ ! -f "$MANIFEST" ]; then
    HAVE_CORPUS=0
    note "corpus not found at $CLIPS -- the duration profile will be skipped."
    note "point CLIPS/MANIFEST at a Common Voice shard to include it."
fi

if [ ! -f "$LOAD_AUDIO" ]; then
    if [ "$HAVE_CORPUS" -eq 1 ]; then
        note "building the load fixture from real clips"
        "$PYTHON" tools/build_load_fixture.py --clips "$CLIPS" \
            --manifest "$MANIFEST" --seconds 45 --out "$LOAD_AUDIO" \
            --transcript "${LOAD_AUDIO%.wav}.txt" 2>&1 | tee -a "$OUT/run.log"
    else
        echo "   MISSING: $LOAD_AUDIO and no corpus to build it from" >&2
        exit 1
    fi
fi

# ---- which devices ---------------------------------------------------------

detect_gpu() {
    "$PYTHON" - <<'PY' 2>/dev/null
import sys
try:
    from streaming_asr_lite.execution import ensure_cuda_libraries
    ensure_cuda_libraries()
    import onnxruntime as ort
    sys.exit(0 if "CUDAExecutionProvider" in ort.get_available_providers() else 1)
except Exception:
    sys.exit(1)
PY
}

case "$DEVICES" in
    auto)
        if detect_gpu; then TARGETS="gpu cpu"; else TARGETS="cpu"; fi ;;
    *)  TARGETS="$DEVICES" ;;
esac
note "devices: $TARGETS"
note "runtime: $RUNTIME"
note "output : $OUT"

"$PYTHON" - <<PY 2>&1 | tee -a "$OUT/run.log"
import json, platform, subprocess
info = {"host": platform.platform(), "python": platform.python_version()}
try:
    info["cpu_count"] = __import__("os").cpu_count()
except Exception:
    pass
try:
    info["gpu"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10).stdout.strip()
except Exception:
    info["gpu"] = "none"
print("   " + json.dumps(info))
open("$OUT/host.json", "w").write(json.dumps(info, indent=2))
PY

# ---- the runs --------------------------------------------------------------

variants="greedy,beam"
if [ -n "$LM" ] && [ -n "$LEXICON" ]; then
    variants="greedy,beam,beam_lm"
    note "beam_lm enabled: $LM"
else
    note "beam_lm skipped (set LM and LEXICON to include it)"
fi

for target in $TARGETS; do
    if [ "$target" = "gpu" ]; then
        providers="CUDAExecutionProvider"
        levels="$GPU_LEVELS"
        device="cuda"
    else
        providers="CPUExecutionProvider"
        levels="$CPU_LEVELS"
        device="cpu"
    fi

    # --- 1. per-duration profile: RTF, decode time excluding silence,
    #        decoder cost per segment. Single stream.
    if [ "$HAVE_CORPUS" -eq 1 ]; then
        log "[$target] duration profile"
        lm_args=""
        if [ "$variants" = "greedy,beam,beam_lm" ]; then
            lm_args="--lm $LM --lexicon $LEXICON"
        fi
        # -u so progress streams through tee instead of arriving at the end.
        # shellcheck disable=SC2086
        "$PYTHON" -u tools/profile_by_duration.py \
            --model "$MODEL" --vocabulary "$VOCAB" \
            --clips "$CLIPS" --manifest "$MANIFEST" \
            --frontend "$FRONTEND" --runtime "$RUNTIME" \
            --bins "$BINS" --variants "$variants" $lm_args \
            --device "$device" --providers "$providers" \
            --json-out "$OUT/profile_${target}.json" \
            2>&1 | tee "$OUT/profile_${target}.log"
    fi

    # --- 2. concurrency sweep against the live service.
    log "[$target] concurrency sweep (levels $levels)"
    ASR_MODEL_PATH="$MODEL" \
    ASR_VOCAB_PATH="$VOCAB" \
    ASR_FRONTEND_PATH="$FRONTEND" \
    ASR_RUNTIME="$RUNTIME" \
    ASR_DEVICE="$device" \
    ASR_PROVIDERS="$providers" \
    ASR_FINAL_BEAM=false \
    ASR_LOG_LEVEL=warning \
    "$PYTHON" -u -m tests.load.run_load_sweep \
        --audio "$LOAD_AUDIO" \
        --levels "$levels" \
        --mode realtime \
        --spawn-server \
        --max-concurrent-streams "$MAX_STREAMS" \
        --results-dir "$OUT" \
        --server-log "$OUT/server_${target}.log" \
        2>&1 | tee "$OUT/sweep_${target}.log" || \
        note "sweep on $target exited non-zero -- partial results kept"
done

# ---- summary ---------------------------------------------------------------

log "Done"
note "results in $OUT"
ls -1 "$OUT" | sed 's/^/     /'

cat <<EOF

Read them in this order:

  1. profile_*.log    where the time goes at each turn length. rtf_speech is
                      the model's true cost; rtf_audio is the capacity figure.
  2. sweep_*.log      the concurrency curve. Size on response latency, never
                      on RTF -- RTF stays under 1 well past the point where
                      callers are waiting 20 s.
  3. host.json        what it ran on, so the numbers stay readable later.

If a GPU was present but a sweep reports CPUExecutionProvider, the run header
says so explicitly. Believe the header, not the intent.
EOF
