#!/usr/bin/env bash
#
# serve-glm52.sh — one-click GLM-5.2 serving on an AMD MI325X node via
# SGLang in a podman container.
#
# Usage:
#   ./serve-glm52.sh [serve]     start the server (default; runs until killed)
#   ./serve-glm52.sh serve --detach
#                                start the server, wait until healthy, then
#                                return the shell (server keeps running in the
#                                background for the life of the SLURM job) —
#                                use this to run opencode on the GPU node itself
#   ./serve-glm52.sh download    prefetch model weights only (no GPU needed,
#                                fine to run on the login node)
#   ./serve-glm52.sh bench       benchmark the running server (sglang.bench_serving)
#   ./serve-glm52.sh stop        stop a running server container
#   ./serve-glm52.sh status      show container state + health endpoint
#
# Configuration comes from glm52.env next to this script (or $GLM52_ENV),
# see glm52-env.example. Environment variables you export beforehand win.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="glm52-sglang"

MODE="serve"
DETACH=0
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *)           MODE="$arg" ;;
    esac
done

log()  { printf '\033[1;34m[glm52]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm52 WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm52 ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Load config ─────────────────────────────────────────────────────────────

ENV_FILE="${GLM52_ENV:-$SCRIPT_DIR/glm52.env}"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    log "Loaded config from $ENV_FILE"
else
    warn "No config file at $ENV_FILE (copy glm52-env.example to glm52.env); using environment only."
fi

MODEL_ID="${MODEL_ID:-zai-org/GLM-5.2-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang-rocm:v0.5.16-rocm720-mi30x-20260731}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-auto}"
CONTEXT_LEN="${CONTEXT_LEN:-262144}"
# Empty = let SGLang size it from the GPU's memory capacity. On a 256 GB MI325X
# its heuristic lands well above the old hard-coded 0.85, which was throwing
# away ~38 GB of KV cache per GPU. Set a number here only to work around OOM.
MEM_FRACTION="${MEM_FRACTION:-}"
SHM_SIZE="${SHM_SIZE:-64g}"

# podman defaults --pids-limit to 2048, and under cgroup v2 *threads* count
# against it. Each TP rank runs an OpenMP pool sized to the whole machine plus
# a weight-loader thread pool, so on a many-core node 8 ranks blow past 2048
# during weight loading and Python raises "can't start new thread". -1 removes
# the cap; podman ignores it if cgroups aren't delegated.
PIDS_LIMIT="${PIDS_LIMIT:--1}"

# Threads per rank for OpenMP/torch CPU ops. Unset, each rank grabs every core
# on the node, so TP_SIZE ranks oversubscribe the CPU by TP_SIZE-fold and pile
# up threads. "auto" divides the node's cores evenly between the ranks.
OMP_THREADS="${OMP_THREADS:-auto}"

ENABLE_AITER_ALLREDUCE_FUSION="${ENABLE_AITER_ALLREDUCE_FUSION:-1}"
READY_TIMEOUT="${READY_TIMEOUT:-7200}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"

# GLM-5.2 is a DeepSeek-Sparse-Attention model (GlmMoeDsaForCausalLM). On ROCm
# SGLang only takes the fast DSA path -- aiter's preshuffle paged-MQA kernel and
# a KV page size of 64 -- when SGLANG_USE_AITER is truthy; otherwise it falls
# back to page_size=1, one token per page. Leave this at 1.
USE_AITER="${USE_AITER:-1}"
# The preshuffle path additionally needs Triton >= 3.5 in the image. If the
# startup log says "Setting page size to 1 for DeepSeek DSA on ROCm", set this
# to 1 to use aiter's ahead-of-time gluon kernels instead.
AITER_AOT_GLUON="${AITER_AOT_GLUON:-0}"

# Prometheus metrics + per-request prefix-cache hit rates. Cheap, and the cache
# report is the only way to tell whether prefix caching is helping the agent.
ENABLE_METRICS="${ENABLE_METRICS:-1}"

# MTP speculative decoding. The FP8 checkpoint ships the NextN draft layer
# (model.layers.78.eh_proj/enorm/hnorm), so no separate draft model is needed.
# Opt-in: the DSA+MTP path is not yet proven on gfx942 -- benchmark it.
ENABLE_MTP="${ENABLE_MTP:-0}"
MTP_NUM_STEPS="${MTP_NUM_STEPS:-3}"
MTP_NUM_DRAFT_TOKENS="${MTP_NUM_DRAFT_TOKENS:-4}"

# HiCache: tier the radix prefix cache into host RAM. Agentic loops re-send a
# near-identical prefix every turn, so this is well matched to opencode -- but
# it needs the page-size-64 DSA path above, so keep USE_AITER=1 with it.
ENABLE_HICACHE="${ENABLE_HICACHE:-0}"
HICACHE_RATIO="${HICACHE_RATIO:-2}"

# 'lpm' (longest prefix match) reorders requests to encourage cache hits, at
# some scheduling cost. Worth setting when several people share the endpoint.
SCHEDULE_POLICY="${SCHEDULE_POLICY:-}"

# ── Simple modes first ──────────────────────────────────────────────────────

case "$MODE" in
    stop)
        log "Stopping $CONTAINER_NAME ..."
        podman stop -t 30 "$CONTAINER_NAME" 2>/dev/null \
            && log "Stopped." \
            || warn "No running container named $CONTAINER_NAME."
        exit 0
        ;;
    status)
        state="$(podman inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "not found")"
        log "Container $CONTAINER_NAME: $state"
        if [[ "$state" == "running" ]]; then
            if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
                log "Health check on port $PORT: OK"
            else
                warn "Container running but http://127.0.0.1:${PORT}/health not responding (still loading?)."
                log "Follow progress with: podman logs -f $CONTAINER_NAME"
            fi
        fi
        exit 0
        ;;
    serve|download|bench) ;;
    *)
        die "Unknown mode '$MODE'. Use: serve | download | bench | stop | status"
        ;;
esac

# ── Preflight ───────────────────────────────────────────────────────────────

command -v podman >/dev/null 2>&1 || die "podman not found on PATH."
command -v curl   >/dev/null 2>&1 || die "curl not found on PATH."

[[ -n "$MODEL_CACHE_DIR" ]] \
    || die "MODEL_CACHE_DIR is not set. Point it at a large shared filesystem (~750 GB needed). See glm52-env.example."
mkdir -p "$MODEL_CACHE_DIR" 2>/dev/null || true
[[ -d "$MODEL_CACHE_DIR" && -w "$MODEL_CACHE_DIR" ]] \
    || die "MODEL_CACHE_DIR '$MODEL_CACHE_DIR' does not exist or is not writable."

# Resolve HF token: env var, then token file. The export matters: we hand the
# token to podman with a bare '-e HF_TOKEN', which only forwards *exported*
# variables. Without it, HF_TOKEN_FILE silently forwarded nothing unless
# glm52.env happened to have exported HF_TOKEN first.
if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
    [[ -r "$HF_TOKEN_FILE" ]] || die "HF_TOKEN_FILE '$HF_TOKEN_FILE' is not readable."
    HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi
export HF_TOKEN="${HF_TOKEN:-}"

# Weights already cached? (hub layout: models--org--name)
weights_dir="$MODEL_CACHE_DIR/hub/models--${MODEL_ID//\//--}"
weights_cached=0
[[ -d "$weights_dir/snapshots" ]] && weights_cached=1

if [[ -z "${HF_TOKEN:-}" && "$weights_cached" -eq 0 ]]; then
    die "No HF_TOKEN / HF_TOKEN_FILE set and weights for $MODEL_ID are not cached yet in $MODEL_CACHE_DIR."
fi

# Free-space warning if we still need to download.
if [[ "$weights_cached" -eq 0 ]]; then
    free_gb="$(df -Pk "$MODEL_CACHE_DIR" | awk 'NR==2 {print int($4/1024/1024)}')"
    if [[ "${free_gb:-0}" -lt 900 ]]; then
        warn "Only ${free_gb} GB free in $MODEL_CACHE_DIR; $MODEL_ID needs ~750 GB. Download may fail."
    fi
    log "Weights not cached yet — first start will download ~750 GB. Consider './serve-glm52.sh download' first."
else
    log "Found cached weights for $MODEL_ID."
fi

# ── Download mode ───────────────────────────────────────────────────────────

if [[ "$MODE" == "download" ]]; then
    log "Prefetching $MODEL_ID into $MODEL_CACHE_DIR (no GPU required) ..."
    podman run --rm --name "${CONTAINER_NAME}-download" --replace \
        --network=host \
        -v "$MODEL_CACHE_DIR":/root/.cache/huggingface \
        -e HF_HOME=/root/.cache/huggingface \
        -e HF_TOKEN \
        -e HF_HUB_ENABLE_HF_TRANSFER=1 \
        "$SGLANG_IMAGE" \
        sh -c "hf download '$MODEL_ID' || huggingface-cli download '$MODEL_ID'"
    log "Download complete."
    exit 0
fi

# ── Bench mode ──────────────────────────────────────────────────────────────
#
# Measures the running server so config changes can be judged rather than
# guessed at. Run it on the serving node while the server is up.

if [[ "$MODE" == "bench" ]]; then
    curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
        || die "No healthy server on http://127.0.0.1:${PORT}. Start it with './serve-glm52.sh serve --detach' first."

    bench_key="${SGLANG_API_KEY:-}"
    if [[ -z "$bench_key" && -r "$MODEL_CACHE_DIR/glm52-api-key" ]]; then
        bench_key="$(<"$MODEL_CACHE_DIR/glm52-api-key")"
    fi

    # Defaults approximate a coding-agent load: long shared prefix, short reply.
    BENCH_PROMPTS="${BENCH_PROMPTS:-200}"
    BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-8192}"
    BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-512}"
    BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-16}"

    log "Benchmarking: $BENCH_PROMPTS prompts, ${BENCH_INPUT_LEN}->${BENCH_OUTPUT_LEN} tokens, concurrency $BENCH_CONCURRENCY"
    podman run --rm --name "${CONTAINER_NAME}-bench" --replace \
        --network=host \
        -e SGLANG_API_KEY="$bench_key" \
        "$SGLANG_IMAGE" \
        python3 -m sglang.bench_serving \
            --backend sglang \
            --host 127.0.0.1 --port "$PORT" \
            --model "$SERVED_MODEL_NAME" \
            --dataset-name random \
            --num-prompts "$BENCH_PROMPTS" \
            --random-input "$BENCH_INPUT_LEN" \
            --random-output "$BENCH_OUTPUT_LEN" \
            --max-concurrency "$BENCH_CONCURRENCY"
    exit 0
fi

# ── Serve-mode preflight (GPU node checks) ──────────────────────────────────

[[ -e /dev/kfd ]] || die "/dev/kfd not found — is this a ROCm GPU node?"
[[ -e /dev/dri ]] || die "/dev/dri not found — is this a ROCm GPU node?"
if [[ ! -r /dev/kfd || ! -w /dev/kfd ]]; then
    warn "/dev/kfd is not accessible by $(id -un). You likely need membership in the 'render' (and possibly 'video') group."
fi

if command -v rocminfo >/dev/null 2>&1; then
    gfx="$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-f]*' || true)"
    case "$gfx" in
        gfx942) log "Detected gfx942 (MI300X/MI325X) — matches the configured image/model." ;;
        gfx950) warn "Detected gfx950 (MI350X/MI355X). Consider the MXFP4 setup instead:
        MODEL_ID=amd/GLM-5.2-MXFP4  SGLANG_IMAGE=lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260731  TP_SIZE=4" ;;
        "")     warn "Could not detect GPU arch from rocminfo." ;;
        *)      warn "Detected $gfx — this recipe is tuned for gfx942 (MI325X)." ;;
    esac
fi
# Determine which GPUs SLURM actually allocated. MI300X/MI325X ship as 8-GPU
# nodes, but on a shared/contended node you may only get a slice — honor that so
# we never touch GPUs belonging to other users' jobs.
GPU_VIS="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
alloc_count=""
if [[ -n "$GPU_VIS" ]]; then
    alloc_count="$(awk -F, '{print NF}' <<<"$GPU_VIS")"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    alloc_count="$SLURM_GPUS_ON_NODE"
elif command -v rocm-smi >/dev/null 2>&1; then
    alloc_count="$(rocm-smi --showid 2>/dev/null | grep -c '^GPU\[' || true)"
fi

# Resolve TP_SIZE=auto to the number of GPUs allocated to this job.
if [[ "$TP_SIZE" == "auto" ]]; then
    if [[ -n "$alloc_count" && "$alloc_count" -gt 0 ]]; then
        TP_SIZE="$alloc_count"
        log "TP_SIZE=auto -> $TP_SIZE (GPUs allocated to this job)."
    else
        TP_SIZE=8
        warn "TP_SIZE=auto but could not detect the allocation; defaulting to 8."
    fi
fi

[[ -n "$GPU_VIS" ]] && log "Restricting to allocated GPUs [$GPU_VIS]."
if [[ -n "$alloc_count" && "$alloc_count" -gt 0 && "$TP_SIZE" -gt "$alloc_count" ]]; then
    warn "TP_SIZE=$TP_SIZE but only $alloc_count GPU(s) allocated — SGLang will fail. Lower TP_SIZE or request more GPUs."
fi
# ~750 GB of FP8 weights must fit: MI300X (192 GB) needs all 8; MI325X (256 GB) needs >=4.
if [[ "$TP_SIZE" -lt 4 ]]; then
    warn "TP_SIZE=$TP_SIZE is almost certainly too small — GLM-5.2-FP8 weights (~750 GB) need at least 4x256GB (MI325X) or 8x192GB (MI300X)."
fi

# ── Thread budget ───────────────────────────────────────────────────────────
#
# This is where weight loading fails on a many-core node. Each of the TP_SIZE
# ranks runs an OpenMP pool plus a loader pool of up to min(32, cores+4)
# threads, and under cgroup v2 threads count against the pids limit. Left
# alone, 8 ranks on a 192-core node want well over podman's default 2048 and
# Python raises "can't start new thread" partway through loading the shards.

host_cores="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)"

# Divide the node's cores between the ranks rather than letting each rank size
# its OpenMP pool to the whole machine.
if [[ "$OMP_THREADS" == "auto" ]]; then
    if [[ "$host_cores" -gt 0 && "$TP_SIZE" -gt 0 ]]; then
        OMP_THREADS=$(( host_cores / TP_SIZE ))
        (( OMP_THREADS < 1 )) && OMP_THREADS=1
    else
        OMP_THREADS=""
        warn "Could not detect the core count; leaving OMP_NUM_THREADS unset."
    fi
fi

nproc_soft="$(ulimit -u 2>/dev/null || echo unknown)"
log "Threads: $host_cores cores, TP=$TP_SIZE, OMP_NUM_THREADS=${OMP_THREADS:-unset}, ulimit -u=$nproc_soft, --pids-limit=$PIDS_LIMIT"

if [[ -n "$OMP_THREADS" ]]; then
    # Each rank: its OpenMP pool, plus the loader pool, plus slack for RCCL and
    # the interpreter. Deliberately a rough floor, not a precise model.
    loader_threads=32
    (( host_cores + 4 < loader_threads )) && loader_threads=$(( host_cores + 4 ))
    want_threads=$(( TP_SIZE * (OMP_THREADS + loader_threads + 16) ))
    log "Estimated peak thread count during load: ~$want_threads"

    if [[ "$nproc_soft" =~ ^[0-9]+$ && "$want_threads" -gt "$nproc_soft" ]]; then
        warn "ulimit -u is $nproc_soft but loading may want ~$want_threads threads."
        warn "  Raise it ('ulimit -u $(( want_threads * 2 ))') or lower OMP_THREADS, or startup will die with 'can't start new thread'."
    fi
    if [[ "$PIDS_LIMIT" =~ ^[0-9]+$ && "$PIDS_LIMIT" -gt 0 && "$want_threads" -gt "$PIDS_LIMIT" ]]; then
        warn "PIDS_LIMIT=$PIDS_LIMIT is below the ~$want_threads threads loading may want. Set PIDS_LIMIT=-1."
    fi
fi

# Port free?
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    exec 3>&- 3<&- || true
    die "Port $PORT is already in use on this node (another server running? try './serve-glm52.sh status')."
fi

# ── API key ─────────────────────────────────────────────────────────────────

API_KEY_FILE="$MODEL_CACHE_DIR/glm52-api-key"
if [[ -z "${SGLANG_API_KEY:-}" ]]; then
    if [[ -r "$API_KEY_FILE" ]]; then
        SGLANG_API_KEY="$(<"$API_KEY_FILE")"
        log "Using API key from $API_KEY_FILE"
    else
        SGLANG_API_KEY="$(openssl rand -hex 24 2>/dev/null || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
        (umask 077 && printf '%s' "$SGLANG_API_KEY" > "$API_KEY_FILE")
        log "Generated new API key and saved it to $API_KEY_FILE"
    fi
fi

# ── Launch ──────────────────────────────────────────────────────────────────

sglang_flags=()

[[ "$ENABLE_AITER_ALLREDUCE_FUSION" == "1" ]] && sglang_flags+=(--enable-aiter-allreduce-fusion)

# Leaving --mem-fraction-static off lets SGLang derive it from the GPU's memory
# capacity, its chunked-prefill size and its decode CUDA-graph batch size.
if [[ -n "$MEM_FRACTION" ]]; then
    sglang_flags+=(--mem-fraction-static "$MEM_FRACTION")
    log "mem-fraction-static pinned to $MEM_FRACTION (unset MEM_FRACTION to let SGLang size it)."
else
    log "mem-fraction-static left to SGLang's per-GPU heuristic."
fi

if [[ "$ENABLE_METRICS" == "1" ]]; then
    # /metrics on $PORT; cache report adds prefix-cache hit rates to responses.
    sglang_flags+=(--enable-metrics --enable-cache-report)
fi

if [[ "$ENABLE_MTP" == "1" ]]; then
    log "MTP speculative decoding ON (steps=$MTP_NUM_STEPS, draft tokens=$MTP_NUM_DRAFT_TOKENS)."
    sglang_flags+=(
        --speculative-algorithm EAGLE
        --speculative-num-steps "$MTP_NUM_STEPS"
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens "$MTP_NUM_DRAFT_TOKENS"
    )
fi

if [[ "$ENABLE_HICACHE" == "1" ]]; then
    if [[ "$USE_AITER" != "1" ]]; then
        warn "ENABLE_HICACHE=1 with USE_AITER=0: SGLang will pin the DSA page size to 1, which the page_first_direct layout needs 64 for. Expect poor results."
    fi
    log "HiCache ON (host-RAM KV tier, ratio $HICACHE_RATIO)."
    sglang_flags+=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-mem-layout page_first_direct
        --hicache-io-backend direct
        --hicache-write-policy write_through
    )
fi

[[ -n "$SCHEDULE_POLICY" ]] && sglang_flags+=(--schedule-policy "$SCHEDULE_POLICY")

# shellcheck disable=SC2206  # intentional word splitting of user-provided extra args
extra_args=($EXTRA_SGLANG_ARGS)

# Forward the SLURM GPU-visibility vars into the container so ROCm only uses the
# GPUs allocated to this job (we map all of /dev/dri, so these are what scope it).
gpu_env=()
[[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && gpu_env+=(-e ROCR_VISIBLE_DEVICES)
[[ -n "${HIP_VISIBLE_DEVICES:-}"  ]] && gpu_env+=(-e HIP_VISIBLE_DEVICES)
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && gpu_env+=(-e CUDA_VISIBLE_DEVICES)

perf_env=(-e "SGLANG_USE_AITER=$USE_AITER")
[[ -n "$OMP_THREADS" ]] && perf_env+=(-e "OMP_NUM_THREADS=$OMP_THREADS")
[[ "$AITER_AOT_GLUON" == "1" ]] && perf_env+=(-e AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1)
if [[ "$USE_AITER" == "1" ]]; then
    log "SGLANG_USE_AITER=1 — expect 'Setting page size to 64 for DeepSeek DSA' in the log."
else
    warn "USE_AITER=0: SGLang will run GLM-5.2's sparse attention at page_size=1. This is much slower; only do this to isolate an aiter bug."
fi

# --kv-cache-dtype fp8_e4m3 is deliberate, not inherited boilerplate: SGLang's
# DSA default keys off CUDA compute capability and would pick bfloat16 on
# gfx942. The tilelang DSA backend it selects on ROCm supports an fp8 KV cache
# (that path is ROCm-only), so we get half-size KV for free.
log "Starting SGLang: $MODEL_ID  (TP=$TP_SIZE, ctx=$CONTEXT_LEN, port=$PORT)"
log "Image: $SGLANG_IMAGE"

podman run -d --rm --name "$CONTAINER_NAME" --replace \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add keep-groups \
    --network=host --shm-size="$SHM_SIZE" \
    --pids-limit "$PIDS_LIMIT" \
    -v "$MODEL_CACHE_DIR":/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    "${perf_env[@]}" \
    ${gpu_env[@]+"${gpu_env[@]}"} \
    "$SGLANG_IMAGE" \
    python3 -m sglang.launch_server \
        --model-path "$MODEL_ID" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --tp "$TP_SIZE" \
        --host 0.0.0.0 --port "$PORT" \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --kv-cache-dtype fp8_e4m3 \
        --context-length "$CONTEXT_LEN" \
        --api-key "$SGLANG_API_KEY" \
        --trust-remote-code \
        ${sglang_flags[@]+"${sglang_flags[@]}"} \
        ${extra_args[@]+"${extra_args[@]}"} \
    >/dev/null

cleanup() {
    log "Shutting down $CONTAINER_NAME ..."
    podman stop -t 30 "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ── Wait for readiness ──────────────────────────────────────────────────────

log "Waiting for the server to become healthy (timeout ${READY_TIMEOUT}s; model load takes several minutes, first-run download much longer) ..."
log "Follow detailed progress in another shell with: podman logs -f $CONTAINER_NAME"

start_ts="$(date +%s)"
while true; do
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    state="$(podman inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "gone")"
    if [[ "$state" != "running" ]]; then
        echo
        podman logs --tail 50 "$CONTAINER_NAME" 2>&1 || true
        die "Container exited during startup (state: $state). Last 50 log lines above."
    fi
    if (( $(date +%s) - start_ts > READY_TIMEOUT )); then
        die "Server did not become healthy within ${READY_TIMEOUT}s. Check: podman logs $CONTAINER_NAME"
    fi
    sleep 10
done

# /health only proves the HTTP server is listening. /health_generate runs a real
# forward pass, so it also covers weight load, CUDA-graph capture and the DSA
# kernels. Done once, after /health, rather than polled -- it costs a generation.
log "HTTP server is up; confirming the model can actually generate ..."
if curl -fsS -m 300 "http://127.0.0.1:${PORT}/health_generate" \
        -H "Authorization: Bearer $SGLANG_API_KEY" >/dev/null 2>&1; then
    log "Generation health check passed."
else
    warn "/health_generate did not pass. The endpoint may still be capturing graphs — check 'podman logs -f $CONTAINER_NAME' before sending real traffic."
fi

# Surface which DSA path SGLang chose; page size 1 means the aiter fast path
# was unavailable and throughput will be well below what this node can do.
dsa_page_line="$(podman logs "$CONTAINER_NAME" 2>&1 | grep -m1 'page size to .* for DeepSeek DSA' || true)"
if [[ -n "$dsa_page_line" ]]; then
    if [[ "$dsa_page_line" == *"page size to 1"* ]]; then
        warn "SGLang fell back to DSA page_size=1. Set AITER_AOT_GLUON=1 in glm52.env and restart."
        warn "  $dsa_page_line"
    else
        log "DSA fast path active: ${dsa_page_line#*] }"
    fi
fi

# ── Connection banner ───────────────────────────────────────────────────────

NODE_HOST="$(hostname -f 2>/dev/null || hostname)"
cat <<EOF

============================================================================
  GLM-5.2 is up and serving.

  Node:        $NODE_HOST
  Endpoint:    http://$NODE_HOST:$PORT/v1   (OpenAI-compatible)
  Model name:  $SERVED_MODEL_NAME
  API key:     $API_KEY_FILE
               export SGLANG_API_KEY="\$(cat $API_KEY_FILE)"

  Smoke test (from this node or the login node):
    curl -s http://$NODE_HOST:$PORT/v1/models \\
         -H "Authorization: Bearer \$SGLANG_API_KEY"

  From your laptop (tunnel through the login node, then use localhost):
    ssh -N -L $PORT:$NODE_HOST:$PORT <user>@<login-node>

  opencode: run ./opencode-setup.sh --host $NODE_HOST --port $PORT
            (or --host localhost when tunnelling), then pick
            '$SERVED_MODEL_NAME' via /models inside opencode.

  Metrics:  http://$NODE_HOST:$PORT/metrics   (Prometheus)
  Bench:    ./serve-glm52.sh bench

  Stop with Ctrl-C, 'scancel <jobid>', or './serve-glm52.sh stop'.
============================================================================

EOF

if [[ "$DETACH" -eq 1 ]]; then
    # Disarm the cleanup traps: the container keeps running in the background
    # (until './serve-glm52.sh stop' or the SLURM job/allocation ends).
    trap - EXIT INT TERM
    log "Detached. You have your shell back — the server keeps running on this node."
    log "  Logs:  podman logs -f $CONTAINER_NAME"
    log "  Stop:  ./serve-glm52.sh stop"
    exit 0
fi

# Stay attached: keeps the SLURM job alive and tears the container down on
# Ctrl-C / scancel via the traps above. Background + wait so signals are
# handled promptly.
podman logs -f "$CONTAINER_NAME" &
wait $!
