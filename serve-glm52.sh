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
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang-rocm:v0.5.13.post1-rocm700-mi30x-20260616}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-auto}"
CONTEXT_LEN="${CONTEXT_LEN:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
SHM_SIZE="${SHM_SIZE:-64g}"
ENABLE_AITER_ALLREDUCE_FUSION="${ENABLE_AITER_ALLREDUCE_FUSION:-1}"
READY_TIMEOUT="${READY_TIMEOUT:-7200}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"

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
    serve|download) ;;
    *)
        die "Unknown mode '$MODE'. Use: serve | download | stop | status"
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

# Resolve HF token: env var, then token file.
if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
    [[ -r "$HF_TOKEN_FILE" ]] || die "HF_TOKEN_FILE '$HF_TOKEN_FILE' is not readable."
    HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi

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
        MODEL_ID=amd/GLM-5.2-MXFP4  SGLANG_IMAGE=lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260618  TP_SIZE=4" ;;
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

aiter_flag=()
[[ "$ENABLE_AITER_ALLREDUCE_FUSION" == "1" ]] && aiter_flag=(--enable-aiter-allreduce-fusion)

# shellcheck disable=SC2206  # intentional word splitting of user-provided extra args
extra_args=($EXTRA_SGLANG_ARGS)

# Forward the SLURM GPU-visibility vars into the container so ROCm only uses the
# GPUs allocated to this job (we map all of /dev/dri, so these are what scope it).
gpu_env=()
[[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && gpu_env+=(-e ROCR_VISIBLE_DEVICES)
[[ -n "${HIP_VISIBLE_DEVICES:-}"  ]] && gpu_env+=(-e HIP_VISIBLE_DEVICES)
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && gpu_env+=(-e CUDA_VISIBLE_DEVICES)

log "Starting SGLang: $MODEL_ID  (TP=$TP_SIZE, ctx=$CONTEXT_LEN, port=$PORT)"
log "Image: $SGLANG_IMAGE"

podman run -d --rm --name "$CONTAINER_NAME" --replace \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add keep-groups \
    --network=host --shm-size="$SHM_SIZE" \
    -v "$MODEL_CACHE_DIR":/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
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
        --mem-fraction-static "$MEM_FRACTION" \
        --context-length "$CONTEXT_LEN" \
        --api-key "$SGLANG_API_KEY" \
        --trust-remote-code \
        "${aiter_flag[@]}" \
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
