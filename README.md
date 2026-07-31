# GLM-5.2 on AMD MI325X — SGLang serving for opencode

Serve [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) from a single 8×
AMD MI325X node on a SLURM cluster, in a **podman** container, via **SGLang**,
exposing an OpenAI-compatible endpoint you can drive with
[opencode](https://opencode.ai) — from the node itself, the login node, your
laptop over SSH, or (optionally) anyone over a public HTTPS tunnel.

Modeled on [wafer.ai's GLM-5.2-on-AMD writeup](https://www.wafer.ai/blog/glm52-amd).
Tested on 8× MI325X (gfx942), ROCm 7.2.4, Ubuntu, rootless podman.

This README is a complete walkthrough — if you have SSH access to the cluster
and an account, you can go from clone to a working coding endpoint by following
it top to bottom.

---

## What gets served

| | |
|---|---|
| Model | `zai-org/GLM-5.2-FP8` (753B MoE), served under the name `glm-5.2` |
| Architecture | `GlmMoeDsaForCausalLM` — MLA attention + DeepSeek Sparse Attention (`index_topk` 2048) |
| Engine | SGLang, pinned ROCm image `lmsysorg/sglang-rocm:v0.5.16-rocm720-mi30x-20260731` |
| Parallelism | TP8 across the node's 8 GPUs |
| Context | 262,144 tokens by default (`CONTEXT_LEN`; model supports up to 1M) |
| KV cache | FP8 (`fp8_e4m3`), MLA-compressed, tiered to host RAM if `ENABLE_HICACHE=1` |
| Tool calling / thinking | `--tool-call-parser glm47 --reasoning-parser glm45` — required for opencode's agentic loop |
| Observability | Prometheus `/metrics` + prefix-cache hit rates (`ENABLE_METRICS`, on by default) |
| Auth | Bearer API key, auto-generated and persisted to `$MODEL_CACHE_DIR/glm52-api-key` |

**Why FP8 and not the blog's MXFP4?** The wafer.ai post ran the MXFP4 quant on
MI355X. That quant (`amd/GLM-5.2-MXFP4`) requires gfx950 silicon, so on MI325X
(gfx942) this repo serves the FP8 variant at TP8 instead — ~755 GB of weights
across 2 TB of HBM. The container ships its own ROCm 7.2 userspace, which runs
on the host's ROCm 7.2.4 (only the kernel driver is shared). If you get MI355X
time, one config change flips to the blog's exact setup — see
[Running on MI355X instead](#running-on-mi355x-instead).

**A note on the KV cache.** GLM-5.2 uses MLA, so the KV cache is compressed to a
512-dim latent plus a 64-dim rope key per token per layer — perhaps two orders of
magnitude smaller than a comparably-sized MHA model. Combined with sparse
attention (each query attends to at most 2048 keys), this is why 262 k context is
affordable here and why the leftover HBM after weights goes a very long way. It
also means KV capacity is rarely the binding constraint; see
[Performance tuning](#performance-tuning) for what actually is.

## Hardware & how many GPUs you need

**MI300X works identically to MI325X** — both are gfx942 and use the same
container image (the `mi30x` tag *is* the MI300X image). The only practical
difference is HBM per GPU (MI300X 192 GB, MI325X 256 GB), which sets how many
GPUs you need, because the ~750 GB of FP8 weights are fixed and must all be
resident (mixture-of-experts still loads every expert). Tensor parallelism
splits the weights evenly, so per-GPU weight load ≈ 750 GB ÷ N; whatever's left
becomes KV cache (your context × concurrency budget).

| GPUs | MI300X (192 GB) | MI325X (256 GB) |
|---|---|---|
| **8** | ✅ ~90 GB/GPU weights, ~560 GB KV — recommended | ✅ tons of headroom |
| **4** | ❌ weights ≈ whole node, no room | ✅ ~120 GB KV — works, modest |
| **2** | ❌ | ❌ |

So the **bare minimum is 4 GPUs on MI325X, 8 GPUs on MI300X**. `TP_SIZE` must
evenly divide the model's attention heads, so use **4 or 8 — not 6**.

These parts only ship as **8-GPU nodes**, but on a shared/contended cluster you
may only be granted a slice. That's fine: request the GPUs you can get and leave
`TP_SIZE=auto` (the default) — the script reads your SLURM allocation, sizes
tensor parallelism to match, and forwards the GPU-visibility variables into the
container so it never touches GPUs belonging to other jobs on the same node.

```bash
# grab a 4-GPU slice of a shared MI325X node:
salloc -N1 --gres=gpu:4        # no --exclusive; TP_SIZE=auto -> 4
```

On MI300X, only a full 8-GPU allocation fits — a 4-GPU slice can't hold the
weights, and the script will warn you if you try.

---

## Prerequisites

Before you start, confirm you have:

- **A SLURM account** on a cluster with an 8×MI325X (or MI300X) node, and the
  right to allocate a whole node (`--exclusive`, all 8 GPUs).
- **`podman`** available on the compute node, with your user able to reach
  `/dev/kfd` and `/dev/dri` (usually via membership in the `render`, sometimes
  also `video`, group — run `id` to check, ask your admins if unsure).
- **A large shared filesystem** the compute nodes can see, with **~800 GB free**
  for the model weights (scratch or project space — *not* your home directory).
- **A HuggingFace account + access token** (read scope). Create one at
  <https://huggingface.co/settings/tokens>. GLM-5.2-FP8 is a gated/large repo;
  make sure you can view <https://huggingface.co/zai-org/GLM-5.2-FP8> while
  logged in.
- **`opencode`** installed wherever you want to *use* the model (the node, the
  login node, or your laptop) — see <https://opencode.ai>.
- Optional, for a nicer opencode config merge: **`jq`** on the machine running
  `opencode-setup.sh`.

---

## Repository contents

| File | Purpose |
|---|---|
| `serve-glm52.sh` | The core one-click script: preflight, download, serve, bench, stop, status |
| `serve-glm52.sbatch` | SLURM batch wrapper around `serve-glm52.sh serve` |
| `glm52-env.example` | Config template — copy to `glm52.env` and edit |
| `opencode-setup.sh` | Writes/merges the opencode provider config on any machine |
| `opencode.glm52.json` | The provider template `opencode-setup.sh` fills in |
| `share-glm52.sh` | Optional: public HTTPS tunnel via Cloudflare for users without SSH |
| `.github/workflows/lint.yml` | CI: shellcheck + opencode config-template checks |
| `README.md` | This file |

Secrets never live in the repo: `glm52.env` (your HF token) and the generated
API key are gitignored / stored under `$MODEL_CACHE_DIR`.

---

## Walkthrough (primary path — you have SSH access)

### Step 0 — Get the code onto the cluster

Clone it (or `scp` the directory) to a location the login **and** compute nodes
can see — your home directory is usually fine for the *scripts* (the big weights
go elsewhere, in Step 1):

```bash
git clone <this-repo-url> glm52-mi325x
cd glm52-mi325x
```

### Step 1 — Configure

```bash
cp glm52-env.example glm52.env
$EDITOR glm52.env
```

At minimum set two values:

- `MODEL_CACHE_DIR` — an absolute path on your big shared filesystem, e.g.
  `/scratch/<project>/glm52/hf-cache`. This is where ~750 GB of weights and the
  API key will live. It must be readable/writable from the compute node.
- `HF_TOKEN` — your HuggingFace token. (Or leave it blank and set `HF_TOKEN_FILE`
  to a path containing just the token — handy if you don't want the token inline.)

Everything else has working defaults for an 8×MI325X node (see
[Configuration reference](#configuration-reference)).

### Step 2 — Prefetch the weights (do this first)

Run this on the **login node** — it needs no GPU and downloads ~750 GB, which
you don't want to do inside a paid GPU allocation:

```bash
source glm52.env               # so MODEL_CACHE_DIR / HF_TOKEN are set
./serve-glm52.sh download
```

This pulls the container image and the model into `$MODEL_CACHE_DIR`. It can
take a while depending on your link; it's resumable (rerun if interrupted).

> If your login node has no outbound internet or can't run podman, run this same
> command inside an interactive allocation instead (Step 3), before the first
> `serve`.

### Step 3 — Allocate a node and serve

Grab a whole node and start the server. GRES naming is site-specific — if
`--gres=gpu:8` is rejected, try `--gpus=8` or `--gres=gpu:mi325x:8` (ask
`sinfo -o "%G"`).

```bash
salloc -N1 --exclusive --gres=gpu:8
# you're now on (or attached to) the allocated node; if not, srun --pty bash
cd ~/glm52-mi325x
./serve-glm52.sh serve --detach
```

`--detach` starts the container, waits until the model is loaded and healthy
(watch progress with `podman logs -f glm52-sglang` in another shell), prints a
connection banner, and **returns your shell** so you can run opencode right there
on the node. Drop `--detach` if you'd rather the script stay attached and tear
the container down on Ctrl-C.

First healthy startup with cached weights takes ~10–20 minutes (loading 750 GB
across 8 GPUs + CUDA-graph capture).

**Prefer batch?** Instead of `salloc`, submit the job:

```bash
mkdir -p logs
sbatch serve-glm52.sbatch
# once RUNNING, read node + endpoint + key from the job log:
grep -A 20 'GLM-5.2 is up' logs/glm52-<jobid>.out
```

The batch job serves until walltime or `scancel <jobid>` (SIGTERM is trapped and
the container stops cleanly).

### Step 4 — Verify it's serving

The banner prints the exact commands, but in short (from the node or, if your
cluster routes login→compute, the login node):

```bash
export SGLANG_API_KEY="$(cat $MODEL_CACHE_DIR/glm52-api-key)"
curl -s http://<node>:30000/v1/models -H "Authorization: Bearer $SGLANG_API_KEY"

curl -s http://<node>:30000/v1/chat/completions \
  -H "Authorization: Bearer $SGLANG_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Say hi in one word."}]}'
```

The first lists `glm-5.2`; the second returns a completion. `<node>` is the
allocated hostname (`hostname` on the node, or the `%N` column of
`squeue -u $USER`).

### Step 5 — Connect opencode

Pick the vantage point that matches where you run opencode. In every case
`opencode-setup.sh` writes/merges a `glm52-local` provider into
`~/.config/opencode/opencode.json` (safe `jq` merge if available; otherwise it
prints the block to paste). The config references `{env:SGLANG_API_KEY}`, so
export that variable in the shell that launches opencode. Then restart opencode
and pick **GLM 5.2 (MI325X/SGLang)** via `/models`.

It also points opencode's `model` and `small_model` at `glm-5.2` — but only if
you haven't already set them, so an existing config is never overridden. The
`small_model` part matters on a compute node: opencode uses it for titles and
summarisation, and left unset it reaches for a cloud provider the node usually
cannot see.

**A) On the GPU node itself** (simplest — pairs with `serve --detach`):

```bash
./opencode-setup.sh --host localhost
export SGLANG_API_KEY="$(cat $MODEL_CACHE_DIR/glm52-api-key)"
opencode
```

**B) On the login node** (only if the cluster network routes login→compute):

```bash
./opencode-setup.sh --host <node>
export SGLANG_API_KEY="$(cat $MODEL_CACHE_DIR/glm52-api-key)"
opencode
```

**C) On your laptop** (tunnel through the login node — works everywhere):

```bash
# in one terminal, keep this open:
ssh -N -L 30000:<node>:30000 <user>@<login-node>

# in another: (grab the key value from the cluster first)
./opencode-setup.sh --host localhost --api-key <key>
export SGLANG_API_KEY=<key>
opencode
```

Once selected, ask opencode to read/edit a file or run a command — tool calls and
reasoning are handled server-side by the `glm47`/`glm45` parsers, so the full
agentic loop just works.

### Step 6 — Shut down

```bash
./serve-glm52.sh stop        # stop the container (or just let the allocation end)
scancel <jobid>              # for a batch job
```

---

## Sharing with someone who has no SSH access (optional)

`share-glm52.sh` exposes the running endpoint over public HTTPS via a
**Cloudflare quick tunnel** — outbound-only (works on locked-down nodes), no
root, no Cloudflare account. Run it on the GPU node after the server is up:

```bash
./share-glm52.sh share --detach     # prints https://<random>.trycloudflare.com
```

On first use it downloads `cloudflared` into `$MODEL_CACHE_DIR/cloudflared/`,
checks the local server is healthy, opens the tunnel, and prints a ready-to-paste
opencode provider block (with the public URL and API key) to hand over. The
recipient drops it into their `~/.config/opencode/opencode.json`, restarts
opencode, and picks **GLM 5.2 (shared)** via `/models` — no SSH, no tunnel, no
cluster account on their end. Manage with `./share-glm52.sh status` / `stop`.

> ⚠️ **The public URL + API key together grant full use of your model and your
> cluster's GPU-hours.** Share the key over a private channel only, rotate it if
> it leaks (delete `$MODEL_CACHE_DIR/glm52-api-key` and restart the server), and
> **check your site's acceptable-use policy before exposing HPC compute
> externally** — the API key is the only gate. Quick-tunnel URLs are random and
> change on every restart; for a stable address, use Tailscale or a named
> Cloudflare tunnel with your own domain.

---

## Script reference

```
./serve-glm52.sh [serve]         start serving (default), stays attached
./serve-glm52.sh serve --detach  start serving, wait until healthy, return the shell
./serve-glm52.sh download        prefetch image + weights only (no GPU)
./serve-glm52.sh bench           benchmark the running server (sglang.bench_serving)
./serve-glm52.sh stop            stop the server container
./serve-glm52.sh status          container state + health check

./opencode-setup.sh [--host H] [--port P] [--api-key K] [--embed-key] [--config PATH]
                                 write/merge the opencode provider config
                                 (--embed-key writes the key literally instead of {env:...})

./share-glm52.sh [share]         open a public HTTPS Cloudflare tunnel (add --detach to background)
./share-glm52.sh stop            take the tunnel offline
./share-glm52.sh status          tunnel state + current public URL
```

Handy extras:

```bash
podman logs -f glm52-sglang                         # follow server startup / requests
srun --overlap --pty --jobid <jobid> bash           # second shell on the serving node
```

---

## Configuration reference

All knobs live in `glm52.env` (copied from `glm52-env.example`). Anything you
`export` in your shell before running a script takes precedence over the file.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_CACHE_DIR` | *(required)* | Big shared-FS path for HF cache + API key (~800 GB free) |
| `HF_TOKEN` | *(required first run)* | HuggingFace token for the weights download |
| `HF_TOKEN_FILE` | — | Alternative to `HF_TOKEN`: path to a file containing just the token |
| `SGLANG_API_KEY` | *(auto-generated)* | Endpoint bearer key; if unset, generated and saved to `$MODEL_CACHE_DIR/glm52-api-key` |
| `MODEL_ID` | `zai-org/GLM-5.2-FP8` | Model repo to serve |
| `SERVED_MODEL_NAME` | `glm-5.2` | Name clients use in the `model` field |
| `SGLANG_IMAGE` | `lmsysorg/sglang-rocm:v0.5.16-rocm720-mi30x-20260731` | Container image |
| `PORT` | `30000` | Endpoint port on the node |
| `TP_SIZE` | `auto` | Tensor-parallel degree (= GPUs used). `auto` = the GPU count SLURM allocated; or set `4`/`8` explicitly |
| `CONTEXT_LEN` | `262144` | Max context length |
| `MEM_FRACTION` | *(empty)* | `--mem-fraction-static`. Empty = let SGLang size it from GPU capacity. Set a number only to work around OOM |
| `SHM_SIZE` | `64g` | Container `/dev/shm` size |
| `USE_AITER` | `1` | `SGLANG_USE_AITER`. Required for the fast DSA path — see [Performance tuning](#performance-tuning) |
| `AITER_AOT_GLUON` | `0` | `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS`. Set `1` if the log still reports DSA page size 1 |
| `ENABLE_AITER_ALLREDUCE_FUSION` | `1` | Toggle `--enable-aiter-allreduce-fusion` (set `0` if allreduce crashes) |
| `ENABLE_METRICS` | `1` | `--enable-metrics --enable-cache-report` |
| `ENABLE_MTP` | `0` | MTP speculative decoding (opt-in, benchmark it) |
| `MTP_NUM_STEPS` / `MTP_NUM_DRAFT_TOKENS` | `3` / `4` | MTP depth |
| `ENABLE_HICACHE` | `0` | Host-RAM KV cache tier (opt-in, benchmark it) |
| `HICACHE_RATIO` | `2` | Host memory as a multiple of GPU KV memory |
| `SCHEDULE_POLICY` | — | Set `lpm` to reorder for prefix-cache hits on a shared endpoint |
| `READY_TIMEOUT` | `7200` | Seconds to wait for health before giving up |
| `EXTRA_SGLANG_ARGS` | — | Extra flags appended verbatim to `sglang.launch_server` |

Benchmark knobs for `./serve-glm52.sh bench`: `BENCH_PROMPTS` (200),
`BENCH_INPUT_LEN` (8192), `BENCH_OUTPUT_LEN` (512), `BENCH_CONCURRENCY` (16).

---

## Performance tuning

### The one that matters: `USE_AITER=1`

GLM-5.2 is a DeepSeek-Sparse-Attention model. On ROCm, SGLang picks the DSA KV
page size like this ([`arg_groups/overrides.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/overrides.py)):

```python
if is_hip() and not aiter_can_use_preshuffle_paged_mqa():
    overrides["page_size"] = 1     # legacy ROCm DSA path
else:
    overrides["page_size"] = 64
```

`aiter_can_use_preshuffle_paged_mqa()` returns `False` unless `SGLANG_USE_AITER`
is set, and upstream it **defaults to off**. Unset, you get one token per KV
page, no zero-copy preshuffle gather, and HiCache's page-first layouts become
unusable. `USE_AITER=1` is the default here for that reason; `serve-glm52.sh`
greps the startup log after launch and warns if SGLang still chose page size 1.
If it does, set `AITER_AOT_GLUON=1` (the preshuffle path also wants Triton ≥
3.5 in the image).

Note that `--enable-aiter-allreduce-fusion` is a *different* switch — it tunes
collectives, not attention, and does not enable this path.

### Let SGLang size the KV pool

SGLang computes `mem_fraction_static` from GPU capacity, chunked-prefill size
and decode CUDA-graph batch size — but **only when the flag is absent**. Pinning
it (this repo used to pin 0.85) caps a 256 GB MI325X roughly 38 GB per GPU below
what the heuristic allows, ~300 GB across the node. Leave `MEM_FRACTION` empty
unless you are chasing an OOM.

### Optional, measure before trusting

Both are off by default. Turn on one at a time and compare with
`./serve-glm52.sh bench`.

- **`ENABLE_MTP=1`** — MTP speculative decoding. The FP8 checkpoint ships the
  NextN draft layer (`model.layers.78.eh_proj` / `.enorm` / `.hnorm`), so no
  separate draft model is needed. Off by default only because the DSA + MTP
  combination is not yet proven on gfx942 — if it works on your node it is the
  biggest single-stream latency win available. (NGRAM speculation is CUDA-only;
  don't bother.)
- **`ENABLE_HICACHE=1`** — tiers the radix prefix cache into host RAM using the
  AMD-friendly `page_first_direct` layout. Well suited to opencode, which
  re-sends a near-identical prefix every turn. Needs the page-size-64 path
  above, so keep `USE_AITER=1` with it.

### Knowing whether it worked

`ENABLE_METRICS=1` (default) exposes Prometheus at `http://<node>:$PORT/metrics`
and adds prefix-cache hit rates to responses. The cache hit rate is the number to
watch for an agent workload — if it is low, opencode is paying full prefill on
every turn and HiCache / `SCHEDULE_POLICY=lpm` are the levers.

---

## Running on MI355X instead

If you get MI355X (gfx950) time and want the blog's exact MXFP4 setup, set these
in `glm52.env`:

```bash
export MODEL_ID="amd/GLM-5.2-MXFP4"
export SGLANG_IMAGE="lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260731"
export TP_SIZE=4
export EXTRA_SGLANG_ARGS="--dp 2 --enable-dp-attention"
```

MXFP4 also requires `SGLANG_USE_AITER=1`, which `USE_AITER=1` already sets.

`serve-glm52.sh` also detects gfx950 at runtime and reminds you of this.

---

## Notes & troubleshooting

- **Startup time**: with cached weights, ~10–20 minutes to healthy. Watch
  details with `podman logs -f glm52-sglang`. The script polls `/health` (HTTP
  server listening), then does one `/health_generate` call, which runs a real
  forward pass and so also covers weight load, graph capture and the DSA
  kernels. The banner prints after both.
- **The API key is visible in the process table.** SGLang only accepts
  `--api-key` as a command-line argument — there is no env-var or file
  equivalent — so it lands in `/proc/<pid>/cmdline`, which is world-readable on
  most clusters unless the site sets `hidepid`. On an `--exclusive` allocation
  this is moot; on a shared node, assume anyone else on that node can read your
  key. Rotate by deleting `$MODEL_CACHE_DIR/glm52-api-key` and restarting.
- **"cannot set shmsize when running in the host IPC Namespace"**: already
  handled — the script uses a private IPC namespace with `--shm-size=64g` (podman
  forbids `--shm-size` together with `--ipc=host`; Docker tolerates it).
- **Detach didn't return my shell**: make sure the copy of `serve-glm52.sh` on
  the cluster is current — `grep -c DETACH serve-glm52.sh` should be non-zero.
- **Rootless podman + GPUs**: your user needs access to `/dev/kfd` and `/dev/dri`
  (the `render`, sometimes `video`, group). The script passes `--group-add
  keep-groups --security-opt seccomp=unconfined --network=host --device=/dev/kfd
  --device=/dev/dri`. If your site uses a shared image store you may need to
  `podman pull` the ~20 GB image once per node image cache.
- **Speculative decoding**: off by default but *available* — the FP8 checkpoint
  ships the MTP/NextN draft layer, so `ENABLE_MTP=1` needs no extra weights. It
  is opt-in because the DSA + MTP path is unproven on gfx942, not because the
  weights are missing. Benchmark it; see
  [Performance tuning](#performance-tuning).
- **`--enable-aiter-allreduce-fusion`** comes from the wafer.ai post (validated
  there on MI355X). If you hit allreduce/RCCL crashes on MI325X, set
  `ENABLE_AITER_ALLREDUCE_FUSION=0`.
- **Port conflicts**: default `PORT=30000`; the script refuses to start if the
  port is already taken on the node.
- **Only N GPUs visible**: if the script warns that fewer than `TP_SIZE` GPUs are
  visible, your allocation didn't include all 8 — recheck your `--gres`/`--gpus`
  request.
- **opencode doesn't see the model**: it only reads config at startup — restart
  it after `opencode-setup.sh`, and make sure `SGLANG_API_KEY` is exported in
  that shell (or you used `--embed-key`).

---

## Sources

- [wafer.ai: GLM 5.2 on AMD](https://www.wafer.ai/blog/glm52-amd) — MXFP4/MI355X reference numbers (213 tok/s single stream, 2626 tok/s/node at TP4×DP2)
- [SGLang cookbook: GLM-5.2](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2) — pinned ROCm images, parser flags, AMD caveats
- [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) · [amd/GLM-5.2-MXFP4](https://huggingface.co/amd/GLM-5.2-MXFP4)
- [opencode custom providers](https://opencode.ai/docs/providers/)

---

## License

The scripts in this repo are provided under the MIT License (see `LICENSE` if
present — add one before publishing). The container images
(`lmsysorg/sglang-rocm`), the SGLang engine, and the model weights
(`zai-org/GLM-5.2-FP8` / `amd/GLM-5.2-MXFP4`) are covered by their own separate
licenses — review and comply with those independently.
