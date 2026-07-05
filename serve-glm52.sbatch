#!/usr/bin/env bash
#SBATCH --job-name=glm52-sglang
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=logs/glm52-%j.out
#SBATCH --error=logs/glm52-%j.out
#
# Batch wrapper around serve-glm52.sh. Submit with:
#
#   mkdir -p logs && sbatch serve-glm52.sbatch
#
# NOTE: GRES naming is site-specific. If '--gres=gpu:8' is rejected, try one
# of these (adjust to your site's convention, ask 'sinfo -o "%G"'):
#   #SBATCH --gpus=8
#   #SBATCH --gres=gpu:mi325x:8
#
# Contended node? You can request fewer GPUs (e.g. --gres=gpu:4 and drop
# --exclusive). TP_SIZE=auto in glm52.env then matches the count. MI325X works
# on 4; MI300X needs all 8 (weights don't fit on 4x192GB). See the README.
#
# The connection banner (node hostname, endpoint URL, API key location,
# tunnel command) is printed into the job log. Once the job is running:
#
#   grep -A 20 'GLM-5.2 is up' logs/glm52-<jobid>.out
#
# The server runs until walltime or 'scancel <jobid>' (the script traps
# SIGTERM and stops the podman container cleanly).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[glm52-sbatch] Job $SLURM_JOB_ID on $(hostname) at $(date)"
exec ./serve-glm52.sh serve
