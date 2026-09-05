#!/bin/bash
#
#SBATCH --job-name=local_refinement
#SBATCH --partition=gpu
#SBATCH --account=standard
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --gres=gpu:A100:1

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pkatte@ucsc.edu

#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source /private/home/pkatte/anaconda3/etc/profile.d/conda.sh
conda activate phylogfn_orig

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
    config_poc2.yaml
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "=============================="
echo "Array Task : ${SLURM_ARRAY_TASK_ID}"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Config     : ${CONFIG}"
echo "Node       : ${SLURM_NODELIST}"
echo "Start      : $(date)"
echo "=============================="

nvidia-smi || true

python train.py \
    --config "${CONFIG}" \
    --device cuda

echo "Finished: $(date)"
