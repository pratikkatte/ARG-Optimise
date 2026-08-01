#!/bin/bash
#
# Train ARG GFlowNet on Phoenix GPU node.
#
# Submit from this directory (logs/ must exist before sbatch):
#   sbatch train.sbatch
#
#SBATCH --job-name=local_refinement_25kb_v2
#SBATCH --partition=gpu
#SBATCH --account=standard
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pkatte@ucsc.edu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source /private/home/pkatte/anaconda3/etc/profile.d/conda.sh
conda activate phylogfn_orig

export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"
nvidia-smi || true

python train.py --config config/config.yaml

echo "End: $(date)"
