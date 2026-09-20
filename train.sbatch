    #!/bin/bash
    #SBATCH --job-name=arg-gfn
    #SBATCH --account=standard
    #SBATCH --partition=gpu
    #SBATCH --array=1,2,4
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --gres=gpu:A100:1
    #SBATCH --mem=100G
    #SBATCH --time=48:00:00
    #SBATCH --output=logs/train-%A_%a.out
    #SBATCH --error=logs/train-%A_%a.err

    set -euo pipefail

    # ============================================================
    # Repository
    # ============================================================

    # cd /u/pratikkatte7/ARG-Optimise

    # Use Python directly from the existing "args" environment.
    # This avoids Conda initialization problems inside sbatch.
    # PYTHON="/u/pratikkatte7/.conda/envs/args/bin/python"

    cd "$HOME/aim3/iclr2017/temp/ARG-Optimise"

    # Use the Python from your Phoenix conda environment
    # PYTHON="$HOME/.conda/envs/phylogfn_orig/bin/python"
    PYTHON="/private/home/pkatte/anaconda3/envs/phylogfn_orig/bin/python"

    export PYTHONNOUSERSITE=1

    # We have 8 CPUs total and two training processes.
    # Give each process approximately 4 CPU threads.
    export OMP_NUM_THREADS=4

    # ============================================================
    # Dataset
    #
    # Array task 1 -> r1
    # Array task 2 -> r2
    # Array task 4 -> r4
    # ============================================================

    DATASET="r${SLURM_ARRAY_TASK_ID}"

    echo "============================================================"
    echo "ARG-GFN training"
    echo "============================================================"
    echo "Slurm job ID:      $SLURM_JOB_ID"
    echo "Array task ID:     $SLURM_ARRAY_TASK_ID"
    echo "Dataset:           $DATASET"
    echo "Node:              $(hostname)"
    echo "Working directory: $(pwd)"
    echo "Python:            $PYTHON"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
    echo "Started:           $(date)"
    echo "============================================================"

    # ============================================================
    # Sanity checks
    # ============================================================

    "$PYTHON" --version

    echo
    echo "=== GPU information ==="
    nvidia-smi

    echo
    echo "=== PyTorch / CUDA check ==="

    "$PYTHON" - <<'PY'
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB",
        )
    PY

    # ============================================================
    # Configuration
    # ============================================================

    CONFIG1="config/paper_datasets/sub_lambda_1/${DATASET}.yaml"
    CONFIG2="config/paper_datasets/sub_lambda_2/${DATASET}.yaml"

    echo
    echo "Lambda 1 config: $CONFIG1"
    echo "Lambda 2 config: $CONFIG2"

    # Fail immediately if either config is missing.
    test -f "$CONFIG1"
    test -f "$CONFIG2"

    # Separate logs for the two simultaneous runs.
    LAMBDA1_LOG="logs/lambda1-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
    LAMBDA2_LOG="logs/lambda2-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

    echo
    echo "Lambda 1 log: $LAMBDA1_LOG"
    echo "Lambda 2 log: $LAMBDA2_LOG"

    # ============================================================
    # Launch BOTH training runs simultaneously
    #
    # Both processes inherit the same CUDA_VISIBLE_DEVICES and
    # therefore intentionally share the A100 allocated to this job.
    # ============================================================

    echo
    echo "============================================================"
    echo "Starting BOTH training runs at $(date)"
    echo "============================================================"

    "$PYTHON" -u train.py \
        --config "$CONFIG1" \
        > "$LAMBDA1_LOG" 2>&1 &

    PID1=$!

    "$PYTHON" -u train.py \
        --config "$CONFIG2" \
        > "$LAMBDA2_LOG" 2>&1 &

    PID2=$!

    echo "Lambda 1 PID: $PID1"
    echo "Lambda 2 PID: $PID2"

    # ============================================================
    # Wait for both processes
    # ============================================================

    set +e

    wait "$PID1"
    STATUS1=$?

    wait "$PID2"
    STATUS2=$?

    set -e

    echo
    echo "============================================================"
    echo "Training finished"
    echo "============================================================"
    echo "Dataset:              $DATASET"
    echo "Lambda 1 exit status: $STATUS1"
    echo "Lambda 2 exit status: $STATUS2"
    echo "Finished:             $(date)"
    echo "============================================================"

    # Mark the Slurm job as failed if either training run failed.
    if [[ "$STATUS1" -ne 0 || "$STATUS2" -ne 0 ]]; then
        echo "ERROR: At least one training run failed." >&2
        exit 1
    fi

    echo "Both training runs completed successfully."
