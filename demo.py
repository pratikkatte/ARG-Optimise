import argparse
import os
import pickle
import random

try:
    from .env import SimpleARGEnvironment
    from .rollout_worker_arg import RolloutWorker
    from .tb_gfn import TBGFlowNetGenerator
    from .utils import load_sequences
except ImportError:
    from env import SimpleARGEnvironment
    from rollout_worker_arg import RolloutWorker
    from tb_gfn import TBGFlowNetGenerator
    from utils import load_sequences


def train_epoch(epoch_id, rollout_worker, generator, batch_size=1):
    """
    """
    batch_size = _ensure_positive_int(batch_size, "batch_size")
    ret, trajectories = rollout_worker.rollout(generator, episodes=batch_size)
    generator.accumulate_loss(ret)

    last_info = generator.update_model()
    log_z = generator.compute_log_Z().detach().cpu().reshape(-1)[0].item()
    print(f"Epoch {epoch_id + 1} loss={last_info['loss']:.4f} logZ={log_z:.4f}")
    return last_info


def train(output_path, batch_size=1, epochs_num=10, dataset_path="../dataset/DS1.pickle"):
    batch_size = _ensure_positive_int(batch_size, "batch_size")
    epochs_num = _ensure_positive_int(epochs_num, "epochs_num")
    Ne = 10000
    r_per_bp = 1e-8
    sequences = load_sequences(dataset_path)
    sequence_length = len(sequences[0])
    rho = 4 * Ne * r_per_bp * sequence_length

    env = SimpleARGEnvironment(
        sequence_length=sequence_length,
        rho=rho,
        num_sequences=len(sequences),
        sequences=sequences,
        num_blocks=10,
        fixed_edge_length=0.02,
        rng=random.Random(7),
    )
    generator = TBGFlowNetGenerator(env)
    rollout_worker = RolloutWorker(env)

    history = []
    for epoch in range(epochs_num):
        info = train_epoch(epoch, rollout_worker, generator, batch_size=batch_size)
        log_z = generator.compute_log_Z().detach().cpu().reshape(-1)[0].item()
        if info is not None:
            info = dict(info)
            info['epoch'] = epoch
            info['log_z'] = log_z
            history.append(info)
            print(f"Epoch {epoch + 1} loss={info['loss']:.4f} logZ={log_z:.4f}")

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'training_history.pkl'), 'wb') as handle:
        pickle.dump(history, handle)

    return history


def _ensure_positive_int(value, name):
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _positive_int(value):
    try:
        return _ensure_positive_int(value, "value")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main():
    parser = argparse.ArgumentParser(description="Train the simplified ARG GFlowNet demo.")
    parser.add_argument("--output-path", default=".")
    parser.add_argument("--dataset-path", default="../dataset/DS1.pickle")
    parser.add_argument("--epochs", type=_positive_int, default=10)
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    args = parser.parse_args()

    train(
        args.output_path,
        batch_size=args.batch_size,
        epochs_num=args.epochs,
        dataset_path=args.dataset_path,
    )

if __name__ == "__main__":
    main()
