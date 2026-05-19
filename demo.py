import random
import os
import pickle
from env import SimpleARGEnvironment
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator
from utils import load_sequences


def train_epoch(epoch_id, rollout_worker, generator):
    """
    """
    ret, trajectories = rollout_worker.rollout(generator)
    generator.accumulate_loss(ret[0])

    last_info = generator.update_model()
    log_z = generator.compute_log_Z().detach().cpu().reshape(-1)[0].item()
    print(f"Epoch {epoch_id + 1} loss={last_info['loss']:.4f} logZ={log_z:.4f}")
    return last_info

def train(output_path):
    Ne = 10000
    r_per_bp = 1e-8
    dataset_path = "../dataset/DS1.pickle"
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
    epochs_num = 10
    for epoch in range(epochs_num):
        info = train_epoch(epoch, rollout_worker, generator)
        log_z = generator.compute_log_Z.detach().cpu().reshape(-1)[0].item()
        if info is not None:
            info = dict(info)
            info['epoch'] = epoch
            info['log_z'] = log_z
            history.append(info)
            print(f"Epoch {epoch + 1} loss={info['loss']:.4f} logZ={log_z:.4f}")
    
    with open(os.path.join(output_path, 'training_history.pkl'), 'wb') as handle:
            pickle.dump(history, handle)

    return history

def main():
    history = train(".")

if __name__ == "__main__":
    main()
