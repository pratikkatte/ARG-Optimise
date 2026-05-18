import random

from env import SimpleARGEnvironment
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator

def main():
    
    Ne = 10000
    r_per_bp = 1e-8
    sequence_length = 8

    rho = 4 * Ne * r_per_bp * sequence_length

    env = SimpleARGEnvironment(
        num_sequences=4,
        sequence_length=sequence_length,
        rho=rho,
        fixed_edge_length=0.02,
        rng=random.Random(7),
    )
    generator = TBGFlowNetGenerator(env)

    rollout_worker = RolloutWorker(env, max_steps=50)
    final_state, trajectory = rollout_worker.rollout(generator)

    print("Simplified discrete CwR ARG prototype demo")
    print(
        f"n={env.num_sequences} sequence_length={env.sequence_length} "
        f"num_blocks={env.num_blocks} rho={env.rho}"
    )
    for item in trajectory:
        print(
            "step={step:02d} action={action} log_prior={log_prior:.4f} "
            "active={active_lineage_count} done={is_done}".format(**item)
        )
    print(
        "finished done={} steps={} log_reward={}".format(
            final_state.is_done,
            len(trajectory),
            final_state.log_reward,
        )
    )

if __name__ == "__main__":
    main()
