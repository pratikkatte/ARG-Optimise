"""Fresh simulation-prior exploration for joint SubTB training.

These trajectories are generated for training. No evaluation paths or cached
policy/flow scores enter this module.
"""
import torch
from env.env import SimpleTrajectory


@torch.no_grad()
def sample_prior_trajectories(env, episodes):
    """Draw complete new paths using the existing CwR prior and waiting law."""
    if episodes < 1:
        raise ValueError('Prior exploration requires positive episodes')
    trajectories = []
    for _ in range(episodes):
        state = env.get_initial_state()
        trajectory = SimpleTrajectory()
        while not state.is_done:
            action, log_prior = env._sample_prior_step(state)
            state = env.apply_action(state, action, log_prior=log_prior)
            trajectory.update(action, log_prior=log_prior, log_reward=state.log_reward)
        trajectories.append(trajectory)
    return trajectories


def train_epoch_with_prior(epoch_id, worker, generator, batch_size, prior_episodes,
                           prior_weight, grad_accum_steps=1):
    """Joint gradients for a fixed mixture of fresh policy and fresh prior paths.

Each component is an unbiased minibatch estimate of its own trajectory-averaged
SubTB objective. PF/PB and both flow endpoints are always scored by the current
model. The weight changes the training visitation distribution, not the reward,
forward policy, physical prior, or per-trajectory SubTB definition.
"""
    from train import length_statistics
    if generator.loss_type != 'subtb' or not 0 < prior_weight < 1:
        raise ValueError('Prior exploration requires SubTB and a mixture weight in (0,1)')
    if min(batch_size, prior_episodes, grad_accum_steps) < 1:
        raise ValueError('Both batch sizes and accumulation steps must be positive')
    lengths, prior_lengths, policy_losses, prior_losses = [], [], [], []
    for _ in range(grad_accum_steps):
        outputs, paths = worker.rollout(generator, episodes=batch_size, collect_flows=True)
        policy_losses.append(float(generator.get_loss_from_rollout_outputs(outputs).detach()))
        lengths.extend(map(len, paths))
        generator.accumulate_loss(outputs, factor=grad_accum_steps/(1-prior_weight))
        del outputs
        prior_paths = sample_prior_trajectories(worker.env, prior_episodes)
        outputs, _ = worker.replay(generator, prior_paths, collect_flows=True)
        prior_losses.append(float(generator.get_loss_from_rollout_outputs(outputs).detach()))
        prior_lengths.extend(map(len, prior_paths))
        generator.accumulate_loss(outputs, factor=grad_accum_steps/prior_weight)
        del outputs
    info = generator.update_model()
    info.update(length_statistics(lengths))
    info.update(length_statistics(prior_lengths, 'prior_'))
    info.update(on_policy_subtb_loss=sum(policy_losses)/len(policy_losses),
                prior_subtb_loss=sum(prior_losses)/len(prior_losses),
                prior_training_weight=prior_weight,
                on_policy_training_episodes=batch_size*grad_accum_steps,
                prior_training_episodes=prior_episodes*grad_accum_steps)
    return info
