import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from env.env import SimpleARGEnvironment
from gfn.rollout import RolloutWorker
from generator import GFlowNetGenerator, TBGFlowNetGenerator
from gfn.objectives import resolve_log_loss, validate_objective
from gfn.tb import tb_diagnostics
from env.time_env import DEFAULT_TIME_BINS, DEFAULT_TIME_DELTA_BIN_WIDTH
from utils import load_sequences
from training.schedules import LearningRateConfig, PolicyTemperatureConfig
from training.trainer import Trainer, TrajectoryMixConfig, trajectory_length_statistics


DEFAULT_NE = 10000
DEFAULT_R_PER_BP = 2e-8
DEFAULT_MU_PER_BP = 2e-8
DEFAULT_INIT_Z_SAMPLE_COUNT = 16
DEFAULT_POLICY_LR = 1e-3
DEFAULT_LOG_Z_LR = 1e-3
DEFAULT_GRAD_CLIP = 10.0
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_EVAL_EPISODES = 128
DEFAULT_EVAL_EVERY = 10
DEFAULT_EMBEDDING_SIZE = 32
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_DROPOUT = 0.0
# New training runs use the smaller breakpoint policy. ARGModel keeps its legacy
# constructor defaults so checkpoints without explicit model settings still load.
DEFAULT_BREAKPOINT_HIDDEN_DIM = 32
DEFAULT_BREAKPOINT_DROPOUT = 0.1
DEFAULT_TRANSFORMER_DEPTH = 6
DEFAULT_TRANSFORMER_HEADS = 4
DEFAULT_TRANSFORMER_MLP_RATIO = 2.0
DEFAULT_ATTENTION_DROPOUT = 0.0
DEFAULT_TIME_HIDDEN_SIZE = 256
DEFAULT_TIME_LAYERS = 3
DEFAULT_TIME_DROPOUT = 0.0
DEFAULT_BREAKPOINT_GAP_HIDDEN_SIZE = 64
DEFAULT_BREAKPOINT_GAP_LAYERS = 1
DEFAULT_BREAKPOINT_GAP_DROPOUT = 0.0
DEFAULT_BREAKPOINT_USE_POSITION_FEATURES = True
DEFAULT_BREAKPOINT_POLICY = "cnn"
DEFAULT_BREAKPOINT_MIXTURE_HIDDEN_DIM = 128
DEFAULT_BREAKPOINT_MIXTURE_LAYERS = 4
DEFAULT_BREAKPOINT_MIXTURE_COMPONENTS = 4
MODEL_VERSION = "cwr-event-transformer-block-partials-v3"

from eval.density_fit import fit_stats
from eval.ess import importance_stats, log_importance_weights


def seed_everything(seed):
    """Seed Python, NumPy, and Torch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_generator(rollout_worker, generator, episodes, seed, fixed_trajectories=None,
                       terminal_evaluator=None, terminal_details=None, eval_density_slope=False):
    """Evaluate fresh or fixed trajectories without perturbing training state."""
    episodes = int(episodes)
    if episodes <= 0:
        return {}
    if fixed_trajectories is not None and terminal_evaluator is not None:
        raise ValueError('Terminal sampling quality requires fresh-policy samples')
    log_loss = getattr(generator, 'log_loss', (getattr(generator, 'loss_type', 'tb'),))

    env = rollout_worker.env
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    env_rng_state = env.rng.getstate() if hasattr(env.rng, "getstate") else None
    module_modes = [(module, module.training) for module in generator.modules()]

    try:
        generator.eval()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(env.rng, "seed"):
            env.rng.seed(seed)

        with torch.no_grad():
            collect = 'subtb' in log_loss
            if fixed_trajectories is None:
                outputs, trajectories = rollout_worker.rollout(
                    generator, episodes=episodes, **({'collect_flows': True} if collect else {}),
                    **({'return_states': True} if terminal_evaluator is not None else {}))
            else:
                if len(fixed_trajectories) != episodes:
                    raise ValueError('Fixed evaluation episode count must match the held-out set')
                outputs, trajectories = rollout_worker.replay(generator, fixed_trajectories, collect_flows=collect)
            log_pf = outputs["log_paths_pf"].double().sum(-1)
            log_pb = outputs["log_paths_pb"].double().sum(-1)
            log_rewards = outputs["log_rewards"].to(log_pf)
            log_weights = log_rewards + log_pb - log_pf
            weight_stats = importance_stats(
                log_importance_weights(log_rewards, log_pf, log_pb), fresh=fixed_trajectories is None)
            initial_state = env.get_initial_state()
            initial_prior_probs = env.compute_event_probabilities(initial_state)
            initial_event_probs = (
                generator.compute_event_probabilities(initial_state)
                if hasattr(generator, "compute_event_probabilities") else initial_prior_probs
            )

        lengths = torch.tensor([len(traj) for traj in trajectories], dtype=torch.float32)
        coal_counts = torch.tensor(
            [
                sum(
                    1
                    for action in traj.actions
                    if action.event_type == "coal"
                )
                for traj in trajectories
            ],
            dtype=torch.float32,
        )
        recomb_counts = torch.tensor(
            [
                sum(
                    1
                    for action in traj.actions
                    if action.event_type == "recomb"
                )
                for traj in trajectories
            ],
            dtype=torch.float32,
        )
        extra = trajectory_length_statistics(lengths, "eval_")
        if 'tb' in log_loss:
            with torch.no_grad():
                extra.update({'eval_' + key: value for key, value in
                              tb_diagnostics(log_pf, log_pb, log_rewards, generator.compute_log_Z()).items()})
        if eval_density_slope:
            # Fit log(P_F/P_B) = intercept + slope * log R on the T=1 evaluation
            # histories. The balance identity gives slope 1 at the target density.
            fit = fit_stats(log_rewards.detach().cpu().numpy(),
                            (log_pf-log_pb).detach().cpu().numpy())
            extra['eval_density_slope'] = fit['slope']
        if terminal_evaluator is not None:
            with torch.no_grad():
                terminal_metrics, details = terminal_evaluator(env, outputs, trajectories)
                extra.update(terminal_metrics)
                if terminal_details is not None:
                    terminal_details.update(details)
        if 'subtb' in log_loss:
            with torch.no_grad():
                extra["eval_subtb_loss"] = generator.get_loss_from_rollout_outputs(outputs).item()
                extra.update({"eval_" + key: value for key, value in
                              generator.get_subtb_diagnostics(outputs, extra["eval_subtb_loss"]).items()})
        return {
            **extra,
            "eval_log_weight_mean": weight_stats["log_weight_mean"],
            "eval_log_weight_std": weight_stats["log_weight_std"],
            # ESS describes importance weights only for fresh on-policy samples.
            "eval_importance_ess_fraction": weight_stats['ess_fraction'],
            ('eval_source_log_flow' if getattr(generator, 'neural_source_flow', False) else 'eval_log_z'):
                float(generator.compute_log_Z().detach()),
            "eval_importance_max_weight": weight_stats['max_normalized_weight'],
            "eval_log_pf_mean": float(log_pf.mean().detach().cpu().item()),
            "eval_log_pb_mean": float(log_pb.mean().detach().cpu().item()),
            "eval_log_reward_mean": float(log_rewards.mean().detach().cpu().item()),
            "eval_trajectory_length_mean": float(lengths.mean().item()),
            "eval_coalescence_count_mean": float(coal_counts.mean().item()),
            "eval_recombination_count_mean": float(recomb_counts.mean().item()),
            "eval_initial_coalescence_prob": float(initial_event_probs.get("coal", 0.0)),
            "eval_initial_recombination_prob": float(initial_event_probs.get("recomb", 0.0)),
            "eval_initial_prior_coalescence_prob": float(initial_prior_probs.get("coal", 0.0)),
            "eval_initial_prior_recombination_prob": float(initial_prior_probs.get("recomb", 0.0)),
        }
    finally:
        for module, training in module_modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if env_rng_state is not None:
            env.rng.setstate(env_rng_state)


def save_best_checkpoints(generator, info, metadata, checkpoints_path, best_scores):
    """Keep independent minima; residual mean is best when closest to zero."""
    criteria = (
        ("loss", "best.pt", "best_loss"),
        ("eval_tb_mse", "best_eval_loss.pt", "best_eval_loss"),
        ("eval_subtb_loss", "best_eval_subtb_loss.pt", "best_eval_subtb_loss"),
        ("eval_residual_mean", "best_residual_mean.pt", "best_abs_residual_mean"),
        ("eval_residual_std", "best_residual_std.pt", "best_residual_std"),
    )
    logged = {}
    for metric, filename, best_key in criteria:
        if metric not in info:
            continue
        value = float(info[metric])
        score = abs(value) if metric == "eval_residual_mean" else value
        if not math.isfinite(score):
            continue
        if score < best_scores.get(metric, float("inf")):
            path = os.path.join(checkpoints_path, filename)
            checkpoint_metadata = {
                **metadata,
                **{key: float(value) for key, value in info.items()
                   if value is not None and (key in {"loss", "tb_loss", "subtb_loss"} or key.startswith("eval_"))},
                "checkpoint_metric": metric,
                "checkpoint_metric_value": value,
                "checkpoint_score": score,
            }
            generator.save(path, metadata=checkpoint_metadata)
            best_scores[metric] = score
            path_key = "best_checkpoint_path" if metric in {"loss", "tb_loss"} else f"{best_key}_checkpoint_path"
            logged[path_key] = path
        logged[best_key] = best_scores[metric]
    return logged


def evaluate_generator_repeated(worker, generator, episodes, seed, evaluator,
                                output_dir, step, phase='joint', repeats=1, first_repeat=0,
                                should_stop=None, eval_density_slope=False):
    """Independent MC replicates of one frozen checkpoint; retain full diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for repeat in range(first_repeat, first_repeat+repeats):
        if should_stop is not None and should_stop():
            break
        actual_seed = seed + repeat * 1000003
        details = {}
        metrics = evaluate_generator(worker, generator, episodes, actual_seed,
                                     terminal_evaluator=evaluator, terminal_details=details,
                                     eval_density_slope=eval_density_slope)
        record = dict(step=step, phase=phase, repeat=repeat, seed=actual_seed,
                      episodes=episodes, protocol_sha256=evaluator.protocol['sha256'], **metrics)
        path = output_dir/f'step_{step:06d}_{phase}_rep_{repeat:02d}.json'
        temporary = path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(dict(record=record, details=details), allow_nan=False))
        temporary.replace(path)
        records.append(record)
    return records


def initialize_from_checkpoint(generator, checkpoint_path):
    """Warm-start matching data/model weights while keeping a fresh optimizer."""
    checkpoint = generator._torch_load(checkpoint_path, map_location='cpu')
    metadata = checkpoint.get('metadata', {})
    env = generator.env
    expected = dict(sequences=list(env.sequences), sequence_length=env.sequence_length,
                    num_blocks=env.num_blocks, effective_population_size=env.population_size,
                    mutation_rate=env.mutation_rate, recombination_rate=env.recombination_rate)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Initialization checkpoint does not match {key}')
    for key, value in metadata.get('model', {}).items():
        if key in generator.model_kwargs and generator.model_kwargs[key] != value:
            raise ValueError(f'Initialization checkpoint does not match model setting {key}')
    generator.load(checkpoint, load_optimizer=False)
    return dict(init_checkpoint=str(Path(checkpoint_path).resolve()),
                init_checkpoint_step=int(metadata.get('epoch', -1)) + 1,
                checkpoint_initialization='weights_only')


def train(
    dataset_path,
    output_path,
    device,
    bp_per_blocks=1,
    batch_size=1,
    epochs_num=10,
    seed=7,
    init_z_sample_count=DEFAULT_INIT_Z_SAMPLE_COUNT,
    use_wandb=True,
    effective_population_size=DEFAULT_NE,
    mutation_rate=DEFAULT_MU_PER_BP,
    recombination_rate=DEFAULT_R_PER_BP,
    policy_lr=DEFAULT_POLICY_LR,
    log_z_lr=DEFAULT_LOG_Z_LR,
    grad_clip=DEFAULT_GRAD_CLIP,
    grad_accum_steps=DEFAULT_GRAD_ACCUM_STEPS,
    eval_episodes=DEFAULT_EVAL_EPISODES,
    eval_every=DEFAULT_EVAL_EVERY,
    time_bins=DEFAULT_TIME_BINS,
    time_delta_bin_width=DEFAULT_TIME_DELTA_BIN_WIDTH,
    embedding_size=DEFAULT_EMBEDDING_SIZE,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    dropout=DEFAULT_DROPOUT,
    breakpoint_hidden_dim=DEFAULT_BREAKPOINT_HIDDEN_DIM,
    breakpoint_dropout=DEFAULT_BREAKPOINT_DROPOUT,
    transformer_depth=DEFAULT_TRANSFORMER_DEPTH,
    transformer_heads=DEFAULT_TRANSFORMER_HEADS,
    transformer_mlp_ratio=DEFAULT_TRANSFORMER_MLP_RATIO,
    attention_dropout=DEFAULT_ATTENTION_DROPOUT,
    verbose=True,
    breakpoint_gap_hidden_size=DEFAULT_BREAKPOINT_GAP_HIDDEN_SIZE,
    breakpoint_gap_layers=DEFAULT_BREAKPOINT_GAP_LAYERS,
    breakpoint_policy=DEFAULT_BREAKPOINT_POLICY,
    breakpoint_mixture_hidden_dim=DEFAULT_BREAKPOINT_MIXTURE_HIDDEN_DIM,
    breakpoint_mixture_layers=DEFAULT_BREAKPOINT_MIXTURE_LAYERS,
    breakpoint_mixture_components=DEFAULT_BREAKPOINT_MIXTURE_COMPONENTS,
    loss_type="tb",
    subtb_lambda=0.9,
    flow_lr=None,
    flow_warmup_steps=0,
    flow_warmup_episodes=32,
    flow_head_version=4,
    event_policy="cwr",
    time_policy="cwr_exponential",
    continuous_time_head='exponential',
    terminal_eval=False,
    terminal_eval_grid_size=100,
    terminal_eval_repeats=3,
    terminal_eval_repeat_every=250,
    tmrca_method='grid',
    arg_prior='hudson',
    exploration_fraction=0.,
    replay_fraction=0.,
    replay_capacity=2048,
    replay_grid_size=16,
    replay_per_topology=4,
    replay_min_size=128,
    lr_schedule='constant',
    lr_schedule_steps=0,
    lr_warmup_steps=0,
    lr_warmup_start_factor=0.1,
    lr_min_factor=0.1,
    policy_temperature_schedule='constant',
    policy_temperature_start=1.0,
    policy_temperature_anneal_steps=0,
    init_checkpoint=None,
    checkpoint_every=0,
    eval_density_slope=False,
    log_loss=None,
):
    """Neural training is unavailable until the infinite-sites encoder migration."""
    from env.workflow import require_neural_migration
    require_neural_migration()
    from dataclasses import asdict
    mix_config = TrajectoryMixConfig(exploration_fraction, replay_fraction, replay_capacity,
                                     replay_grid_size, replay_per_topology, replay_min_size)
    mix_config.validate()
    lr_config = LearningRateConfig(lr_schedule, lr_schedule_steps or epochs_num,
                                   lr_warmup_steps, lr_warmup_start_factor, lr_min_factor)
    lr_config.validate()
    temperature_config = PolicyTemperatureConfig(policy_temperature_schedule, policy_temperature_start,
                                                 policy_temperature_anneal_steps)
    temperature_config.validate_training(loss_type, event_policy, exploration_fraction, replay_fraction,
                                         flow_warmup_steps)
    flow_lr = policy_lr if flow_lr is None else flow_lr
    validate_objective(loss_type, subtb_lambda, flow_lr)
    log_loss = resolve_log_loss(log_loss, loss_type)
    if checkpoint_every < 0:
        raise ValueError('checkpoint_every must be nonnegative; 0 disables periodic saves')
    if eval_density_slope and eval_episodes < 2:
        raise ValueError('Density slope evaluation needs at least two eval_episodes')
    if flow_warmup_steps < 0 or flow_warmup_episodes < 1:
        raise ValueError("Flow warm-up needs nonnegative steps and positive episodes")
    if time_policy == "cwr_exponential" and os.path.exists(output_path) and os.listdir(output_path):
        raise ValueError("Continuous training requires a separate empty output directory")
    seed_everything(seed)
    device = torch.device(device)

    sequences = load_sequences(dataset_path)
    sequence_length = len(sequences[0])

    env = SimpleARGEnvironment(
        sequence_length=sequence_length,
        num_sequences=len(sequences),
        bp_per_blocks = bp_per_blocks,
        sequences=sequences,
        device=device,
        recombination_rate=recombination_rate,
        population_size=effective_population_size,
        mutation_rate=mutation_rate,
        time_bins=time_bins,
        time_delta_bin_width=time_delta_bin_width,
        time_policy=time_policy,
        arg_prior=arg_prior,
    )
    model_kwargs = {
        "time_policy": time_policy,
        "continuous_time_head": continuous_time_head,
        "event_policy": event_policy,
        "embedding_size": int(embedding_size),
        "hidden_size": int(hidden_size),
        "dropout": float(dropout),
        "breakpoint_hidden_dim": int(breakpoint_hidden_dim),
        "breakpoint_policy": breakpoint_policy,
        "breakpoint_mixture_hidden_dim": int(breakpoint_mixture_hidden_dim),
        "breakpoint_mixture_layers": int(breakpoint_mixture_layers),
        "breakpoint_mixture_components": int(breakpoint_mixture_components),
        "breakpoint_dropout": float(breakpoint_dropout),
        "transformer_depth": int(transformer_depth),
        "transformer_heads": int(transformer_heads),
        "transformer_mlp_ratio": float(transformer_mlp_ratio),
        "attention_dropout": float(attention_dropout),
        "time_hidden_size": int(DEFAULT_TIME_HIDDEN_SIZE),
        "time_layers": int(DEFAULT_TIME_LAYERS),
        "time_dropout": float(DEFAULT_TIME_DROPOUT),
        "breakpoint_gap_hidden_size": int(breakpoint_gap_hidden_size),
        "breakpoint_gap_layers": int(breakpoint_gap_layers),
        "breakpoint_gap_dropout": float(DEFAULT_BREAKPOINT_GAP_DROPOUT),
        "breakpoint_use_position_features": bool(DEFAULT_BREAKPOINT_USE_POSITION_FEATURES),
    }

    generator = GFlowNetGenerator(
        env,
        init_z_sample_count=init_z_sample_count,
        device=device,
        verbose=verbose,
        policy_lr=policy_lr,
        log_z_lr=log_z_lr,
        grad_clip=grad_clip,
        model_kwargs=model_kwargs,
        loss_type=loss_type, subtb_lambda=subtb_lambda, flow_lr=flow_lr,
        log_loss=log_loss,
        flow_head_version=flow_head_version,
        initialize_z_from_policy=init_checkpoint is None,
    )
    initialization_metadata = {}
    if init_checkpoint is not None:
        initialization_metadata = initialize_from_checkpoint(generator, init_checkpoint)
        print(f"Initialized weights from {init_checkpoint}; optimizer and schedules start at update 0")
    print(f"Generator device: {generator.device}")
    rollout_worker = RolloutWorker(env)
    terminal_evaluator = None
    if terminal_eval:
        from eval.posterior_summary import TerminalSamplingEvaluator
        terminal_evaluator = TerminalSamplingEvaluator.from_dataset(
            dataset_path, env, terminal_eval_grid_size, tmrca_method=tmrca_method)
        evaluate_generator_repeated(rollout_worker, generator, eval_episodes, seed+100000,
                                    terminal_evaluator, Path(output_path)/'terminal_quality', 0,
                                    phase='initial', repeats=terminal_eval_repeats,
                                    eval_density_slope=eval_density_slope)
    generator.configure_lr_schedule(lr_config)
    from gfn.flow_training import warmup_flow
    warmup_metrics = warmup_flow(generator, flow_warmup_steps, flow_warmup_episodes, seed + 100019)
    if warmup_metrics:
        print(f"Flow warm-up: {warmup_metrics}")

    rollout_worker = RolloutWorker(env)
    print(f"Training on device: {generator.device}")
    trainer = Trainer(generator, rollout_worker, mix_config, temperature_config, seed)

    os.makedirs(output_path, exist_ok=True)
    checkpoints_path = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    history = []
    best_scores = {}
    wandb_run = None
    
    print(f"use_wandb: {use_wandb}")
    if use_wandb:
        wandb_run = wandb.init()
        wandb.config.update({
            "device": str(generator.device),
            **env.time_metadata,
            "effective_population_size": float(effective_population_size),
            "mutation_rate": float(mutation_rate),
            "recombination_rate": float(recombination_rate),
            "loss_type": loss_type, "log_loss": list(log_loss), "subtb_lambda": subtb_lambda, "flow_lr": flow_lr,
            "flow_head_version": generator.flow_head_version if loss_type == "subtb" else None,
            **warmup_metrics,
            **initialization_metadata,
            "policy_lr": float(policy_lr),
            **({'source_flow_parameterization': 'shared_flow_network', 'flow_initialization': 'fixed_initial_policy_residual_mean'}
               if generator.neural_source_flow else
               {'log_z_lr': float(log_z_lr), 'log_z_initialization': 'policy_tb_mean'}),
            "init_z_sample_count": int(init_z_sample_count),
            ('initial_source_log_flow' if generator.neural_source_flow else 'initial_log_z'):
                float(generator.compute_log_Z().detach().cpu().item()),
            "grad_clip": float(grad_clip),
            "grad_accum_steps": int(grad_accum_steps),
            "eval_episodes": int(eval_episodes),
            "eval_every": int(eval_every),
            "checkpoint_every": int(checkpoint_every),
            "eval_density_slope": bool(eval_density_slope),
            "bp_per_blocks": int(bp_per_blocks),
            **model_kwargs,
            "model_version": MODEL_VERSION, 'arg_prior': arg_prior, 'tmrca_method': tmrca_method,
            **asdict(mix_config),
            'lr_schedule': asdict(lr_config),
        })

    try:
        for epoch in range(epochs_num):
            if generator.device.type == 'cuda':
                torch.cuda.synchronize(generator.device)
            epoch_start = time.perf_counter()
            info = trainer.train_epoch(epoch + 1, batch_size, grad_accum_steps)
            if generator.device.type == 'cuda':
                torch.cuda.synchronize(generator.device)
            train_seconds = time.perf_counter() - epoch_start
            log_z = generator.compute_log_Z().detach().cpu().reshape(-1)[0].item()
            if info is None:
                continue

            info = dict(info)
            info["epoch"] = epoch
            info['train_seconds'] = train_seconds
            info['source_log_flow' if generator.neural_source_flow else 'log_z'] = log_z
            should_eval = int(eval_episodes) > 0 and (
                (epoch == 0 and not terminal_eval)
                or int(eval_every) <= 1
                or (epoch + 1) % int(eval_every) == 0
            )

            if should_eval:
                if terminal_evaluator is not None:
                    records = evaluate_generator_repeated(
                        rollout_worker, generator, eval_episodes, seed+100000+epoch+1,
                        terminal_evaluator, Path(output_path)/'terminal_quality', epoch+1,
                        repeats=terminal_eval_repeats if (epoch+1)%terminal_eval_repeat_every==0 else 1,
                        eval_density_slope=eval_density_slope)
                    info.update({k:v for k,v in records[0].items() if k.startswith('eval_')})
                else:
                    info.update(evaluate_generator(rollout_worker, generator, eval_episodes, seed+100000+epoch,
                                                   eval_density_slope=eval_density_slope))
            loss = float(info["loss"])

            metadata = build_checkpoint_metadata(
                epoch=epoch,
                best_loss=min(best_scores.get("loss", float("inf")), loss),
                log_z=log_z,
                sequences=sequences,
                sequence_length=sequence_length,
                bp_per_blocks=bp_per_blocks,
                time_metadata=env.time_metadata,
                rho=env.rho,
                effective_population_size=effective_population_size,
                mutation_rate=mutation_rate,
                recombination_rate=recombination_rate,
                policy_lr=policy_lr,
                log_z_lr=log_z_lr,
                grad_clip=grad_clip,
                grad_accum_steps=grad_accum_steps,
                eval_episodes=eval_episodes,
                eval_every=eval_every,
                model_kwargs=model_kwargs,
                seed=seed,
                init_z_sample_count=init_z_sample_count,
                model_version=MODEL_VERSION,
            )
            metadata.update(tmrca_method=tmrca_method, checkpoint_every=int(checkpoint_every),
                            eval_density_slope_enabled=bool(eval_density_slope))
            metadata.update(initialization_metadata)
            if temperature_config.schedule != 'constant':
                metadata['policy_temperature_state'] = temperature_config.state_dict(epoch+1)
            if mix_config.enabled:
                metadata['replay_training_state'] = trainer.state_dict()
            if terminal_evaluator is not None:
                metadata['terminal_protocol_sha256'] = terminal_evaluator.protocol['sha256']
            info.update(save_best_checkpoints(
                generator, info, {**metadata, **warmup_metrics}, checkpoints_path, best_scores,
            ))
            if checkpoint_every and ((epoch+1) % checkpoint_every == 0 or epoch+1 == epochs_num):
                path = os.path.join(checkpoints_path, f'checkpoint_{epoch+1:04d}.pt')
                generator.save(path, metadata={**metadata, **warmup_metrics,
                    **{key: value for key, value in info.items() if key.startswith('eval_')}})
                info['checkpoint_path'] = path
            if generator.device.type == 'cuda':
                torch.cuda.synchronize(generator.device)
            # Includes evaluation and checkpoints; excludes history/log output below.
            info['epoch_seconds'] = time.perf_counter() - epoch_start
            history.append(info)
            if 'checkpoint_path' in info:
                with open(os.path.join(output_path, 'training_history.pkl'), 'wb') as handle:
                    pickle.dump(history, handle)

            if wandb_run is not None:
                wandb.log(info, step=epoch + 1)

            eval_text = ''.join(f" eval_{name}_loss={info['eval_' + name + '_loss']:.4f}"
                                for name in log_loss if 'eval_' + name + '_loss' in info)
            print(f"Epoch {epoch + 1} loss={loss:.4f} source_log_flow={log_z:.4f}"
                  f" time={info['epoch_seconds']:.2f}s train={train_seconds:.2f}s{eval_text}")

        with open(os.path.join(output_path, "training_history.pkl"), "wb") as handle:
            pickle.dump(history, handle)
    finally:
        if wandb_run is not None:
            wandb.finish()
    return history


def build_checkpoint_metadata(
    epoch,
    best_loss,
    log_z,
    sequences,
    sequence_length,
    bp_per_blocks,
    time_metadata,
    rho,
    effective_population_size,
    mutation_rate,
    recombination_rate,
    policy_lr,
    log_z_lr,
    grad_clip,
    grad_accum_steps,
    eval_episodes,
    eval_every,
    model_kwargs,
    seed,
    init_z_sample_count,
    model_version,
):
    """Build stable model, environment, optimizer, and run metadata."""
    return {
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "log_z": float(log_z),
        "sequences": list(sequences),
        "num_sequences": len(sequences),
        "sequence_length": int(sequence_length),
        "num_blocks": int(sequence_length // bp_per_blocks),
        "bp_per_blocks": int(bp_per_blocks),
        "rho": float(rho),
        "time": dict(time_metadata),
        **dict(time_metadata),
        "effective_population_size": float(effective_population_size),
        "mutation_rate": float(mutation_rate),
        "recombination_rate": float(recombination_rate),
        "policy_lr": float(policy_lr),
        "log_z_lr": float(log_z_lr),
        "grad_clip": float(grad_clip),
        "grad_accum_steps": int(grad_accum_steps),
        "eval_episodes": int(eval_episodes),
        "eval_every": int(eval_every),
        "model": dict(model_kwargs),
        "seed": int(seed),
        "init_z_sample_count": int(init_z_sample_count),
        "log_z_initialization": "policy_tb_mean",
        "model_version": str(model_version),
    }


def parse_train_args(argv=None):
    """Parse CLI and YAML configuration values and validate their combination."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Train the simplified ARG GFlowNet demo.")
    parser.add_argument("--config", help="YAML settings file; command-line options override its values")
    parser.add_argument('--init-checkpoint',
                        help='Initialize matching model/flow weights; reset optimizer, schedules and epoch count')
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dataset-path",required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--bp-per-blocks",
        type=int,
        default=1,
        help="Number of bp per block",
    )
    parser.add_argument(
        "--init-z-sample-count", type=int, default=DEFAULT_INIT_Z_SAMPLE_COUNT,
        help="Initial-policy trajectories used to center the trajectory-balance residual",
    )
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--effective-population-size", type=float, default=DEFAULT_NE)
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MU_PER_BP)
    parser.add_argument("--recombination-rate", type=float, default=DEFAULT_R_PER_BP)
    parser.add_argument("--loss-type", choices=("tb", "subtb"), default="tb")
    parser.add_argument('--log-loss', nargs='+', choices=('tb', 'subtb'), default=None,
                        help='Losses to report; defaults to loss-type. SubTB can optionally also report TB.')
    parser.add_argument("--event-policy", choices=("cwr", "cwr_residual"), default="cwr")
    parser.add_argument("--subtb-lambda", type=float, default=0.9)
    parser.add_argument("--flow-lr", type=float, default=None)
    parser.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='constant')
    parser.add_argument('--lr-schedule-steps', type=int, default=0,
                        help='Joint optimizer updates in the LR schedule; 0 uses epochs')
    parser.add_argument('--lr-warmup-steps', type=int, default=0)
    parser.add_argument('--lr-warmup-start-factor', type=float, default=0.1)
    parser.add_argument('--lr-min-factor', type=float, default=0.1,
                        help='Final LR as a fraction of each parameter group base LR')
    parser.add_argument('--policy-temperature-schedule', choices=('constant', 'linear'), default='constant')
    parser.add_argument('--policy-temperature-start', type=float, default=1.0)
    parser.add_argument('--policy-temperature-anneal-steps', type=int, default=0)
    parser.add_argument("--flow-warmup-steps", type=int, default=0)
    parser.add_argument("--flow-warmup-episodes", type=int, default=32)
    parser.add_argument("--flow-head-version", type=int, choices=(1, 2, 3, 4, 5), default=4)
    parser.add_argument('--arg-prior', choices=('hudson',), default='hudson')
    parser.add_argument('--exploration-fraction', type=float, default=0.,
                        help='Fraction of the training trajectory budget drawn freshly from the prior')
    parser.add_argument('--replay-fraction', type=float, default=0.,
                        help='Fraction of the training budget rescored from training-only replay')
    parser.add_argument('--replay-capacity', type=int, default=2048)
    parser.add_argument('--replay-grid-size', type=int, default=16)
    parser.add_argument('--replay-per-topology', type=int, default=4)
    parser.add_argument('--replay-min-size', type=int, default=128)
    parser.add_argument("--policy-lr", type=float, default=DEFAULT_POLICY_LR)
    parser.add_argument("--log-z-lr", type=float, default=DEFAULT_LOG_Z_LR)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM_STEPS,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help='Save numbered checkpoints every N updates and at completion; 0 disables')
    parser.add_argument('--eval-density-slope', action=argparse.BooleanOptionalAction, default=False,
                        help='Log the evaluation slope of log(P_F/P_B) against log reward (target: 1)')
    parser.add_argument('--terminal-eval', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--terminal-eval-grid-size', type=int, default=100)
    parser.add_argument('--terminal-eval-repeats', type=int, default=3)
    parser.add_argument('--terminal-eval-repeat-every', type=int, default=250)
    parser.add_argument('--tmrca-method', choices=('grid', 'point_accuracy'), default='grid')
    parser.add_argument("--time-bins", type=int, default=DEFAULT_TIME_BINS)
    parser.add_argument("--time-policy", choices=("cwr_exponential",), default="cwr_exponential")
    parser.add_argument('--continuous-time-head', choices=('exponential', 'gamma'), default='exponential',
                        help='Learned wait distribution; the continuous CwR prior stays exponential')
    parser.add_argument("--time-delta-bin-width", type=float, default=DEFAULT_TIME_DELTA_BIN_WIDTH)
    parser.add_argument("--embedding-size", type=int, default=DEFAULT_EMBEDDING_SIZE)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--breakpoint-hidden-dim", type=int, default=DEFAULT_BREAKPOINT_HIDDEN_DIM)
    parser.add_argument("--breakpoint-policy", choices=("cnn", "sparse_mixture"), default=DEFAULT_BREAKPOINT_POLICY)
    parser.add_argument("--breakpoint-mixture-hidden-dim", type=int, default=DEFAULT_BREAKPOINT_MIXTURE_HIDDEN_DIM)
    parser.add_argument("--breakpoint-mixture-layers", type=int, default=DEFAULT_BREAKPOINT_MIXTURE_LAYERS)
    parser.add_argument("--breakpoint-mixture-components", type=int, default=DEFAULT_BREAKPOINT_MIXTURE_COMPONENTS)
    parser.add_argument("--breakpoint-dropout", type=float, default=DEFAULT_BREAKPOINT_DROPOUT)
    parser.add_argument(
        "--breakpoint-gap-hidden-size", type=int, default=DEFAULT_BREAKPOINT_GAP_HIDDEN_SIZE,
        help="Width of the breakpoint scoring MLP's hidden layers",
    )
    parser.add_argument(
        "--breakpoint-gap-layers", type=int, default=DEFAULT_BREAKPOINT_GAP_LAYERS,
        help="Number of hidden layers in the breakpoint scoring MLP (0 for a linear head)",
    )
    parser.add_argument("--transformer-depth", type=int, default=DEFAULT_TRANSFORMER_DEPTH)
    parser.add_argument("--transformer-heads", type=int, default=DEFAULT_TRANSFORMER_HEADS)
    parser.add_argument("--transformer-mlp-ratio", type=float, default=DEFAULT_TRANSFORMER_MLP_RATIO)
    parser.add_argument("--attention-dropout", type=float, default=DEFAULT_ATTENTION_DROPOUT)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_path = config_parser.parse_known_args(argv)[0].config
    config_args = []
    if config_path:
        import yaml

        try:
            with open(config_path) as handle:
                settings = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as exc:
            parser.error(f"Cannot read config {config_path}: {exc}")
        if not isinstance(settings, dict):
            parser.error("Config must contain a mapping of argument names to values")
        actions = {action.dest: action for action in parser._actions
                   if action.dest not in {"help", "config"}}
        for key, value in settings.items():
            if key == 'evaluation':
                if not isinstance(value, dict):
                    parser.error('Config setting evaluation must be a mapping')
                continue
            if key not in actions:
                parser.error(f"Unknown config setting: {key}")
            action = actions[key]
            if key == 'log_loss':
                if isinstance(value, str):
                    value = [value]
                if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
                    parser.error('Config setting log_loss must be a nonempty list of loss names')
                config_args.extend([action.option_strings[0], *value])
            elif isinstance(action, argparse.BooleanOptionalAction):
                if not isinstance(value, bool):
                    parser.error(f"Config setting {key} must be true or false")
                config_args.append(action.option_strings[0 if value else 1])
            else:
                if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                    parser.error(f"Config setting {key} must be a scalar value")
                config_args.append(f"{action.option_strings[0]}={value}")
    args = parser.parse_args(config_args + argv)
    if args.flow_lr is None:
        args.flow_lr = args.policy_lr
    try:
        validate_objective(args.loss_type, args.subtb_lambda, args.flow_lr)
        args.log_loss = list(resolve_log_loss(args.log_loss, args.loss_type))
        if args.lr_schedule_steps < 0:
            raise ValueError('lr_schedule_steps must be nonnegative')
        LearningRateConfig.from_namespace(args).validate()
        PolicyTemperatureConfig.from_namespace(args).validate_training(
            args.loss_type, args.event_policy, args.exploration_fraction, args.replay_fraction,
            args.flow_warmup_steps)
        mix_config = TrajectoryMixConfig.from_namespace(args)
        mix_config.validate()
        if mix_config.enabled and (args.loss_type != 'subtb' or args.event_policy != 'cwr_residual'):
            raise ValueError('Exploration/replay requires SubTB with cwr_residual policy')
        from policy.time_model import validate_continuous_time_head
        validate_continuous_time_head(args.continuous_time_head, args.time_policy)
        if args.flow_warmup_steps < 0 or args.flow_warmup_episodes < 1:
            raise ValueError("Flow warm-up needs nonnegative steps and positive episodes")
        if min(args.terminal_eval_grid_size, args.terminal_eval_repeats, args.terminal_eval_repeat_every) < 1:
            raise ValueError('Terminal evaluation grid size and repeat settings must be positive')
        if args.terminal_eval and args.eval_episodes < 1:
            raise ValueError('Terminal evaluation needs positive eval_episodes')
        if args.checkpoint_every < 0:
            raise ValueError('checkpoint_every must be nonnegative; 0 disables periodic saves')
        if args.eval_density_slope and args.eval_episodes < 2:
            raise ValueError('Density slope evaluation needs at least two eval_episodes')
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    """Run training from command-line arguments."""
    args = parse_train_args()

    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
                
    print(f"Selected devicesss: {selected_device}")

    train(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        epochs_num=args.epochs,
        bp_per_blocks=args.bp_per_blocks,
        init_z_sample_count=args.init_z_sample_count,
        init_checkpoint=args.init_checkpoint,
        verbose=args.verbose,
        seed=args.seed,
        device=selected_device,
        use_wandb=args.wandb,
        effective_population_size=args.effective_population_size,
        mutation_rate=args.mutation_rate,
        recombination_rate=args.recombination_rate,
        policy_lr=args.policy_lr,
        event_policy=args.event_policy,
        loss_type=args.loss_type, subtb_lambda=args.subtb_lambda, flow_lr=args.flow_lr,
        log_loss=args.log_loss,
        flow_warmup_steps=args.flow_warmup_steps, flow_warmup_episodes=args.flow_warmup_episodes,
        lr_schedule=args.lr_schedule, lr_schedule_steps=args.lr_schedule_steps,
        lr_warmup_steps=args.lr_warmup_steps, lr_warmup_start_factor=args.lr_warmup_start_factor,
        lr_min_factor=args.lr_min_factor,
        policy_temperature_schedule=args.policy_temperature_schedule,
        policy_temperature_start=args.policy_temperature_start,
        policy_temperature_anneal_steps=args.policy_temperature_anneal_steps,
        flow_head_version=args.flow_head_version,
        arg_prior=args.arg_prior,
        exploration_fraction=args.exploration_fraction, replay_fraction=args.replay_fraction,
        replay_capacity=args.replay_capacity, replay_grid_size=args.replay_grid_size,
        replay_per_topology=args.replay_per_topology, replay_min_size=args.replay_min_size,
        log_z_lr=args.log_z_lr,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        eval_episodes=args.eval_episodes,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        eval_density_slope=args.eval_density_slope,
        terminal_eval=args.terminal_eval, terminal_eval_grid_size=args.terminal_eval_grid_size,
        terminal_eval_repeats=args.terminal_eval_repeats, terminal_eval_repeat_every=args.terminal_eval_repeat_every,
        tmrca_method=args.tmrca_method,
        time_bins=args.time_bins,
        time_policy=args.time_policy,
        continuous_time_head=args.continuous_time_head,
        time_delta_bin_width=args.time_delta_bin_width,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        breakpoint_hidden_dim=args.breakpoint_hidden_dim,
        breakpoint_policy=args.breakpoint_policy,
        breakpoint_mixture_hidden_dim=args.breakpoint_mixture_hidden_dim,
        breakpoint_mixture_layers=args.breakpoint_mixture_layers,
        breakpoint_mixture_components=args.breakpoint_mixture_components,
        breakpoint_dropout=args.breakpoint_dropout,
        breakpoint_gap_hidden_size=args.breakpoint_gap_hidden_size,
        breakpoint_gap_layers=args.breakpoint_gap_layers,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        transformer_mlp_ratio=args.transformer_mlp_ratio,
        attention_dropout=args.attention_dropout,
    )


if __name__ == "__main__":
    main()
