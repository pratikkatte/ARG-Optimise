import math
import os

import numpy as np
import torch

from models import ARGModel
from env import CoalescenceChoice, RecombinationChoice, SimpleTrajectory
from dataclasses import replace
from breakpoint_model import BreakpointSplitPositionCNN
from time_model import TimeModel

LOSS_FN = {
    'MSE': torch.nn.MSELoss(),
    'HUBER': torch.nn.HuberLoss(delta=1.0),
}

class TBGFlowNetGenerator(torch.nn.Module):
    def __init__(
        self,
        env,
        init_z_sample_count,
        cfg=None,
        device=None,
        verbose=True,
        arg_model_lr=0.001,
        z_lr=0.001,
        grad_clip=10.0,
        model_kwargs=None,
        policy_lr=None,
        log_z_lr=None,
        ddp=False,
        local_rank=0,
        time_layers=3,
        breakpoint_hidden_dim=128,
        breakpoint_gap_hidden_size=256,
        breakpoint_gap_layers=3,
        breakpoint_gap_dropout=0.0,
        breakpoint_use_position_features=True,
        embedding_size=32,
        time_hidden_size=256,
        time_dropout=0.0,
        breakpoint_dropout=0.1,

    ):
        super().__init__()
        print(f"verbose: {verbose}")
        self.env = env
        self.verbose = verbose
        self.ddp = ddp
        self.local_rank = local_rank
        self.device = torch.device(device) if device is not None else torch.device(env.device)
        self.env.device = self.device
        if hasattr(self.env, "seq_arrays"):
            self.env.seq_arrays = torch.nn.Parameter(
                self.env.seq_arrays.detach().to(self.device),
                requires_grad=False,
            )
        if hasattr(self.env, "block_seq_arrays"):
            self.env.block_seq_arrays = torch.nn.Parameter(
                self.env.block_seq_arrays.detach().to(self.device),
                requires_grad=False,
            )
        self.init_z_sample_count = init_z_sample_count

        ## Policy model
        if policy_lr is not None:
            arg_model_lr = policy_lr
        if log_z_lr is not None:
            z_lr = log_z_lr
        self.arg_model_lr = float(arg_model_lr)
        self.z_lr = float(z_lr)
        self.model_kwargs = dict(model_kwargs or {})
        embedding_size = int(self.model_kwargs.get("embedding_size", embedding_size))
        arg_model_kwargs = {
            "embedding_size": embedding_size,
            "hidden_size": int(self.model_kwargs.get("hidden_size", 64)),
            "dropout": float(self.model_kwargs.get("dropout", 0.0)),
            "transformer_depth": int(self.model_kwargs.get("transformer_depth", 6)),
            "transformer_heads": int(self.model_kwargs.get("transformer_heads", 4)),
            "transformer_mlp_ratio": float(self.model_kwargs.get("transformer_mlp_ratio", 2.0)),
            "attention_dropout": float(self.model_kwargs.get("attention_dropout", 0.0)),
        }
        self.arg_model = ARGModel(env, **arg_model_kwargs).to(self.device)
        time_hidden_size = int(self.model_kwargs.get("time_hidden_size", time_hidden_size))
        time_layers = int(self.model_kwargs.get("time_layers", time_layers))
        time_dropout = float(self.model_kwargs.get("time_dropout", time_dropout))
        breakpoint_hidden_dim = int(self.model_kwargs.get("breakpoint_hidden_dim", breakpoint_hidden_dim))
        breakpoint_dropout = float(self.model_kwargs.get("breakpoint_dropout", breakpoint_dropout))
        breakpoint_gap_hidden_size = int(
            self.model_kwargs.get("breakpoint_gap_hidden_size", breakpoint_gap_hidden_size)
        )
        breakpoint_gap_layers = int(self.model_kwargs.get("breakpoint_gap_layers", breakpoint_gap_layers))
        breakpoint_gap_dropout = float(
            self.model_kwargs.get("breakpoint_gap_dropout", breakpoint_gap_dropout)
        )
        breakpoint_use_position_features = bool(
            self.model_kwargs.get(
                "breakpoint_use_position_features",
                breakpoint_use_position_features,
            )
        )
        self.time_model = TimeModel(
            embedding_size * 4,
            time_hidden_size,
            time_dropout,
            env.time_env.bins,
            layers=time_layers,
        ).to(self.device)

        self.breakpoint_model = BreakpointSplitPositionCNN(
            hidden_dim=breakpoint_hidden_dim,
            dropout=breakpoint_dropout,
            action_context_dim=embedding_size * 4,
            gap_hidden_dim=breakpoint_gap_hidden_size,
            gap_layers=breakpoint_gap_layers,
            gap_dropout=breakpoint_gap_dropout,
            use_position_features=breakpoint_use_position_features,
        ).to(self.device)

        ## Z partition

        self.max_reward_seen = float("-inf")
        if init_z_sample_count > 0:
            log_rewards = env.sample_log_rewards(self.init_z_sample_count, verbose=verbose)
            self.max_reward_seen = float(np.max(log_rewards))
            init_Z = self.max_reward_seen
        else:
            self.max_reward_seen = -6300.0
            init_Z = -6300.0
        self._Z = torch.nn.Parameter(  # in log
                torch.ones(256, device=self.device) * init_Z / 256, requires_grad=True
                )
        
        self.arg_model_params = (
            list(self.arg_model.parameters()) +
            list(self.time_model.parameters()) +
            list(self.breakpoint_model.parameters())
        )
        self.policy_params = self.arg_model_params

        params = [{'params': self.arg_model_params, 'lr': self.arg_model_lr}]
        params.append({'params': [self._Z], 'lr': self.z_lr})

        # gradient clipping exclude the Z part
        self.gradient_clipping_params = (
            list(self.arg_model.parameters()) +
            list(self.time_model.parameters()) +
            list(self.breakpoint_model.parameters())
        )
        self.grad_clip = float(grad_clip)

        self.opt = torch.optim.Adam(
            params,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            amsgrad=True,
        )

        self.scheduler = None

        self.loss_fn = LOSS_FN['MSE']

        self.grad_norm = lambda model: math.sqrt(sum(
            [p.grad.norm().item() ** 2 for p in self.gradient_clipping_params if p.grad is not None]))
        self.param_norm = lambda model: math.sqrt(sum([p.norm().item() ** 2 for p in self.gradient_clipping_params]))

        # scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler()

        self.loss = 0

        self.loss = torch.tensor(0.0, device=self.device)
        self.accumulated_batches = 0
        self.log_z_target_sum = 0.0
        self.log_z_target_count = 0
        self.last_log_z_target = float(self.compute_log_Z().detach().cpu().item())


    def _encode_states(self, states, region_contexts=None):
        model = self.arg_model.module if self.ddp else self.arg_model
        return model._encode_states(states, region_contexts=region_contexts)


    def save(self, path, metadata=None):
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "generator_state_dict": self.state_dict(),
                "opt_state_dict": self.opt.state_dict(),
                "metadata": dict(metadata or {}),
            },
            path,
        )

    def load(self, path, load_optimizer=True, map_location=None):
        if map_location is None:
            map_location = self.device
        checkpoint = (
            path
            if isinstance(path, dict)
            else self._torch_load(path, map_location=map_location)
        )
        state_dict = checkpoint.get("generator_state_dict", checkpoint)
        self.load_state_dict(state_dict)
        self.to(self.device)
        self.last_log_z_target = float(self.compute_log_Z().detach().cpu().item())

        if load_optimizer and "opt_state_dict" in checkpoint:
            self.opt.load_state_dict(checkpoint["opt_state_dict"])
            self._move_optimizer_state_to_device()
        return checkpoint.get("metadata", {})

    def _move_optimizer_state_to_device(self):
        for state in self.opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _torch_load(self, path, map_location=None):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    
    def compute_log_Z(self, scale_key=None):
        return self._Z.sum()

    def _pad_log_path_lists(self, log_path_lists, dtype, device):
        vectors = [
            torch.stack(log_paths).to(dtype=dtype, device=device)
            if log_paths
            else torch.empty(0, dtype=dtype, device=device)
            for log_paths in log_path_lists
        ]
        return self._pad_log_path_vectors(vectors, dtype, device)

    def _pad_log_path_vectors(self, vectors, dtype, device):
        max_length = max((vector.numel() for vector in vectors), default=0)
        padded = torch.zeros(len(vectors), max_length, dtype=dtype, device=device)
        for row_idx, vector in enumerate(vectors):
            if vector.numel() > 0:
                padded[row_idx, :vector.numel()] = vector.to(dtype=dtype, device=device)
        return padded

    def _forward_one_step(self, input_dict):

        states = input_dict.get("states")

        random_spec = input_dict.get("random_spec")
        

        event = input_dict.get("event")
        event_probs = [
            float(event[idx]["probability"])
            for idx in range(len(states))
        ]
        log_event_pf = torch.log(
            torch.tensor(event_probs, dtype=torch.float32, device=self.device)
        )

        all_actions = input_dict.get("input_actions")
        region_contexts = input_dict.get("region_contexts")

        lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts = self._encode_states(states, region_contexts=region_contexts)
        # input_dict = self._move_input_to_device(input_dict)
        model = self.arg_model.module if self.ddp else self.arg_model
        ret = model(all_actions, lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts, random_spec)

        log_action_pf, selected_action_indices, choosen_actions, choosen_action_features = ret

        log_p_breakpoints = []
        for idx, chosen_action in enumerate(choosen_actions):
            if isinstance(chosen_action, RecombinationChoice):
                lineage_idx = int(chosen_action.active_lineage_i)
                lineage_feature = lineage_seq_features[idx, lineage_idx]
                breakpoint, log_p_breakpoint = self.breakpoint_model(
                    chosen_action,
                    lineage_feature,
                    int(self.env.sequence_length),
                    int(self.env.num_blocks),
                    action_context=choosen_action_features[idx],
                    random_spec=random_spec,
                )
                choosen_actions[idx] = replace(chosen_action, breakpoint=breakpoint)
                log_p_breakpoints.append(log_p_breakpoint)
            else:
                log_p_breakpoints.append(torch.tensor(0.0, device=self.device))

        log_breakpoint_pf = torch.stack(log_p_breakpoints)

        selected_action_features = torch.stack(choosen_action_features, dim=0)  # shape: [B, F]
        time_logits = self.time_model(selected_action_features)
        time_actions = self.time_model.sample(time_logits, random_spec)

        for batch_idx, action in enumerate(choosen_actions):
            time = int(time_actions[batch_idx].detach().cpu().item())
            choosen_actions[batch_idx] = replace(action, time_action=time)

        log_time_pf = self.time_model.compute_log_time_pf(time_logits, time_actions)

        total_log_pf = log_event_pf + log_action_pf + log_breakpoint_pf + log_time_pf

        probs = torch.exp(total_log_pf)
        
        return total_log_pf, probs, choosen_actions

    def _fixed_action_candidates(self, state, action):
        if action.time_action is None:
            raise ValueError(f"Fixed refinement action is missing time_action: {action}")

        coal_actions, recomb_actions = self.env.enumerate_actions(state)
        event_probs = self.env.compute_event_probabilities(state, (coal_actions, recomb_actions))

        if isinstance(action, CoalescenceChoice):
            candidates = coal_actions
            event_probability = event_probs["coal"]
            for idx, candidate in enumerate(candidates):
                if (
                    candidate.active_lineage_i == action.active_lineage_i
                    and candidate.active_lineage_j == action.active_lineage_j
                ):
                    return candidates, idx, event_probability
        elif isinstance(action, RecombinationChoice):
            if action.breakpoint is None:
                raise ValueError(f"Fixed refinement recombination action is missing breakpoint: {action}")
            candidates = recomb_actions
            event_probability = event_probs["recomb"]
            for idx, candidate in enumerate(candidates):
                if (
                    candidate.active_lineage_i == action.active_lineage_i
                    and candidate.material_count == action.material_count
                    and candidate.span_start == action.span_start
                    and candidate.span_end == action.span_end
                ):
                    return candidates, idx, event_probability
        else:
            raise ValueError(f"Unsupported fixed refinement action: {action}")

        raise ValueError(f"Fixed refinement action could not be matched to model candidates: {action}")

    def _score_given_actions(self, states, actions, window_start=0, window_end=None):
        if len(states) != len(actions):
            raise ValueError(
                f"states and actions must have the same length, got {len(states)} and {len(actions)}"
            )
        if not states:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        if window_end is None:
            window_end = self.env.num_blocks

        all_actions = []
        selected_action_indices = []
        event_probs = []
        for state, action in zip(states, actions):
            candidates, action_idx, event_probability = self._fixed_action_candidates(state, action)
            all_actions.append(candidates)
            selected_action_indices.append(action_idx)
            event_probs.append(float(event_probability))

        log_event_pf = torch.log(
            torch.tensor(event_probs, dtype=torch.float32, device=self.device)
        )
        region_contexts = torch.tensor(
            [
                [window_start / self.env.num_blocks, window_end / self.env.num_blocks]
                for _ in states
            ],
            dtype=torch.float32,
            device=self.device,
        )

        lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts = (
            self._encode_states(states, region_contexts=region_contexts)
        )
        model = self.arg_model.module if self.ddp else self.arg_model
        logits, action_features = model._score_candidates(
            all_actions,
            lineage_reps,
            summary_reps,
        )

        action_indices = torch.tensor(
            selected_action_indices,
            dtype=torch.long,
            device=self.device,
        )
        log_action_pf = model.compute_log_path_pf(logits, action_indices)
        batch_indices = torch.arange(len(states), dtype=torch.long, device=self.device)
        selected_action_features = action_features[batch_indices, action_indices]

        log_p_breakpoints = []
        for idx, action in enumerate(actions):
            if isinstance(action, RecombinationChoice):
                valid_breakpoints = self.breakpoint_model._valid_breakpoints_list(action)
                breakpoint = int(action.breakpoint)
                if breakpoint not in valid_breakpoints:
                    raise ValueError(
                        f"Fixed refinement breakpoint {breakpoint} is not valid for action: {action}"
                    )

                lineage_idx = int(action.active_lineage_i)
                lineage_feature = lineage_seq_features[idx, lineage_idx]
                valid_logits = self.breakpoint_model.valid_breakpoint_logits(
                    action,
                    lineage_feature,
                    int(self.env.sequence_length),
                    int(self.env.num_blocks),
                    action_context=selected_action_features[idx],
                )
                local_idx = valid_breakpoints.index(breakpoint)
                log_p_breakpoints.append(torch.log_softmax(valid_logits, dim=0)[local_idx])
            else:
                log_p_breakpoints.append(logits.new_tensor(0.0))

        log_breakpoint_pf = torch.stack(log_p_breakpoints)
        time_logits = self.time_model(selected_action_features)
        time_actions = torch.tensor(
            [int(action.time_action) for action in actions],
            dtype=torch.long,
            device=self.device,
        )
        log_time_pf = self.time_model.compute_log_time_pf(time_logits, time_actions)

        return log_event_pf + log_action_pf + log_breakpoint_pf + log_time_pf

    def _forward_rollout_batch(
        self,
        episodes,
        base_state,
        log_pfs=None,
        backward_num_parents_by_traj=None,
        random_spec=None,
        return_states=False,
        window_start=0,
        window_end=None,
    ):
        if isinstance(base_state, list):
            if len(base_state) == 0:
                raise ValueError("base_state list must contain at least one state.")
            candidate_indices = torch.randint(len(base_state), (episodes,)).tolist()
            states = [
                base_state[candidate_idx].clone(copy_partials=True)
                for candidate_idx in candidate_indices
            ]
            if log_pfs is not None and backward_num_parents_by_traj is not None:
                log_paths_pf_by_traj = [
                    log_pfs[candidate_idx].copy()
                    for candidate_idx in candidate_indices
                ]
                backward_num_parents_by_traj = [
                    backward_num_parents_by_traj[candidate_idx].copy()
                    for candidate_idx in candidate_indices
                ]
            else:
                log_paths_pf_by_traj = [[] for _ in range(episodes)]
                backward_num_parents_by_traj = [[] for _ in range(episodes)]

            if isinstance(window_start, list) and isinstance(window_end, list):
                batch_window_starts = [window_start[idx] for idx in candidate_indices]
                batch_window_ends = [window_end[idx] for idx in candidate_indices]
            else:
                batch_window_starts = window_start
                batch_window_ends = window_end
        else:
            states = [base_state.clone(copy_partials=True) for _ in range(episodes)]
            if log_pfs is not None and backward_num_parents_by_traj is not None:
                log_paths_pf_by_traj = [log_pfs.copy() for _ in range(episodes)]
                backward_num_parents_by_traj = [
                    backward_num_parents_by_traj.copy()
                    for _ in range(episodes)
                ]
            else:
                log_paths_pf_by_traj = [[] for _ in range(episodes)]
                backward_num_parents_by_traj = [[] for _ in range(episodes)]

            batch_window_starts = window_start
            batch_window_ends = window_end

        trajectories = [SimpleTrajectory() for _ in states]
        unfinished = [idx for idx, state in enumerate(states) if not state.is_done]

        while unfinished:
            active_states = [states[idx] for idx in unfinished]
            if isinstance(batch_window_starts, list):
                active_window_starts = [batch_window_starts[idx] for idx in unfinished]
            else:
                active_window_starts = batch_window_starts

            if isinstance(batch_window_ends, list):
                active_window_ends = [batch_window_ends[idx] for idx in unfinished]
            else:
                active_window_ends = batch_window_ends

            input_dict = self.env.prepare_state_rollout_inputs(
                active_states,
                random_spec=random_spec,
                window_start=active_window_starts,
                window_end=active_window_ends,
            )

            total_log_pf, probs, choosen_actions = self._forward_one_step(input_dict)

            for batch_idx, traj_idx in enumerate(unfinished):
                state = states[traj_idx]
                coal_actions, recomb_actions = self.env.enumerate_actions(state)

                action = choosen_actions[batch_idx]
                log_paths_pf_by_traj[traj_idx].append(total_log_pf[batch_idx])
                log_prior = self.env.compute_cwr_event_log_prior(
                    state,
                    (coal_actions, recomb_actions),
                    action,
                )

                next_state = self.env.apply_action(
                    state,
                    action,
                    log_prior=log_prior,
                )
                states[traj_idx] = next_state
                trajectories[traj_idx].update(
                    action,
                    log_prior=log_prior,
                    log_reward=next_state.log_reward,
                )

                backward_num_parents_by_traj[traj_idx].append(
                    self.env.count_backward_parents(next_state)
                )
            unfinished = [idx for idx, state in enumerate(states) if not state.is_done]

        log_paths_pf = self._pad_log_path_lists(
            log_paths_pf_by_traj,
            torch.float32,
            self.device,
        )

        log_paths_pb = [
            -torch.log(torch.tensor(num_parents, dtype=torch.float32, device=self.device))
            for num_parents in backward_num_parents_by_traj
        ]
        log_paths_pb = self._pad_log_path_vectors(log_paths_pb, torch.float32, self.device)

        log_rewards = torch.tensor(
            [state.log_reward for state in states],
            dtype=torch.float32,
            device=self.device,
        )

        data = {
            "log_paths_pf": log_paths_pf,
            "log_paths_pb": log_paths_pb,
            "log_rewards": log_rewards,
            "log_z": self.compute_log_Z(),
        }
        if return_states:
            data["states"] = states

        return data, trajectories

    def forward(self, input_dict):
        if input_dict.get("mode") == "rollout_batch":
            return self._forward_rollout_batch(
                episodes=input_dict["episodes"],
                base_state=input_dict["base_state"],
                log_pfs=input_dict.get("log_pfs"),
                backward_num_parents_by_traj=input_dict.get("backward_num_parents_by_traj"),
                random_spec=input_dict.get("random_spec"),
                return_states=input_dict.get("return_states", False),
                window_start=input_dict.get("window_start", 0),
                window_end=input_dict.get("window_end"),
            )
        if input_dict.get("mode") == "score_actions":
            return self._score_given_actions(
                states=input_dict["states"],
                actions=input_dict["actions"],
                window_start=input_dict.get("window_start", 0),
                window_end=input_dict.get("window_end"),
            )

        total_log_pf, probs, choosen_actions = self._forward_one_step(input_dict)
        return total_log_pf, probs, choosen_actions, self.compute_log_Z()


    def update_model(self):
        
        info = {'grad_norm': self.grad_norm(self),
                # 'z_grad_norm': self._Z.grad.norm().item(),
                'param_norm': self.param_norm(self),
                'loss': self.loss.detach().cpu().numpy().tolist()}
        
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        self.opt.step()
        self.opt.zero_grad()
        self.loss = torch.tensor(0.0, device=self.device)

        return info

    def get_loss_from_rollout_outputs(self, rollout_outputs):
        """
        Compute the Trajectory Balance loss from rollout outputs.
        """
        log_paths_pf = rollout_outputs['log_paths_pf']
        log_paths_pb = rollout_outputs['log_paths_pb']
        log_rewards = torch.as_tensor(
            rollout_outputs['log_rewards'],
            dtype=log_paths_pf.dtype,
            device=log_paths_pf.device,
        )
        
        log_pf = log_paths_pf.sum(-1)
        log_pb = log_paths_pb.sum(-1)

        log_z = rollout_outputs.get("log_z")
        if log_z is None:
            log_z = self.compute_log_Z(None)
        log_z = log_z.reshape(-1).to(log_paths_pf)

        forward_value = log_z + log_pf
        backward_value = log_rewards + log_pb

        loss = self.loss_fn(forward_value, backward_value)

        return loss
        
    def accumulate_loss(self, rollout_outputs, factor=1.0):
        loss = self.get_loss_from_rollout_outputs(rollout_outputs)
        loss = (loss / factor)
        loss.backward()
        self.loss += loss 
