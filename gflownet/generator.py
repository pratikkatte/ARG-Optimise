from dataclasses import replace
import math

import numpy as np
import torch

from arg_environment import RecombinationChoice
from model import ARGModel

from .backward import BackwardPolicyMixin
from .checkpoint import CheckpointMixin


class TBGFlowNetGenerator(CheckpointMixin, BackwardPolicyMixin, torch.nn.Module):
    def __init__(
        self,
        env,
        init_z_sample_count,
        device=None,
        verbose=True,
        policy_lr=0.001,
        log_z_lr=0.001,
        grad_clip=10.0,
        model_kwargs=None,
        initialize_z_from_prior=True,
    ):
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.device = torch.device(device if device is not None else env.device)
        self.env.device = self.device
        self.env.seq_arrays = torch.nn.Parameter(
            self.env.seq_arrays.detach().to(self.device), requires_grad=False,
        )
        self.env.block_seq_arrays = torch.nn.Parameter(
            self.env.block_seq_arrays.detach().to(self.device), requires_grad=False,
        )
        self.arg_model = ARGModel(env, **dict(model_kwargs or {})).to(self.device)
        self.time_model = self.arg_model.time_scorer
        self.breakpoint_model = self.arg_model.breakpoint_scorer
        initial_log_z = self._initial_log_z(
            init_z_sample_count, initialize_z_from_prior,
        )
        self._Z = torch.nn.Parameter(
            torch.tensor(float(initial_log_z), device=self.device)
        )

        self.gradient_clipping_params = list(self.arg_model.parameters())
        self.gradient_groups = self._build_gradient_groups()
        self.grad_clip = float(grad_clip)
        self.opt = torch.optim.Adam(
            [
                {"params": self.gradient_clipping_params, "lr": float(policy_lr)},
                {"params": [self._Z], "lr": float(log_z_lr)},
            ],
            weight_decay=0.0,
            betas=(0.9, 0.999),
            amsgrad=True,
        )
        self.loss_fn = torch.nn.MSELoss()
        self.loss = torch.tensor(0.0, device=self.device)

    def _initial_log_z(self, sample_count, initialize_from_prior):
        if not initialize_from_prior:
            return 0.0
        rewards = self.env.sample_log_rewards(sample_count, verbose=self.verbose)
        return float(np.max(rewards))

    def _build_gradient_groups(self):
        groups = {
            "encoder": [
                self.arg_model.summary_token,
                *self.arg_model.seq_embedding.parameters(),
                *self.arg_model.encoder.parameters(),
                *self.arg_model.preview_pair_embedding.parameters(),
            ],
            "event": list(self.arg_model.event_scorer.parameters()),
            "action": list(self.arg_model.action_scorer.parameters()),
            "breakpoint": list(self.arg_model.breakpoint_scorer.parameters()),
            "time": list(self.arg_model.time_scorer.parameters()),
        }
        assigned = {id(parameter) for values in groups.values() for parameter in values}
        groups["other"] = [
            parameter for parameter in self.gradient_clipping_params
            if id(parameter) not in assigned
        ]
        return groups

    def _encode_states(self, states):
        return self.arg_model._encode_states(states)

    def _candidate_action_groups(self, states, candidate_actions=None):
        if candidate_actions is None:
            candidate_actions = [self.env.enumerate_actions(state) for state in states]
        if len(candidate_actions) != len(states):
            raise ValueError("candidate actions must contain one entry per state")
        return candidate_actions

    def _event_policy_distributions(
        self,
        states,
        candidate_actions,
        summary_reps,
        random_spec=None,
    ):
        valid_mask = []
        cwr_probs = []
        for state, (coal_actions, recomb_actions) in zip(states, candidate_actions):
            probabilities = self.env.compute_event_probabilities(
                state,
                (coal_actions, recomb_actions),
            )
            valid_mask.append([
                bool(coal_actions) and probabilities["coal"] > 0.0,
                bool(recomb_actions) and probabilities["recomb"] > 0.0,
            ])
            cwr_probs.append([
                probabilities[event_type] for event_type in self.env.event_types
            ])

        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        cwr_probs = torch.tensor(cwr_probs, dtype=summary_reps.dtype, device=self.device)
        learned_probs, mixed_probs = self.arg_model.compute_event_probabilities(
            summary_reps,
            valid_mask,
            cwr_probs,
            random_spec=random_spec,
        )
        return learned_probs, cwr_probs, mixed_probs

    def compute_event_probabilities(self, states, random_spec=None):
        """Expose learned, CwR, and mixed event probabilities for diagnostics."""
        candidate_actions = self._candidate_action_groups(states)
        _, summary_reps, _, _ = self._encode_states(states)
        learned_probs, cwr_probs, mixed_probs = self._event_policy_distributions(
            states,
            candidate_actions,
            summary_reps,
            random_spec=random_spec,
        )
        return {
            "learned": learned_probs,
            "cwr": cwr_probs,
            "mixed": mixed_probs,
        }


    def forward(self, input_dict):
        states = input_dict.get("states")
        if not states:
            raise ValueError("generator forward requires at least one state")
        random_spec = input_dict.get("random_spec")
        candidate_actions = self._candidate_action_groups(
            states,
            input_dict.get("candidate_actions"),
        )
        lineage_reps, summary_reps, lineage_seq_features, _ = self._encode_states(states)
        _, _, mixed_event_probs = self._event_policy_distributions(
            states,
            candidate_actions,
            summary_reps,
            random_spec=random_spec,
        )
        selected_event_indices = torch.distributions.Categorical(
            probs=mixed_event_probs
        ).sample()
        batch_indices = torch.arange(len(states), device=self.device)
        log_event_pf = torch.log(mixed_event_probs[batch_indices, selected_event_indices])
        selected_event_indices_list = selected_event_indices.detach().cpu().tolist()
        all_actions = [
            candidate_actions[batch_idx][event_idx]
            for batch_idx, event_idx in enumerate(selected_event_indices_list)
        ]

        log_action_pf, chosen_actions, chosen_action_features = self.arg_model(
            all_actions, lineage_reps, summary_reps, random_spec,
        )

        chosen_actions, log_breakpoint_pf = self._sample_breakpoints(
            chosen_actions, chosen_action_features, lineage_seq_features, random_spec,
        )
        chosen_actions, log_time_pf = self._sample_times(
            states, chosen_actions, random_spec,
        )
        total_log_pf = log_event_pf + log_action_pf + log_breakpoint_pf + log_time_pf
        return total_log_pf, chosen_actions

    def _sample_breakpoints(
        self, actions, action_features, lineage_features, random_spec,
    ):
        log_probabilities = []
        actions = list(actions)
        for idx, chosen_action in enumerate(actions):
            if isinstance(chosen_action, RecombinationChoice):
                lineage_idx = int(chosen_action.active_lineage_i)
                breakpoint, log_p_breakpoint = self.breakpoint_model(
                    chosen_action,
                    lineage_features[idx, lineage_idx],
                    int(self.env.num_blocks),
                    action_context=action_features[idx],
                    random_spec=random_spec,
                )
                actions[idx] = replace(chosen_action, breakpoint=breakpoint)
                log_probabilities.append(log_p_breakpoint)
            else:
                log_probabilities.append(torch.tensor(0.0, device=self.device))
        return actions, torch.stack(log_probabilities)

    def _sample_times(self, states, actions, random_spec):
        post_action_states = [
            self.env.preview_action_for_time_model(state, action)
            for state, action in zip(states, actions)
        ]
        _, post_action_summary_reps, _, _ = self._encode_states(post_action_states)
        time_logits = self.time_model(post_action_summary_reps)
        time_actions = self.time_model.sample(time_logits, random_spec)
        actions = list(actions)
        for batch_idx, action in enumerate(actions):
            time = int(time_actions[batch_idx].detach().cpu().item())
            actions[batch_idx] = replace(action, time_action=time)
        return actions, self.time_model.compute_log_time_pf(time_logits, time_actions)
    def update_model(self):
        grad_norm_pre = self.grad_norm()
        info = {
            "grad_norm": grad_norm_pre,
            "grad_norm_pre": grad_norm_pre,
            "log_z_grad_norm": self._grad_norm([self._Z]),
            "param_norm": self.param_norm(),
            "loss": self.loss.detach().cpu().item(),
        }
        info.update({
            f"grad_norm_{name}": self._grad_norm(parameters)
            for name, parameters in self.gradient_groups.items()
        })
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        info["grad_norm_post"] = self.grad_norm()
        self.opt.step()
        self.opt.zero_grad()
        self.loss = torch.tensor(0.0, device=self.device)
        return info

    def get_loss_from_rollout_outputs(self, rollout_outputs):
        log_paths_pf = rollout_outputs["log_paths_pf"]
        log_paths_pb = rollout_outputs["log_paths_pb"]
        log_rewards = torch.as_tensor(
            rollout_outputs["log_rewards"],
            dtype=log_paths_pf.dtype,
            device=log_paths_pf.device,
        )
        log_pf = log_paths_pf.sum(-1)
        log_pb = log_paths_pb.sum(-1)
        log_z = self.compute_log_Z().reshape(-1).to(log_paths_pf)
        return self.loss_fn(log_z + log_pf, log_rewards + log_pb)

    def accumulate_loss(self, rollout_outputs, factor=1.0):
        loss = self.get_loss_from_rollout_outputs(rollout_outputs) / factor
        loss.backward()
        self.loss += loss.detach()

    @staticmethod
    def _grad_norm(params):
        return math.sqrt(sum(
            parameter.grad.detach().norm().item() ** 2
            for parameter in params if parameter.grad is not None
        ))

    @staticmethod
    def _param_norm(params):
        return math.sqrt(sum(
            parameter.detach().norm().item() ** 2 for parameter in params
        ))

    def grad_norm(self):
        return self._grad_norm(self.gradient_clipping_params)

    def param_norm(self):
        return self._param_norm(self.gradient_clipping_params)

    def compute_log_Z(self):
        return self._Z
