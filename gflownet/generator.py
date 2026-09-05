from dataclasses import replace
import math

import numpy as np
import torch

from arg_environment import RecombinationChoice
from .backward import BackwardPolicyMixin
from .checkpoint import CheckpointMixin
from model import ARGModel

LOSS_FN = {"MSE": torch.nn.MSELoss(), "HUBER": torch.nn.HuberLoss(delta=1.0)}


class TBGFlowNetGenerator(CheckpointMixin, BackwardPolicyMixin, torch.nn.Module):
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
        initialize_z_from_prior=True,
    ):
        super().__init__()
        self.env = env
        self.verbose = verbose
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
        self.arg_model = ARGModel(env, **self.model_kwargs).to(self.device)
        self.time_model = self.arg_model.time_scorer
        self.breakpoint_model = self.arg_model.breakpoint_scorer

        ## Z partition
        self.max_reward_seen = float("-inf")
        if initialize_z_from_prior:
            log_rewards = env.sample_log_rewards(self.init_z_sample_count, verbose=verbose)
            self.max_reward_seen = float(np.max(log_rewards))
            init_Z = self.max_reward_seen
        else:
            self.max_reward_seen = 0.0
            init_Z = 0.0
        self._Z = torch.nn.Parameter(  # in log
                torch.ones(256, device=self.device) * init_Z / 256, requires_grad=True
                )
        
        self.arg_model_params = list(self.arg_model.parameters())
        self.policy_params = self.arg_model_params

        params = [{'params': self.arg_model_params, 'lr': self.arg_model_lr}]
        params.append({'params': [self._Z], 'lr': self.z_lr})

        # gradient clipping exclude the Z part
        self.gradient_clipping_params = list(self.arg_model.parameters())
        self.grad_clip = float(grad_clip)

        self.opt = torch.optim.Adam(
            params,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            amsgrad=True,
        )

        self.loss_fn = LOSS_FN['MSE']
        self.loss = torch.tensor(0.0, device=self.device)
        self.last_log_z_target = float(self.compute_log_Z().detach().cpu().item())


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
            cwr_probs.append([probabilities[event_type] for event_type in self.env.event_types])

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
        lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts = self._encode_states(states)
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

        ret = self.arg_model(all_actions, lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts, random_spec)

        log_action_pf, _, chosen_actions, chosen_action_features = ret

        log_p_breakpoints = []
        for idx, chosen_action in enumerate(chosen_actions):
            if isinstance(chosen_action, RecombinationChoice):
                lineage_idx = int(chosen_action.active_lineage_i)
                lineage_feature = lineage_seq_features[idx, lineage_idx]
                breakpoint, log_p_breakpoint = self.breakpoint_model(
                    chosen_action,
                    lineage_feature,
                    int(self.env.sequence_length),
                    int(self.env.num_blocks),
                    action_context=chosen_action_features[idx],
                    random_spec=random_spec,
                )
                chosen_actions[idx] = replace(chosen_action, breakpoint=breakpoint)
                log_p_breakpoints.append(log_p_breakpoint)
            else:
                log_p_breakpoints.append(torch.tensor(0.0, device=self.device))

        log_breakpoint_pf = torch.stack(log_p_breakpoints)

        post_action_states = [
            self.env.preview_action_for_time_model(state, action)
            for state, action in zip(states, chosen_actions)
        ]
        _, post_action_summary_reps, _, _ = self._encode_states(post_action_states)
        time_logits = self.time_model(post_action_summary_reps)
        time_actions = self.time_model.sample(time_logits, random_spec)

        for batch_idx, action in enumerate(chosen_actions):
            time = int(time_actions[batch_idx].detach().cpu().item())
            chosen_actions[batch_idx] = replace(action, time_action=time)

        log_time_pf = self.time_model.compute_log_time_pf(time_logits, time_actions)

        total_log_pf = log_event_pf + log_action_pf + log_breakpoint_pf + log_time_pf

        log_probs = torch.exp(total_log_pf)
        
        return total_log_pf, log_probs, chosen_actions


    def update_model(self):
        
        info = {'grad_norm': self.grad_norm(),
                'param_norm': self.param_norm(),
                'loss': self.loss.detach().cpu().numpy().tolist()}
        
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        self.opt.step()
        self.opt.zero_grad()
        self.loss = 0

        return info

    def get_loss_from_rollout_outputs(self, rollout_outputs):
        log_paths_pf = rollout_outputs['log_paths_pf']
        log_paths_pb = rollout_outputs['log_paths_pb']
        log_rewards = torch.as_tensor(
            rollout_outputs['log_rewards'],
            dtype=log_paths_pf.dtype,
            device=log_paths_pf.device,
        )

        log_pf = log_paths_pf.sum(-1)
        log_pb = log_paths_pb.sum(-1)

        
        log_z = self.compute_log_Z(None).reshape(-1).to(log_paths_pf)

        forward_value = log_z + log_pf
        backward_value = log_rewards + log_pb

        loss = self.loss_fn(forward_value, backward_value)

        return loss
        
    
    def accumulate_loss(self, rollout_outputs, factor=1.0):
        loss = self.get_loss_from_rollout_outputs(rollout_outputs)
        loss = (loss / factor)
        loss.backward()
        self.loss += loss 

    @staticmethod
    def _grad_norm(params):
        return math.sqrt(sum(
            parameter.grad.detach().norm().item() ** 2
            for parameter in params if parameter.grad is not None
        ))

    @staticmethod
    def _param_norm(params):
        return math.sqrt(sum(parameter.detach().norm().item() ** 2 for parameter in params))

    def grad_norm(self):
        return self._grad_norm(self.gradient_clipping_params)

    def param_norm(self):
        return self._param_norm(self.gradient_clipping_params)

    def compute_log_Z(self, scale_key=None):
        return self._Z.sum()

