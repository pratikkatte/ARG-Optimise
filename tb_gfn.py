import math
import os

import numpy as np
import torch

from models import ARGModel
from rollout_worker_arg import RolloutWorker

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
        verbose=False,
        log_z_lr=0.001,
    ):
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.device = device
        self.init_z_sample_count = init_z_sample_count

        ## Policy model
        self.arg_model = ARGModel(env, cfg=cfg).to(self.device)

        ## Z partition
        self._logZ = torch.nn.Parameter(
            torch.tensor(0.0, device=self.device),
            requires_grad=True,
        )
        self.max_reward_seen = float("-inf")

        # self._initialize_log_z_from_rollouts()
        self.policy_params = list(self.arg_model.parameters())
        self.log_z_params = [self._logZ]
        params = [{'params': self.policy_params, 'lr': 0.001}]
        params.append({'params': [self._logZ], 'lr': float(log_z_lr)})

        self.gradient_clipping_params = self.policy_params + self.log_z_params
        self.grad_clip = 10.0
        self.opt = torch.optim.Adam(
            params,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            amsgrad=True,
        )

        self.loss_fn = LOSS_FN['MSE']
        self.loss = torch.tensor(0.0, device=self.device)
        self.accumulated_batches = 0
        self.log_z_target_sum = 0.0
        self.log_z_target_count = 0
        self.last_log_z_target = float(self.compute_log_Z().detach().cpu().item())

    def _initialize_log_z_from_rollouts(self):
        if self.init_z_sample_count <= 0:
            raise ValueError("init_z_sample_count must be positive")
        if self.verbose:
            print(f"Initializing scalar logZ from {self.init_z_sample_count} on-policy rollout(s)...")
        
        worker = RolloutWorker(self.env, verbose=self.verbose)

        with torch.no_grad():
            outputs, _ = worker.rollout(
                self,
                episodes=int(self.init_z_sample_count),
                compute_reward=True,
            )
            log_pf = outputs["log_paths_pf"].sum(-1).to(self.device)
            log_pb = outputs["log_paths_pb"].sum(-1).to(self.device)
            log_rewards = outputs["log_rewards"].to(self.device)
            targets = log_rewards + log_pb - log_pf
            finite_targets = targets[torch.isfinite(targets)]
            finite_rewards = log_rewards[torch.isfinite(log_rewards)]
            if finite_rewards.numel() > 0:
                self.max_reward_seen = float(finite_rewards.max().detach().cpu().item())
            if finite_targets.numel() > 0:
                self._logZ.data.copy_(finite_targets.mean().detach())

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
        checkpoint = self._torch_load(path, map_location=map_location)
        state_dict = checkpoint.get("generator_state_dict", checkpoint)
        self.load_state_dict(state_dict)
        self.to(self.device)

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

    @staticmethod
    def _grad_norm(params):
        return math.sqrt(sum(
            p.grad.detach().norm().item() ** 2 for p in params if p.grad is not None
        ))

    @staticmethod
    def _param_norm(params):
        return math.sqrt(sum(p.detach().norm().item() ** 2 for p in params))

    def grad_norm(self):
        return self._grad_norm(self.gradient_clipping_params)
    
    def param_norm(self):
        return self._param_norm(self.gradient_clipping_params)

    def policy_grad_norm(self):
        return self._grad_norm(self.policy_params)

    def policy_param_norm(self):
        return self._param_norm(self.policy_params)

    def log_z_grad(self):
        if self._logZ.grad is None:
            return 0.0
        return float(self._logZ.grad.detach().cpu().reshape(-1)[0].item())

    def log_z_grad_norm(self):
        return self._grad_norm([self._logZ])

    def compute_log_Z(self):
        return self._logZ

    def forward(self, input_dict):
        # input_dict = self._move_input_to_device(input_dict)
        ret = self.arg_model(input_dict)
        return ret

    def update_model(self):

        raw_grad_norm = self.grad_norm()
        raw_policy_grad_norm = self.policy_grad_norm()
        raw_log_z_grad = self.log_z_grad()
        raw_log_z_grad_norm = self.log_z_grad_norm()
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        info = {
            'raw_grad_norm': raw_grad_norm,
            'grad_norm': self.grad_norm(),
            'raw_policy_grad_norm': raw_policy_grad_norm,
            'policy_grad_norm': self.policy_grad_norm(),
            'param_norm': self.param_norm(),
            'policy_param_norm': self.policy_param_norm(),
            'raw_log_z_grad': raw_log_z_grad,
            'log_z_grad': self.log_z_grad(),
            'raw_log_z_grad_norm': raw_log_z_grad_norm,
            'log_z_grad_norm': self.log_z_grad_norm(),
            'log_z_target': self.last_log_z_target,
            'loss': self.loss.detach().cpu().item(),
        }
        self.opt.step()
        self.opt.zero_grad()
        self.loss = torch.tensor(0.0, device=self.device)
        self.accumulated_batches = 0
        self.log_z_target_sum = 0.0
        self.log_z_target_count = 0
        if self.verbose:
            print(
                "update: loss={loss:.6f} raw_grad_norm={raw_grad_norm:.4f} "
                "grad_norm={grad_norm:.4f} param_norm={param_norm:.4f}".format(
                    **info
                )
            )
        return info

    def _record_log_z_targets(self, targets):
        finite_targets = targets[torch.isfinite(targets)]
        if finite_targets.numel() == 0:
            return
        self.log_z_target_sum += float(finite_targets.sum().detach().cpu().item())
        self.log_z_target_count += int(finite_targets.numel())
        self.last_log_z_target = (
            self.log_z_target_sum / max(self.log_z_target_count, 1)
        )


    def sample_backward_from_arg(self, arg_state):
        state = arg_state.clone()
        reverse_actions = []
        backward_states = [state.clone()]
        num_parents_by_backward_step = []
        log_path_pb = 0.0

        while not self._is_initial_arg_state(state):
            inverse_actions = self._enumerate_inverse_arg_actions(state)
            num_parents = len(inverse_actions)
            if num_parents == 0:
                raise ValueError("No valid ARG parent states were found for backward sampling.")

            rng = getattr(self.env, "rng", None)
            if rng is not None and hasattr(rng, "randrange"):
                inverse_action = inverse_actions[rng.randrange(num_parents)]
            else:
                inverse_action = inverse_actions[np.random.randint(num_parents)]

            state, forward_action = self._apply_inverse_arg_action(state, inverse_action)
            reverse_actions.append(forward_action)
            num_parents_by_backward_step.append(num_parents)
            log_path_pb -= math.log(num_parents)
            backward_states.append(state.clone())

        return {
            "forward_actions": list(reversed(reverse_actions)),
            "log_path_pb": log_path_pb,
            "num_parents": list(reversed(num_parents_by_backward_step)),
            "backward_states": backward_states,
        }

    def count_backward_parents(self, arg_state):
        return len(self._enumerate_inverse_arg_actions(arg_state))

    def _is_initial_arg_state(self, state):
        initial_ids = set(range(self.env.num_sequences))
        if set(state.all_nodes) != initial_ids:
            return False
        if {lineage.node_id for lineage in state.active_lineages} != initial_ids:
            return False

        for node_id in initial_ids:
            lineage = state.all_nodes[node_id]
            if lineage.children or lineage.parents:
                return False
            if lineage.material_segments.segments != ((0, self.env.num_blocks),):
                return False
        return True

    def _enumerate_inverse_arg_actions(self, state):
        inverse_actions = []

        for active_idx, lineage in enumerate(state.active_lineages):
            if lineage.event_type != "coal" or len(lineage.children) != 2:
                continue
            if not self._is_latest_time_event(state, lineage.node_id):
                continue
            child_i, child_j = lineage.children
            if (
                child_i in state.all_nodes
                and child_j in state.all_nodes
                and lineage.node_id in state.all_nodes[child_i].parents
                and lineage.node_id in state.all_nodes[child_j].parents
            ):
                inverse_actions.append(
                    {
                        "event_type": "coal",
                        "active_idx": active_idx,
                        "parent_id": lineage.node_id,
                        "child_ids": (child_i, child_j),
                    }
                )

        recomb_by_event = {}
        for active_idx, lineage in enumerate(state.active_lineages):
            if (
                lineage.event_type != "recomb"
                or len(lineage.children) != 1
                or lineage.breakpoint is None
                or lineage.recombination_side not in ("left", "right")
            ):
                continue
            key = (lineage.children[0], lineage.breakpoint)
            recomb_by_event.setdefault(key, {})[lineage.recombination_side] = (active_idx, lineage.node_id)

        for (child_id, breakpoint), sides in recomb_by_event.items():
            if "left" not in sides or "right" not in sides or child_id not in state.all_nodes:
                continue
            left_idx, left_id = sides["left"]
            right_idx, right_id = sides["right"]
            child = state.all_nodes[child_id]
            left_parent = state.all_nodes[left_id]
            right_parent = state.all_nodes[right_id]
            if not self._is_latest_time_event(state, left_id, right_id):
                continue
            if set(child.parents) != {left_id, right_id}:
                continue
            if left_parent.material_segments.intersection_count(right_parent.material_segments) > 0:
                continue
            if left_parent.material_segments.union(right_parent.material_segments) != child.material_segments:
                continue
            inverse_actions.append(
                {
                    "event_type": "recomb",
                    "active_indices": (left_idx, right_idx),
                    "parent_ids": (left_id, right_id),
                    "child_id": child_id,
                    "breakpoint": breakpoint,
                }
            )

        return inverse_actions

    def _is_latest_time_event(self, state, *node_ids):
        current_time = float(state.current_time)
        return all(
            math.isclose(
                float(state.all_nodes[node_id].time),
                current_time,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for node_id in node_ids
        )

    def _max_node_time(self, state):
        if not state.all_nodes:
            return 0.0
        return max(float(lineage.time) for lineage in state.all_nodes.values())

    def _apply_inverse_arg_action(self, state, inverse_action):
        if inverse_action["event_type"] == "coal":
            return self._apply_inverse_coalescence(state, inverse_action)
        if inverse_action["event_type"] == "recomb":
            return self._apply_inverse_recombination(state, inverse_action)
        raise ValueError(f"Unknown inverse ARG action: {inverse_action}")

    def _apply_inverse_coalescence(self, state, inverse_action):
        parent_state = state.clone()
        parent_id = inverse_action["parent_id"]
        child_ids = inverse_action["child_ids"]

        remaining_lineages = [
            lineage for lineage in parent_state.active_lineages if lineage.node_id != parent_id
        ]
        parent_state.all_nodes.pop(parent_id)
        parent_state.active_lineages = []
        for child_id in child_ids:
            child = parent_state.all_nodes[child_id]
            child.parents = [node_id for node_id in child.parents if node_id != parent_id]
            parent_state.active_lineages.append(child)
        parent_state.active_lineages.extend(remaining_lineages)
        parent_state.total_active_blocks = None

        active_idx_by_id = self._active_index_by_node_id(parent_state)
        forward_action = {
            "event_type": "coal",
            "active_lineage_i": active_idx_by_id[child_ids[0]],
            "active_lineage_j": active_idx_by_id[child_ids[1]],
        }
        parent_state.current_time = self._max_node_time(parent_state)
        delta_t = float(state.current_time) - float(parent_state.current_time)
        rates = self.env.enumerate_prior_options(parent_state).rates
        forward_action["time_action"] = self.env._time_action_for_delta(delta_t, rates)
        self._finalize_backward_parent_state(parent_state, state, forward_action)
        return parent_state, forward_action

    def _apply_inverse_recombination(self, state, inverse_action):
        parent_state = state.clone()
        left_id, right_id = inverse_action["parent_ids"]
        child_id = inverse_action["child_id"]

        remaining_lineages = [
            lineage for lineage in parent_state.active_lineages if lineage.node_id not in (left_id, right_id)
        ]
        parent_state.all_nodes.pop(left_id)
        parent_state.all_nodes.pop(right_id)

        child = parent_state.all_nodes[child_id]
        child.parents = []
        parent_state.active_lineages = [child] + remaining_lineages
        parent_state.total_active_blocks = None

        active_idx_by_id = self._active_index_by_node_id(parent_state)
        forward_action = {
            "event_type": "recomb",
            "active_lineage_i": active_idx_by_id[child_id],
            "breakpoint": inverse_action["breakpoint"],
        }
        parent_state.current_time = self._max_node_time(parent_state)
        delta_t = float(state.current_time) - float(parent_state.current_time)
        rates = self.env.enumerate_prior_options(parent_state).rates
        forward_action["time_action"] = self.env._time_action_for_delta(delta_t, rates)
        self._finalize_backward_parent_state(parent_state, state, forward_action)
        return parent_state, forward_action

    def _finalize_backward_parent_state(self, parent_state, child_state, forward_action):
        parent_state.max_node_idx = max(parent_state.all_nodes) if parent_state.all_nodes else -1
        parent_state.log_reward = None
        parent_state.action_options = None
        parent_state.rates = None
        parent_state.prior_options = None
        parent_state.is_done = self.env.is_terminal(parent_state)

        log_prior = self.env.compute_cwr_event_log_prior(parent_state, forward_action)
        if math.isfinite(log_prior):
            parent_state.accumulated_log_prior = child_state.accumulated_log_prior - log_prior
        parent_state.action_options = None
        parent_state.rates = None
        parent_state.prior_options = None

    def _active_index_by_node_id(self, state):
        return {lineage.node_id: idx for idx, lineage in enumerate(state.active_lineages)}

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

        
        log_z = self.compute_log_Z().reshape(-1).to(log_paths_pf)

        residuals = log_z + log_pf - (log_rewards + log_pb)

        loss = residuals.pow(2).mean()
        
        return loss
    
    def accumulate_loss(self, rollout_outputs, factor=1.0):
        loss = self.get_loss_from_rollout_outputs(rollout_outputs) / factor
        loss.backward()
        self.loss = self.loss + loss.detach()
        self.accumulated_batches += 1
        if self.verbose:
            print(
                f"accumulated loss={loss.item():.6f} "
                f"total_loss={self.loss.item():.6f} batches={self.accumulated_batches}"
            )

    def accumulate_streaming_tb_loss(self,rollout_outputs,):
        """Accumulate exact TB gradients without retaining the full rollout graph."""
        log_paths_pf = rollout_outputs["log_paths_pf"].detach().to(self.device)
        log_paths_pb = rollout_outputs["log_paths_pb"].detach().to(device=self.device,)
        log_rewards = torch.as_tensor(rollout_outputs["log_rewards"], device=self.device)


        log_pf = log_paths_pf.sum(-1)
        log_pb = log_paths_pb.sum(-1)

        log_z = self.compute_log_Z().detach()


        target_log_z_by_traj = (log_rewards + log_pb - log_pf).detach()
        self._record_log_z_targets(target_log_z_by_traj)
        policy_log_z = (
            target_log_z_by_traj[torch.isfinite(target_log_z_by_traj)].mean()
            if torch.isfinite(target_log_z_by_traj).any()
            else self.compute_log_Z().detach().to(log_paths_pf)
        )
        
        log_z_value = self.compute_log_Z().detach().to(log_paths_pf)
        
        residuals = (log_z_value + log_pf - (log_rewards + log_pb)).detach()

        loss = residuals.pow(2).mean()

        return loss
