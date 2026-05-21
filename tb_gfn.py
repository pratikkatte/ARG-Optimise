import math
import os

import numpy as np
import torch

try:
    from .models import ARGModel
except ImportError:
    from models import ARGModel

try:
    from .rollout_worker_arg import RolloutWorker
except ImportError:
    from rollout_worker_arg import RolloutWorker

LOSS_FN = {
    'MSE': torch.nn.MSELoss(),
    'HUBER': torch.nn.HuberLoss(delta=1.0),
}

class TBGFlowNetGenerator(torch.nn.Module):
    def __init__(self, env, init_z_sample_count, cfg=None, device=None, verbose=False):
        super().__init__()
        self.cfg = cfg
        self.env = env
        self.verbose = verbose
        self.device = self._resolve_device(device)
        self.init_z_sample_count = init_z_sample_count
        ## Policy model
        self.arg_model = ARGModel(env, cfg).to(self.device)

        ## Z partition
        with torch.no_grad():
            if self.verbose:
                print(
                    f"Initializing log_Z from {self.init_z_sample_count} prior rollout(s)..."
                )
            worker = RolloutWorker(env, verbose=self.verbose)
            trajs = []
            for _ in range(self.init_z_sample_count):
                state, _ = worker._rollout_one()
                trajs.append(state)
        self.max_reward_seen = np.max([x.log_reward for x in trajs])
        init_z = self.max_reward_seen
        self._Z = torch.nn.Parameter(
            torch.ones(256, device=self.device) * init_z / 256,
            requires_grad=True,
        )
        model_params = list(self.arg_model.parameters())
        params = [{'params': model_params, 'lr': 0.001}]
        params.append({'params': [self._Z], 'lr': 0.001})

        self.gradient_clipping_params = model_params
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

    def _resolve_device(self, device):
        if device is None or device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available.")
        return resolved

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

    def grad_norm(self):
        return math.sqrt(sum(
            p.grad.norm().item() ** 2 for p in self.gradient_clipping_params if p.grad is not None
        ))
    
    def param_norm(self):
        return math.sqrt(sum(p.norm().item() ** 2 for p in self.gradient_clipping_params))

    def compute_log_Z(self):
        return self._Z.sum()

    def forward(self, input_dict):
        input_dict = self._move_input_to_device(input_dict)
        ret = self.arg_model(input_dict)
        return ret

    def _move_input_to_device(self, input_dict):
        moved = {}
        for key, value in input_dict.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        return moved

    def update_model(self):
        info = {
            'grad_norm': self.grad_norm(),
            'param_norm': self.param_norm(),
            'loss': self.loss.detach().cpu().item(),
        }
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        self.opt.step()
        self.opt.zero_grad()
        self.loss = torch.tensor(0.0, device=self.device)
        self.accumulated_batches = 0
        if self.verbose:
            print(
                "update: loss={loss:.6f} grad_norm={grad_norm:.4f} param_norm={param_norm:.4f}".format(
                    **info
                )
            )
        return info

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
        if not getattr(self.env, "learn_times", False):
            return True
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
        if getattr(self.env, "learn_times", False):
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
        if getattr(self.env, "learn_times", False):
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
        log_paths_pb = torch.as_tensor(log_paths_pb, dtype=log_paths_pf.dtype, device=log_paths_pf.device)
        if log_paths_pb.ndim == 0:
            log_pb = log_paths_pb.expand_as(log_pf)
        elif log_paths_pb.shape == log_pf.shape:
            log_pb = log_paths_pb
        else:
            log_pb = log_paths_pb.sum(-1)
        
        log_z = self.compute_log_Z().reshape(-1).to(log_paths_pf)
        forward_value = log_z + log_pf
        backward_value = log_rewards + log_pb
        if backward_value.ndim == 0 and forward_value.ndim > 0:
            backward_value = backward_value.expand_as(forward_value)
        elif backward_value.shape != forward_value.shape and backward_value.numel() == forward_value.numel():
            backward_value = backward_value.reshape_as(forward_value)
        return self.loss_fn(forward_value, backward_value)
    
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

    def accumulate_streaming_tb_loss(self, rollout_worker, rollout_outputs, trajectories, factor=1.0):
        """Accumulate exact TB gradients without retaining the full rollout graph."""
        log_paths_pf = rollout_outputs["log_paths_pf"].detach().to(self.device)
        log_paths_pb = rollout_outputs["log_paths_pb"].detach().to(
            dtype=log_paths_pf.dtype,
            device=self.device,
        )
        log_rewards = torch.as_tensor(
            rollout_outputs["log_rewards"],
            dtype=log_paths_pf.dtype,
            device=self.device,
        )
        log_pf = log_paths_pf.sum(-1)
        log_pb = log_paths_pb.sum(-1) if log_paths_pb.ndim > 1 else log_paths_pb
        log_z_value = self.compute_log_Z().detach().to(log_paths_pf)
        residuals = (log_z_value + log_pf - (log_rewards + log_pb)).detach()
        batch_size = max(int(residuals.numel()), 1)
        coefficients = (2.0 / (float(batch_size) * float(factor))) * residuals
        loss_value = (residuals.pow(2).mean() / float(factor)).detach()

        (coefficients.sum() * self.compute_log_Z()).backward()
        for traj_idx, trajectory in enumerate(trajectories):
            coefficient = coefficients[traj_idx].detach()
            if coefficient.item() == 0.0:
                continue
            self._replay_trajectory_for_streaming_gradient(
                rollout_worker,
                trajectory,
                coefficient,
            )

        self.loss = self.loss + loss_value
        self.accumulated_batches += 1
        if self.verbose:
            print(
                f"streamed loss={loss_value.item():.6f} "
                f"total_loss={self.loss.item():.6f} batches={self.accumulated_batches}"
            )

    def _replay_trajectory_for_streaming_gradient(self, rollout_worker, trajectory, coefficient):
        state = self.env.get_initial_state()
        for record in trajectory:
            action = dict(record["action"])
            probs = self.env.compute_event_probabilities(state)
            event_prob = float(probs.get(action["event_type"], 0.0))
            if event_prob <= 0.0:
                raise RuntimeError(f"Cannot replay invalid event type from state: {action}")
            input_dict = self.env.prepare_state_rollout_inputs(
                [state],
                input_actions=[action],
                random_spec=None,
                device=self.device,
            )
            input_dict["selected_event_types"] = [action["event_type"]]
            input_dict["log_event_probs"] = torch.tensor(
                [math.log(event_prob)],
                dtype=torch.float32,
                device=self.device,
            )
            ret = self(input_dict)
            log_path_pf = ret["log_paths_pf"].reshape(-1)[0]
            (coefficient * log_path_pf).backward()

            log_prior = self.env.compute_cwr_event_log_prior(state, action)
            state = self.env.apply_action(
                state,
                action,
                log_prior,
                compute_reward=False,
            )
