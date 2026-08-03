import math
import os
import tempfile

import numpy as np
import torch

try:
    from .models import ARGModel
    from .env import (
        CoalescenceChoice,
        FixedAttachmentChoice,
        RecombinationChoice,
    )
except ImportError:  # Support the repository's script-style entry points.
    from models import ARGModel
    from env import (
        CoalescenceChoice,
        FixedAttachmentChoice,
        RecombinationChoice,
    )
from dataclasses import replace

LOSS_FN = {
    'MSE': torch.nn.MSELoss(),
    'HUBER': torch.nn.HuberLoss(delta=1.0),
}
LOSS_MODES = {"tb", "subtb", "fl_subtb"}
CHECKPOINT_FORMAT_VERSION = 2

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
        time_policy_lr=None,
        log_z_lr=None,
        initialize_z_from_prior=True,
        loss_mode="tb",
        subtb_lambda=0.9,
        subtb_max_span=None,
    ):
        super().__init__()
        print(f"verbose: {verbose}")
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
        for attr in (
            "variant_position_tensor",
            "variant_boundary_tensor",
            "variant_prev_gap_tensor",
            "variant_next_gap_tensor",
        ):
            if hasattr(self.env, attr):
                value = getattr(self.env, attr)
                setattr(
                    self.env,
                    attr,
                    torch.nn.Parameter(value.detach().to(self.device), requires_grad=False),
                )
        self.init_z_sample_count = int(init_z_sample_count)

        ## Policy model
        if policy_lr is not None:
            arg_model_lr = policy_lr
        if log_z_lr is not None:
            z_lr = log_z_lr
        self.arg_model_lr = float(arg_model_lr)
        self.time_policy_lr = (
            None if time_policy_lr is None else float(time_policy_lr)
        )
        if self.time_policy_lr is not None and self.time_policy_lr <= 0.0:
            raise ValueError("time_policy_lr must be positive when provided")
        self.z_lr = float(z_lr)
        self.loss_mode = str(loss_mode).lower()
        if self.loss_mode not in LOSS_MODES:
            raise ValueError(f"loss_mode must be one of {sorted(LOSS_MODES)}, got {loss_mode!r}")
        self.subtb_lambda = float(subtb_lambda)
        if self.subtb_lambda <= 0.0:
            raise ValueError(f"subtb_lambda must be positive, got {subtb_lambda!r}")
        self.subtb_max_span = (
            None if subtb_max_span is None else int(subtb_max_span)
        )
        if self.subtb_max_span is not None and self.subtb_max_span <= 0:
            raise ValueError(
                f"subtb_max_span must be positive when provided, got {subtb_max_span!r}"
            )
        self.model_kwargs = dict(model_kwargs or {})
        self.arg_model = ARGModel(env, **self.model_kwargs).to(self.device)
        self.time_model = self.arg_model.time_scorer
        self.breakpoint_model = self.arg_model.breakpoint_scorer

        ## Z partition
        self.max_reward_seen = float("-inf")
        if initialize_z_from_prior and self.init_z_sample_count > 0:
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

        if self.time_policy_lr is None:
            params = [{'params': self.arg_model_params, 'lr': self.arg_model_lr}]
        else:
            time_parameter_ids = {
                id(parameter) for parameter in self.time_model.parameters()
            }
            non_time_parameters = [
                parameter
                for parameter in self.arg_model_params
                if id(parameter) not in time_parameter_ids
            ]
            params = [
                {'params': non_time_parameters, 'lr': self.arg_model_lr},
                {
                    'params': list(self.time_model.parameters()),
                    'lr': self.time_policy_lr,
                },
            ]
        if self.loss_mode == "tb":
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

        self.scheduler = None

        self.loss_fn = LOSS_FN['MSE']

        self.grad_norm = lambda model: math.sqrt(sum(
            [p.grad.norm().item() ** 2 for p in self.gradient_clipping_params if p.grad is not None]))
        self.param_norm = lambda model: math.sqrt(sum([p.norm().item() ** 2 for p in self.gradient_clipping_params]))

        # Retained for checkpoint/runtime compatibility; AMP is not otherwise
        # used by this trainer yet.
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda",
        )

        self.loss = 0

        self.loss = torch.tensor(0.0, device=self.device)
        self.accumulated_batches = 0
        self.log_z_target_sum = 0.0
        self.log_z_target_count = 0
        self.last_log_z_target = float(self.compute_log_Z().detach().cpu().item())
        self.last_time_subtb_diagnostics = {}


    def _encode_states(self, states):
        return self.arg_model._encode_states(states)


    def save(self, path, metadata=None, training_state=None):
        """Atomically save a portable training/inference checkpoint.

        Model metadata is kept separate from mutable training state so
        inference only needs to inspect ``metadata`` and the model weights.
        ``opt_state_dict`` remains the canonical optimizer key for backward
        compatibility with existing checkpoints.
        """
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "generator_state_dict": self.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "metadata": dict(metadata or {}),
            "training_state": dict(training_state or {}),
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            payload["scaler_state_dict"] = self.scaler.state_dict()

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory,
                prefix=f".{os.path.basename(path)}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = handle.name
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def load(self, path, load_optimizer=True, map_location=None):
        if map_location is None:
            map_location = self.device
        checkpoint = (
            path
            if isinstance(path, dict)
            else self._torch_load(path, map_location=map_location)
        )
        metadata = dict(checkpoint.get("metadata") or {})
        if (
            any(
                key in metadata
                for key in (
                    "time_bin_scheme",
                    "time_bins",
                    "time_delta_bin_width",
                )
            )
            or str(metadata.get("model_version", "")).endswith("-v1")
        ):
            raise ValueError(
                "Fixed-bin v1 checkpoints are incompatible with "
                "continuous-time v2 and must be retrained."
            )
        state_dict = checkpoint.get("generator_state_dict", checkpoint)
        load_result = self.load_state_dict(state_dict, strict=False)
        allowed_missing = [
            key for key in load_result.missing_keys
            if (
                key.startswith("arg_model.flow_head.")
                or key.startswith("arg_model.local_")
            )
        ]
        unexpected = list(load_result.unexpected_keys)
        non_flow_missing = [
            key for key in load_result.missing_keys
            if key not in allowed_missing
        ]
        if non_flow_missing or unexpected:
            raise RuntimeError(
                "Checkpoint state_dict is incompatible with this generator: "
                f"missing={non_flow_missing}, unexpected={unexpected}"
            )
        self.to(self.device)
        self.last_log_z_target = float(self.compute_log_Z().detach().cpu().item())

        if load_optimizer and "opt_state_dict" in checkpoint:
            self.opt.load_state_dict(checkpoint["opt_state_dict"])
            self._move_optimizer_state_to_device()
        if (
            load_optimizer
            and self.scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if load_optimizer and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint.get("metadata", {})

    def _move_optimizer_state_to_device(self):
        for state in self.opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _torch_load(self, path, map_location=None):
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
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
        if self._Z.grad is None:
            return 0.0
        return float(self._Z.grad.detach().cpu().reshape(-1)[0].item())

    def log_z_grad_norm(self):
        return self._grad_norm([self._Z])

    def compute_log_Z(self, scale_key=None):
        return self._Z.sum()

    def compute_log_state_flows(self, states):
        if not states:
            return torch.empty(0, dtype=self._model_dtype(), device=self.device)

        log_flows = [None] * len(states)
        nonterminal_states = []
        nonterminal_indices = []
        for idx, state in enumerate(states):
            if state.is_done:
                if state.log_reward is None:
                    raise ValueError("Terminal ARGState is missing log_reward")
                log_flows[idx] = torch.tensor(
                    float(state.log_reward),
                    dtype=self._model_dtype(),
                    device=self.device,
                )
            else:
                nonterminal_indices.append(idx)
                nonterminal_states.append(state)

        if nonterminal_states:
            _, summary_reps, _, _ = self._encode_states(nonterminal_states)
            nonterminal_log_flows = self.arg_model.compute_log_state_flows(summary_reps)
            for idx, log_flow in zip(nonterminal_indices, nonterminal_log_flows):
                if self.loss_mode == "fl_subtb":
                    log_flow = log_flow + log_flow.new_tensor(
                        float(getattr(states[idx], "partial_log_reward", 0.0))
                    )
                log_flows[idx] = log_flow

        return torch.stack(log_flows)

    def compute_root_log_flow(self):
        return self.compute_log_state_flows([self.env.get_initial_state()])[0]

    def _model_dtype(self):
        return next(self.arg_model.parameters()).dtype

    def _sample_continuous_times(
        self,
        selected_action_features,
        states,
        selected_actions,
        rates,
        max_deltas,
        random_spec,
    ):
        context_features = self.time_model.context_features(
            rates,
            max_deltas,
            device=self.device,
            dtype=selected_action_features.dtype,
        )
        biological_context, context_diagnostics = (
            self.arg_model.build_time_context(
                states,
                selected_actions,
                max_deltas,
                dtype=selected_action_features.dtype,
            )
        )
        mixture_logits = self.time_model(
            torch.cat(
                [
                    selected_action_features,
                    context_features,
                    biological_context,
                ],
                dim=-1,
            )
        )
        time_quantiles = self.time_model.sample(
            mixture_logits,
            random_spec,
        )
        delta_times = []
        generated_masses = []
        for quantile, rate, max_delta, state in zip(
            time_quantiles.detach().cpu().tolist(),
            rates,
            max_deltas,
            states,
        ):
            delta_time = self.env.time_env.quantile_to_delta(
                float(quantile),
                float(rate),
                max_delta=max_delta,
            )
            boundary_time = (
                None
                if max_delta is None
                else self.env.next_fixed_ancestor_time(state)
            )
            delta_time = self.env.time_env.event_time_after_delta(
                delta_time,
                state.current_time,
                boundary_time,
            ) - float(state.current_time)
            delta_times.append(delta_time)
            generated_masses.append(
                self.env.time_env.generated_probability(
                    float(rate),
                    max_delta=max_delta,
                )
            )
        log_time_pf = self.time_model.log_time_density(
            mixture_logits,
            time_quantiles,
            delta_times,
            rates,
            generated_masses,
            random_spec=random_spec,
        )
        mixture_diagnostics = self.time_model.mixture_diagnostics(
            mixture_logits,
            random_spec=random_spec,
        )
        return (
            time_quantiles,
            delta_times,
            log_time_pf,
            context_diagnostics,
            mixture_diagnostics,
        )

    def forward(self, input_dict):
        if bool(input_dict.get("local_mode", False)):
            return self._forward_local(input_dict)

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

        lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts = self._encode_states(states)
        # input_dict = self._move_input_to_device(input_dict)
        ret = self.arg_model(all_actions, lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts, random_spec)

        log_action_pf, selected_action_indices, choosen_actions, choosen_action_features = ret

        log_p_breakpoints = []
        for idx, chosen_action in enumerate(choosen_actions):
            if isinstance(chosen_action, RecombinationChoice):
                lineage_idx = int(chosen_action.active_lineage_i)
                if getattr(self.env, "input_mode", "dense") == "vcf":
                    lineage_feature = states[idx].active_lineages[lineage_idx]
                else:
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

        selected_action_features = torch.stack(
            choosen_action_features,
            dim=0,
        )
        rates = [
            self.env._total_event_rate(state.rates)
            for state in states
        ]
        (
            time_quantiles,
            delta_times,
            log_time_pf,
            time_context_diagnostics,
            time_mixture_diagnostics,
        ) = (
            self._sample_continuous_times(
                selected_action_features,
                states,
                choosen_actions,
                rates,
                [None] * len(states),
                random_spec,
            )
        )

        for batch_idx, action in enumerate(choosen_actions):
            time_quantile = float(
                time_quantiles[batch_idx].detach().cpu().item()
            )
            choosen_actions[batch_idx] = replace(
                action,
                time_quantile=time_quantile,
                delta_time=float(delta_times[batch_idx]),
                waiting_rate=float(rates[batch_idx]),
                fixed_horizon=None,
                time_log_density=float(
                    log_time_pf[batch_idx].detach().cpu().item()
                ),
                time_policy_entropy=float(
                    time_mixture_diagnostics["entropy"][batch_idx]
                    .detach()
                    .cpu()
                    .item()
                ),
                time_effective_components=float(
                    time_mixture_diagnostics["effective_components"][batch_idx]
                    .detach()
                    .cpu()
                    .item()
                ),
                time_context_diagnostics=dict(
                    time_context_diagnostics[batch_idx]
                ),
            )

        total_log_pf = log_event_pf + log_action_pf + log_breakpoint_pf + log_time_pf

        log_probs = torch.exp(total_log_pf)
        
        return total_log_pf, log_probs, choosen_actions

    def _forward_local(self, input_dict):
        states = input_dict["states"]
        random_spec = input_dict.get("random_spec")
        rollout = input_dict["rollout"]
        all_actions = input_dict["input_actions"]

        (
            lineage_reps,
            summary_reps,
            lineage_seq_features,
            batch_active_lineage_counts,
        ) = self._encode_states(states)

        gate_logits = self.arg_model.compute_local_gate_logits(summary_reps)
        prior_gate_weight = float(
            getattr(self.arg_model, "local_prior_gate_logit_bias", 0.0)
        )
        if prior_gate_weight != 0.0:
            epsilon = torch.finfo(gate_logits.dtype).tiny
            prior_gate_rows = []
            for decision in rollout:
                generated_mass = (
                    float(decision["generated_prior_mass"])
                    if bool(decision["can_generate"])
                    else 0.0
                )
                fixed_mass = (
                    float(decision["survival_prior_mass"])
                    if bool(decision["can_attach_fixed"])
                    else 0.0
                )
                prior_gate_rows.append(
                    [
                        math.log(max(generated_mass, float(epsilon))),
                        math.log(max(fixed_mass, float(epsilon))),
                    ]
                )
            gate_logits = gate_logits + prior_gate_weight * torch.as_tensor(
                prior_gate_rows,
                dtype=gate_logits.dtype,
                device=self.device,
            )
        gate_mask = torch.tensor(
            [
                [
                    bool(decision["can_generate"]),
                    bool(decision["can_attach_fixed"]),
                ]
                for decision in rollout
            ],
            dtype=torch.bool,
            device=self.device,
        )
        masked_gate_logits = gate_logits.masked_fill(
            ~gate_mask,
            float("-inf"),
        )
        sample_gate_logits = masked_gate_logits
        if random_spec is not None and "T" in random_spec:
            sample_gate_logits = (
                sample_gate_logits / float(random_spec["T"])
            )
        gate_actions = torch.distributions.Categorical(
            logits=sample_gate_logits
        ).sample()
        gate_log_probabilities = torch.log_softmax(
            masked_gate_logits,
            dim=1,
        )
        batch_indices = torch.arange(
            len(states),
            dtype=torch.long,
            device=self.device,
        )
        selected_gate_log_pf = gate_log_probabilities[
            batch_indices,
            gate_actions,
        ]

        chosen_actions = [None] * len(states)
        total_log_pf = selected_gate_log_pf.clone()
        component_action = torch.zeros_like(selected_gate_log_pf)
        component_breakpoint = torch.zeros_like(selected_gate_log_pf)
        component_time = torch.zeros_like(selected_gate_log_pf)
        generated_indices = [
            index
            for index, gate_action in enumerate(
                gate_actions.detach().cpu().tolist()
            )
            if int(gate_action) == 0
        ]
        fixed_indices = [
            index
            for index, gate_action in enumerate(
                gate_actions.detach().cpu().tolist()
            )
            if int(gate_action) == 1
        ]

        for index in fixed_indices:
            event_time = rollout[index]["next_fixed_time"]
            if event_time is None:
                raise RuntimeError(
                    "local gate selected a fixed attachment without a boundary"
                )
            horizon = float(rollout[index]["max_delta"])
            waiting_rate = float(rollout[index]["total_rate"])
            chosen_actions[index] = FixedAttachmentChoice(
                event_time=float(event_time),
                waiting_rate=waiting_rate,
                fixed_horizon=horizon,
                survival_log_probability=-waiting_rate * horizon,
            )

        if generated_indices:
            generated_actions = [
                all_actions[index] for index in generated_indices
            ]
            if any(not actions for actions in generated_actions):
                raise RuntimeError(
                    "local gate selected generation without legal actions"
                )
            index_tensor = torch.tensor(
                generated_indices,
                dtype=torch.long,
                device=self.device,
            )
            generated_lineage_reps = lineage_reps.index_select(
                0,
                index_tensor,
            )
            generated_summary_reps = summary_reps.index_select(
                0,
                index_tensor,
            )
            generated_counts = batch_active_lineage_counts.index_select(
                0,
                index_tensor,
            )
            if torch.is_tensor(lineage_seq_features):
                generated_lineage_features = (
                    lineage_seq_features.index_select(0, index_tensor)
                )
            else:
                generated_lineage_features = [
                    lineage_seq_features[index]
                    for index in generated_indices
                ]
            (
                log_action_pf,
                _selected_action_indices,
                generated_chosen_actions,
                generated_action_features,
            ) = self.arg_model(
                generated_actions,
                generated_lineage_reps,
                generated_summary_reps,
                generated_lineage_features,
                generated_counts,
                random_spec,
            )

            log_breakpoint_pf = []
            for local_index, action in enumerate(
                generated_chosen_actions
            ):
                state_index = generated_indices[local_index]
                state = states[state_index]
                if isinstance(action, RecombinationChoice):
                    lineage_index = int(action.active_lineage_i)
                    lineage = state.active_lineages[lineage_index]
                    valid_breakpoints = self.env.valid_breakpoints(
                        state,
                        action,
                    )
                    breakpoint, log_probability = self.breakpoint_model(
                        valid_breakpoints,
                        lineage,
                        int(self.env.sequence_length),
                        max(len(state.block_boundaries or ()) - 1, 1),
                        action_context=generated_action_features[
                            local_index
                        ],
                        random_spec=random_spec,
                        state=state,
                    )
                    action = replace(action, breakpoint=breakpoint)
                    generated_chosen_actions[local_index] = action
                    log_breakpoint_pf.append(log_probability)
                else:
                    log_breakpoint_pf.append(
                        log_action_pf.new_tensor(0.0)
                    )
            log_breakpoint_pf = torch.stack(log_breakpoint_pf)

            selected_features = torch.stack(
                generated_action_features,
                dim=0,
            )
            rates = [
                float(rollout[index]["total_rate"])
                for index in generated_indices
            ]
            max_deltas = [
                rollout[index]["max_delta"]
                for index in generated_indices
            ]
            (
                time_quantiles,
                delta_times,
                log_time_pf,
                time_context_diagnostics,
                time_mixture_diagnostics,
            ) = (
                self._sample_continuous_times(
                    selected_features,
                    [states[index] for index in generated_indices],
                    generated_chosen_actions,
                    rates,
                    max_deltas,
                    random_spec,
                )
            )

            for local_index, action in enumerate(
                generated_chosen_actions
            ):
                state_index = generated_indices[local_index]
                time_quantile = float(
                    time_quantiles[local_index].detach().cpu().item()
                )
                diagnostics = dict(time_context_diagnostics[local_index])
                sampled_delta = float(delta_times[local_index])
                lower_distance = sampled_delta
                upper_distance = (
                    None
                    if max_deltas[local_index] is None
                    else max(float(max_deltas[local_index]) - sampled_delta, 0.0)
                )
                diagnostics.update(
                    {
                        "sampled_quantile": time_quantile,
                        "sampled_delta_time": sampled_delta,
                        "sampled_event_time": float(states[state_index].current_time)
                        + sampled_delta,
                        "distance_to_lower_bound": lower_distance,
                        "distance_to_upper_bound": upper_distance,
                        "normalized_sample_location": time_quantile,
                    }
                )
                chosen_actions[state_index] = replace(
                    action,
                    time_quantile=time_quantile,
                    delta_time=sampled_delta,
                    waiting_rate=float(rates[local_index]),
                    fixed_horizon=(
                        None
                        if max_deltas[local_index] is None
                        else float(max_deltas[local_index])
                    ),
                    time_log_density=float(
                        log_time_pf[local_index]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    time_policy_entropy=float(
                        time_mixture_diagnostics["entropy"][local_index]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    time_effective_components=float(
                        time_mixture_diagnostics["effective_components"][local_index]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    time_context_diagnostics=diagnostics,
                )
                total_log_pf[state_index] = (
                    total_log_pf[state_index]
                    + log_action_pf[local_index]
                    + log_breakpoint_pf[local_index]
                    + log_time_pf[local_index]
                )
                component_action[state_index] = log_action_pf[local_index]
                component_breakpoint[state_index] = log_breakpoint_pf[local_index]
                component_time[state_index] = log_time_pf[local_index]

        if any(action is None for action in chosen_actions):
            raise RuntimeError("local policy failed to choose every action")
        self.last_forward_log_components = {
            "gate": selected_gate_log_pf.detach(),
            "atomic_action": component_action.detach(),
            "breakpoint": component_breakpoint.detach(),
            "time": component_time.detach(),
            "total": total_log_pf.detach(),
        }
        return total_log_pf, torch.exp(total_log_pf), chosen_actions


    def update_model(self):
        time_gradient_squared = sum(
            parameter.grad.norm().item() ** 2
            for parameter in self.time_model.parameters()
            if parameter.grad is not None
        )
        info = {'grad_norm': self.grad_norm(self),
                'time_head_grad_norm': math.sqrt(time_gradient_squared),
                # 'z_grad_norm': self._Z.grad.norm().item(),
                'param_norm': self.param_norm(self),
                'loss': self.loss.detach().cpu().numpy().tolist()}
        info.update(self.last_time_subtb_diagnostics)
        
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        self.opt.step()
        self.opt.zero_grad()
        self.loss = 0

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

    def count_backward_parents(self, arg_state):
        if bool(getattr(self.env, "is_local", False)):
            return self.env.backward_parent_count(arg_state)
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

        # Use one loop to collect both coal and recomb candidates efficiently
        # Prepare coal candidates in a single pass with a list comprehension
        coal_candidates = [
            (active_idx, lineage)
            for active_idx, lineage in enumerate(state.active_lineages)
            if (
                lineage.event_type == "coal"
                and len(lineage.children) == 2
                and self._is_latest_time_event(state, lineage.node_id)
                and lineage.children[0] in state.all_nodes
                and lineage.children[1] in state.all_nodes
                and lineage.node_id in state.all_nodes[lineage.children[0]].parents
                and lineage.node_id in state.all_nodes[lineage.children[1]].parents
            )
        ]
        for active_idx, lineage in coal_candidates:
            child_i, child_j = lineage.children
            inverse_actions.append(
                {
                    "event_type": "coal",
                    "active_idx": active_idx,
                    "parent_id": lineage.node_id,
                    "child_ids": (child_i, child_j),
                }
            )

        # Prepare recomb_by_event using a single pass with a dictionary
        recomb_by_event = {}
        for active_idx, lineage in enumerate(state.active_lineages):
            if (
                lineage.event_type == "recomb"
                and len(lineage.children) == 1
                and lineage.breakpoint is not None
                and lineage.recombination_side in ("left", "right")
            ):
                key = (lineage.children[0], lineage.breakpoint)
                recomb_by_event.setdefault(key, {})[lineage.recombination_side] = (active_idx, lineage.node_id)

        # We can iterate efficiently over recomb_by_event rather than collecting in a list
        for (child_id, breakpoint), sides in recomb_by_event.items():
            if "left" not in sides or "right" not in sides or child_id not in state.all_nodes:
                continue
            left_idx, left_id = sides["left"]
            right_idx, right_id = sides["right"]
            child = state.all_nodes[child_id]
            left_parent = state.all_nodes[left_id]
            right_parent = state.all_nodes[right_id]

            # Fast short-circuit checks, in a single conditional
            if (
                not self._is_latest_time_event(state, left_id, right_id)
                or set(child.parents) != {left_id, right_id}
                or left_parent.material_segments.intersection_count(right_parent.material_segments) > 0
                or left_parent.material_segments.union(right_parent.material_segments) != child.material_segments
            ):
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
        forward_action["time_quantile"] = (
            self.env.time_env.delta_to_quantile(
                delta_t,
                self.env._total_event_rate(rates),
            )
        )
        forward_action["delta_time"] = delta_t
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
        prior_options = self.env.enumerate_prior_options(parent_state)
        rates = prior_options.rates
        total_rate = self.env._total_event_rate(rates)
        reconstructed_choice = next(
            (
                choice
                for choice in prior_options.recomb_choices
                if int(choice.active_lineage_i)
                == int(forward_action["active_lineage_i"])
            ),
            None,
        )
        if reconstructed_choice is None:
            raise RuntimeError(
                "backward recombination has no matching forward candidate"
            )
        forward_action.update(
            {
                "material_count": int(
                    reconstructed_choice.material_count
                ),
                "span_start": int(reconstructed_choice.span_start),
                "span_end": int(reconstructed_choice.span_end),
            }
        )
        forward_action["time_quantile"] = (
            self.env.time_env.delta_to_quantile(delta_t, total_rate)
        )
        forward_action["delta_time"] = delta_t
        self._finalize_backward_parent_state(parent_state, state, forward_action)
        return parent_state, forward_action

    def _finalize_backward_parent_state(self, parent_state, child_state, forward_action):
        parent_state.max_node_idx = max(parent_state.all_nodes) if parent_state.all_nodes else -1
        parent_state.log_reward = None
        parent_state.action_options = None
        parent_state.rates = None
        parent_state.prior_options = None
        parent_state.total_active_blocks = sum(
            lineage.material_segments.count
            for lineage in parent_state.active_lineages
        )
        parent_state.is_done = self.env.is_terminal(parent_state)

        reconstructed_action = (
            CoalescenceChoice.from_action(forward_action)
            or RecombinationChoice.from_action(forward_action)
        )
        if reconstructed_action is None:
            raise RuntimeError(
                "backward reconstruction produced an invalid forward action"
            )
        log_prior = self.env.compute_cwr_event_log_prior(
            parent_state,
            reconstructed_action,
        )
        if math.isfinite(log_prior):
            parent_state.accumulated_log_prior = child_state.accumulated_log_prior - log_prior
        parent_state.action_options = None
        parent_state.rates = None
        parent_state.prior_options = None

    def _active_index_by_node_id(self, state):
        return {lineage.node_id: idx for idx, lineage in enumerate(state.active_lineages)}

    def get_loss_from_rollout_outputs(self, rollout_outputs):
        if self.loss_mode in {"subtb", "fl_subtb"}:
            return self.compute_subtb_loss_from_rollout_outputs(rollout_outputs)
        return self.compute_tb_loss_from_rollout_outputs(rollout_outputs)

    def compute_tb_loss_from_rollout_outputs(self, rollout_outputs):
        log_paths_pf = rollout_outputs['log_paths_pf']
        log_paths_pb = rollout_outputs['log_paths_pb']
        terminal_mask = rollout_outputs.get("terminal_mask")
        if terminal_mask is not None and not bool(terminal_mask.all().detach().cpu().item()):
            raise ValueError(
                "Trajectory balance requires terminal rollout outputs; "
                "use subtb/fl_subtb for capped partial-to-partial rollouts."
            )
        log_rewards = torch.as_tensor(
            rollout_outputs['log_rewards'],
            dtype=log_paths_pf.dtype,
            device=log_paths_pf.device,
        )
        if not bool(torch.isfinite(log_rewards).all().detach().cpu().item()):
            raise ValueError(
                "Trajectory balance requires finite terminal log rewards for every trajectory."
            )

        log_pf = log_paths_pf.sum(-1)
        log_pb = log_paths_pb.sum(-1)

        
        log_z = self.compute_log_Z(None).reshape(-1).to(log_paths_pf)

        forward_value = log_z + log_pf
        backward_value = log_rewards + log_pb

        loss = self.loss_fn(forward_value, backward_value)

        return loss

    def compute_subtb_loss_from_rollout_outputs(self, rollout_outputs):
        if "trajectory_states" not in rollout_outputs:
            raise ValueError("SubTB loss requires rollout_outputs['trajectory_states']")
        if "trajectory_lengths" not in rollout_outputs:
            raise ValueError("SubTB loss requires rollout_outputs['trajectory_lengths']")

        state_paths = rollout_outputs["trajectory_states"]
        flat_states = [state for path in state_paths for state in path]
        flat_log_flows = self.compute_log_state_flows(flat_states)

        log_flows_by_traj = []
        cursor = 0
        for path in state_paths:
            next_cursor = cursor + len(path)
            log_flows_by_traj.append(flat_log_flows[cursor:next_cursor])
            cursor = next_cursor

        loss = self._subtb_loss_from_log_flows(
            log_flows_by_traj,
            rollout_outputs["log_paths_pf"],
            rollout_outputs["log_paths_pb"],
            rollout_outputs["trajectory_lengths"],
            self.subtb_lambda,
            self.subtb_max_span,
        )
        self.last_time_subtb_diagnostics = self._time_subtb_diagnostics(
            log_flows_by_traj,
            rollout_outputs["log_paths_pf"],
            rollout_outputs["log_paths_pb"],
            rollout_outputs["trajectory_lengths"],
            rollout_outputs.get("trajectory_actions", ()),
            self.subtb_lambda,
            self.subtb_max_span,
        )
        return loss

    @staticmethod
    def _time_subtb_diagnostics(
        log_flows_by_traj,
        log_paths_pf,
        log_paths_pb,
        trajectory_lengths,
        trajectory_actions,
        subtb_lambda,
        subtb_max_span,
    ):
        """Summarize SubTB residuals spanning generated time actions."""

        lengths = (
            trajectory_lengths.detach().cpu().tolist()
            if torch.is_tensor(trajectory_lengths)
            else list(trajectory_lengths)
        )
        residuals = []
        weights = []
        one_step = []
        terminal_reaching = []
        for traj_idx, length_value in enumerate(lengths):
            length = int(length_value)
            actions = (
                trajectory_actions[traj_idx]
                if traj_idx < len(trajectory_actions)
                else ()
            )
            time_mask = [
                action.get("time_quantile") is not None
                for action in actions[:length]
            ]
            if not any(time_mask):
                continue
            zero = log_paths_pf.new_zeros(1)
            pf_prefix = torch.cat(
                [zero, torch.cumsum(log_paths_pf[traj_idx, :length], dim=0)]
            )
            pb_prefix = torch.cat(
                [zero, torch.cumsum(log_paths_pb[traj_idx, :length], dim=0)]
            )
            flows = log_flows_by_traj[traj_idx]
            for start in range(length):
                max_end = length + 1
                if subtb_max_span is not None:
                    max_end = min(
                        max_end,
                        start + int(subtb_max_span) + 1,
                    )
                for end in range(start + 1, max_end):
                    if not any(time_mask[start:end]):
                        continue
                    residual = (
                        flows[start]
                        + pf_prefix[end]
                        - pf_prefix[start]
                        - flows[end]
                        - pb_prefix[end]
                        + pb_prefix[start]
                    ).detach()
                    residuals.append(residual)
                    weights.append(float(subtb_lambda) ** (end - start))
                    if end - start == 1:
                        one_step.append(residual)
                    if end == length:
                        terminal_reaching.append(residual)
        if not residuals:
            return {
                "time_subtb_count": 0,
                "time_subtb_residual_mean": 0.0,
                "time_subtb_residual_variance": 0.0,
                "time_subtb_squared_residual_mean": 0.0,
                "time_subtb_one_step_squared_residual_mean": 0.0,
                "time_subtb_terminal_squared_residual_mean": 0.0,
            }
        values = torch.stack(residuals).to(torch.float64)
        weight_tensor = torch.as_tensor(
            weights,
            dtype=torch.float64,
            device=values.device,
        )
        weight_total = weight_tensor.sum().clamp_min(
            torch.finfo(weight_tensor.dtype).tiny
        )
        weighted_mean = torch.sum(values * weight_tensor) / weight_total

        def mean_square(items):
            if not items:
                return 0.0
            return float(
                torch.stack(items)
                .to(torch.float64)
                .square()
                .mean()
                .cpu()
                .item()
            )

        return {
            "time_subtb_count": int(values.numel()),
            "time_subtb_residual_mean": float(weighted_mean.cpu().item()),
            "time_subtb_residual_variance": float(
                (
                    torch.sum(
                        weight_tensor * (values - weighted_mean).square()
                    )
                    / weight_total
                )
                .cpu()
                .item()
            ),
            "time_subtb_squared_residual_mean": float(
                (torch.sum(weight_tensor * values.square()) / weight_total)
                .cpu()
                .item()
            ),
            "time_subtb_one_step_squared_residual_mean": mean_square(one_step),
            "time_subtb_terminal_squared_residual_mean": mean_square(
                terminal_reaching
            ),
        }

    @staticmethod
    def _subtb_loss_from_log_flows(
        log_flows_by_traj,
        log_paths_pf,
        log_paths_pb,
        trajectory_lengths,
        subtb_lambda,
        subtb_max_span=None,
    ):
        if subtb_max_span is not None:
            subtb_max_span = int(subtb_max_span)
            if subtb_max_span <= 0:
                raise ValueError("subtb_max_span must be positive when provided")
        if torch.is_tensor(trajectory_lengths):
            lengths = trajectory_lengths.detach().cpu().tolist()
        else:
            lengths = list(trajectory_lengths)

        weighted_sum = log_paths_pf.new_tensor(0.0)
        weight_sum = log_paths_pf.new_tensor(0.0)
        for traj_idx, length in enumerate(lengths):
            length = int(length)
            if length <= 0:
                continue
            log_flows = log_flows_by_traj[traj_idx]
            if int(log_flows.numel()) != length + 1:
                raise ValueError(
                    "Each SubTB state path must have exactly trajectory length + 1 "
                    f"log flows, got {int(log_flows.numel())} for length {length}"
                )

            zero = log_paths_pf.new_zeros(1)
            pf_prefix = torch.cat([
                zero,
                torch.cumsum(log_paths_pf[traj_idx, :length], dim=0),
            ])
            pb_prefix = torch.cat([
                zero,
                torch.cumsum(log_paths_pb[traj_idx, :length], dim=0),
            ])

            for start in range(length):
                max_end = length + 1
                if subtb_max_span is not None:
                    max_end = min(max_end, start + int(subtb_max_span) + 1)
                for end in range(start + 1, max_end):
                    span = end - start
                    log_pf = pf_prefix[end] - pf_prefix[start]
                    log_pb = pb_prefix[end] - pb_prefix[start]
                    residual = log_flows[start] + log_pf - log_flows[end] - log_pb
                    weight = log_paths_pf.new_tensor(float(subtb_lambda) ** span)
                    weighted_sum = weighted_sum + weight * residual.pow(2)
                    weight_sum = weight_sum + weight

        if float(weight_sum.detach().cpu().item()) == 0.0:
            return weighted_sum
        return weighted_sum / weight_sum
        
    
    def accumulate_loss(self, rollout_outputs, factor=1.0):
        loss = self.get_loss_from_rollout_outputs(rollout_outputs)
        loss = (loss / factor)
        loss.backward()
        self.loss = self.loss + loss.detach()
