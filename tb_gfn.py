import math
import os
import warnings

import torch

from models import ARGModel, PackedLineageFeatures
from subtb import geometric_subtb_loss, subtb_diagnostics, validate_objective
from env.actions import CoalescenceChoice, RecombinationChoice
from rollout_worker_arg import RolloutWorker
from dataclasses import replace
from time_env import checkpoint_time_policy, validate_temperature
from flow_likelihood import PartialLikelihoodTracker
from flow_encoder import FrozenFlowEncoder
from learning_rate_schedule import WarmupCosineScheduler

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
        initialize_z_from_policy=True,
        loss_type="tb",
        subtb_lambda=0.9,
        flow_lr=None,
        flow_head_version=4,
    ):
        super().__init__()
        resolved_policy_lr = arg_model_lr if policy_lr is None else policy_lr
        self.flow_lr = float(resolved_policy_lr if flow_lr is None else flow_lr)
        self.loss_type = loss_type
        self.subtb_lambda = float(subtb_lambda)
        validate_objective(self.loss_type, self.subtb_lambda, self.flow_lr)
        if flow_head_version not in (1, 2, 3, 4, 5):
            raise ValueError("Unsupported flow-head version")
        self.flow_head_version = int(flow_head_version)
        self.neural_source_flow = self.flow_head_version == 5
        if self.neural_source_flow and self.loss_type != 'subtb':
            raise ValueError('Neural source flow requires SubTB')
        print(f"verbose: {verbose}")
        self.env = env
        self.verbose = verbose
        self.device = torch.device(device) if device is not None else torch.device(env.device)
        self.env.device = self.device
        if self.loss_type == "subtb" and self.flow_head_version >= 2:
            self.env.flow_likelihood = PartialLikelihoodTracker(self.env)
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
        self.init_z_sample_count = int(init_z_sample_count)
        if initialize_z_from_policy and self.init_z_sample_count < 1:
            raise ValueError("init_z_sample_count must be at least 1 for policy initialization")

        ## Policy model
        if policy_lr is not None:
            arg_model_lr = policy_lr
        if log_z_lr is not None:
            z_lr = log_z_lr
        self.arg_model_lr = float(arg_model_lr)
        self.z_lr = float(z_lr)
        self.model_kwargs = dict(model_kwargs or {})
        self.model_kwargs.setdefault("time_policy", env.time_policy)
        self.model_kwargs.setdefault("continuous_time_head", 'exponential')
        self.model_kwargs.setdefault("event_policy", "cwr")
        self.model_kwargs.setdefault("breakpoint_policy", "cnn")
        self.model_kwargs.setdefault("breakpoint_mixture_hidden_dim", 128)
        self.model_kwargs.setdefault("breakpoint_mixture_layers", 4)
        self.model_kwargs.setdefault("breakpoint_mixture_components", 4)
        self.arg_model = ARGModel(env, **self.model_kwargs).to(self.device)
        self.model_kwargs["time_policy"] = self.arg_model.time_policy
        self.time_model = self.arg_model.time_scorer
        self.breakpoint_model = self.arg_model.breakpoint_scorer

        ## Z partition
        self.max_reward_seen = float("-inf")
        # One scalar makes log_z_lr control the normalizer directly. Float64
        # preserves small updates even when log rewards have large magnitudes.
        if not self.neural_source_flow:
            self._Z = torch.nn.Parameter(
                torch.zeros((), dtype=torch.float64, device=self.device)
            )
        
        self.arg_model_params = list(self.arg_model.parameters())
        self.policy_params = self.arg_model_params

        params = [{'params': self.arg_model_params, 'lr': self.arg_model_lr}]
        if not self.neural_source_flow:
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

        # scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler()

        self.loss = 0

        self.loss = torch.tensor(0.0, device=self.device)
        self.tb_reporting_loss = 0.0
        self.accumulated_diagnostics = {}
        self.accumulated_batches = 0
        self.log_z_target_sum = 0.0
        self.log_z_target_count = 0
        self.last_log_z_target = 0.0
        self.initial_log_z_target_std = max(1.0, math.sqrt(self.env.sequence_length))

        if initialize_z_from_policy:
            self.initialize_log_z_from_policy()

        self.flow_params = []
        if self.loss_type == "subtb":
            # CPU construction under fork_rng leaves policy/sampling RNG unchanged.
            with torch.random.fork_rng(devices=[]):
                self.flow_head = torch.nn.Sequential(
                    torch.nn.Linear(self.model_kwargs.get("embedding_size", 32) + (8 if self.flow_head_version >= 2 else 4),
                                    self.model_kwargs.get("hidden_size", 64), device="cpu"),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.model_kwargs.get("hidden_size", 64), 1, device="cpu"),
                ).to(self.device)
                torch.nn.init.zeros_(self.flow_head[-1].weight)
                torch.nn.init.zeros_(self.flow_head[-1].bias)
            self.register_buffer("flow_init_offset", torch.tensor(
                self.last_log_z_target, dtype=torch.float64, device=self.device))
            if self.flow_head_version >= 2:
                self.register_buffer("flow_output_scale", torch.tensor(
                    max(1.0, self.initial_log_z_target_std), dtype=torch.float64, device=self.device))
            if self.flow_head_version >= 3:
                self.flow_encoder = FrozenFlowEncoder(self.arg_model)
            if self.flow_head_version == 4:
                # A fixed initialization buffer, not a view or detached alias
                # of the live normalizer. Subsequent Z updates cannot move it.
                self.register_buffer('flow_baseline_log_z', self.compute_log_Z().detach().clone())
            self.flow_params = list(self.flow_head.parameters())
            self.opt.add_param_group({"params": self.flow_params, "lr": self.flow_lr})
            self.gradient_clipping_params.extend(self.flow_params)

    def configure_lr_schedule(self, config, completed_updates=0):
        config.validate()
        if self.scheduler is not None:
            if self.scheduler.config != config or self.scheduler.completed_updates != completed_updates:
                raise ValueError('Requested learning-rate schedule differs from the resumed checkpoint')
            return
        if config.schedule == 'cosine':
            self.scheduler = WarmupCosineScheduler(self.opt, config, completed_updates)
            self._sync_learning_rates()

    def _sync_learning_rates(self):
        self.arg_model_lr = self.opt.param_groups[0]['lr']
        if not self.neural_source_flow:
            self.z_lr = self.opt.param_groups[1]['lr']
        if self.loss_type == 'subtb':
            self.flow_lr = self.opt.param_groups[1 if self.neural_source_flow else 2]['lr']

    @torch.no_grad()
    def initialize_log_z_from_policy(self):
        """Center TB residuals on rollouts from the initial, unchanged policy.

        Preserve the policy's mode (including training dropout) to match its
        rollout distribution. This is a sampled TB minimizer for fixed policy
        weights, not an exact estimate of the target's log partition function.
        """
        if self.init_z_sample_count < 1:
            raise ValueError("init_z_sample_count must be at least 1 for policy initialization")
        worker = RolloutWorker(self.env)
        targets = []
        max_reward = float("-inf")
        for index in range(self.init_z_sample_count):
            if self.verbose:
                print(
                    f"Sampling initial-policy trajectory {index + 1}/"
                    f"{self.init_z_sample_count} for log Z init..."
                )
            # One trajectory at a time bounds initialization memory for long sequences.
            outputs, _ = worker.rollout(self, episodes=1)
            log_pf = outputs["log_paths_pf"].sum(-1).double()
            log_pb = outputs["log_paths_pb"].sum(-1).double()
            log_rewards = outputs["log_rewards"].double()
            target = log_rewards + log_pb - log_pf
            if not torch.isfinite(target).all():
                raise ValueError("Initial-policy rollout produced a non-finite log Z target")
            targets.append(target.cpu())
            max_reward = max(max_reward, float(log_rewards.max().item()))

        targets = torch.cat(targets)
        initial_log_z = float(targets.mean().item())
        self.initial_log_z_target_std = float(targets.std(unbiased=False).item())
        # Keep the existing parameter and optimizer references.
        if not self.neural_source_flow:
            self._Z.fill_(initial_log_z)
        self.max_reward_seen = max_reward
        self.last_log_z_target = initial_log_z
        if self.verbose:
            print(
                f"Initialized flow center={initial_log_z:.4f} from "
                f"{targets.numel()} initial-policy trajectories "
                f"(target_std={targets.std(unbiased=False).item():.4f})"
            )


    def _encode_states(self, states):
        return self.arg_model._encode_states(states)


    def save(self, path, metadata=None):
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        metadata = dict(metadata or {})
        metadata.update(loss_type=self.loss_type, subtb_lambda=self.subtb_lambda,
                        flow_lr=self.flow_lr, policy_lr=self.arg_model_lr, log_z_lr=self.z_lr)
        metadata['arg_prior'] = self.env.arg_prior
        metadata['action_probability_version'] = 2
        if hasattr(self, '_action_probability_migration'):
            metadata['action_probability_migration'] = self._action_probability_migration
        if self.env.arg_prior == 'hudson':
            metadata['prior_stop_at_local_mrca'] = False
            metadata['prior_stopping_rule'] = 'Stop when every genomic position has one ancestor; retain resolved material until then'
        if self.neural_source_flow:
            for name in ('log_z_lr', 'log_z', 'initial_log_z', 'log_z_initialization'):
                metadata.pop(name, None)
            metadata.update(source_flow_parameterization='shared_flow_network',
                            source_log_flow=float(self.compute_source_log_flow().detach()),
                            flow_initialization='fixed_initial_policy_residual_mean')
        if self.loss_type == "subtb":
            metadata.update(flow_head_version=self.flow_head_version, flow_init_offset=self.flow_init_offset.item())
            if self.flow_head_version >= 2:
                metadata["flow_output_scale"] = self.flow_output_scale.item()
            if self.flow_head_version == 4:
                metadata["flow_baseline_log_z"] = self.flow_baseline_log_z.item()
        metadata["model"] = {**metadata.get("model", {}), **self.model_kwargs}
        metadata["time"] = dict(self.env.time_metadata)
        metadata.update(self.env.time_metadata)
        checkpoint = {
                "generator_state_dict": self.state_dict(),
                "opt_state_dict": self.opt.state_dict(),
                "metadata": metadata,
            }
        if self.scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.scheduler.state_dict()
            metadata['lr_schedule'] = self.scheduler.state_dict()['config']
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=True, map_location=None, allow_action_probability_migration=False):
        if map_location is None:
            map_location = self.device
        checkpoint = (
            path
            if isinstance(path, dict)
            else self._torch_load(path, map_location=map_location)
        )
        metadata = checkpoint.get("metadata", {})
        action_version = metadata.get('action_probability_version', 1)
        if action_version not in (1, 2):
            raise ValueError('Unsupported action probability version')
        if (load_optimizer and checkpoint.get('opt_state_dict', {}).get('state') and action_version == 1
                and not allow_action_probability_migration):
            raise ValueError('Checkpoint used batch-dependent single-candidate action scores. '
                             'Load weights with load_optimizer=False, start from update 0, or explicitly '
                             'allow_action_probability_migration; old metrics are not a corrected baseline.')
        if metadata.get('arg_prior', 'overlap') != self.env.arg_prior:
            raise ValueError('Cannot load checkpoints across ARG priors; start a fresh run')
        saved_time_policy = checkpoint_time_policy(metadata)
        if saved_time_policy != self.arg_model.time_policy:
            raise ValueError(
                "Cannot load checkpoints across time_policy modes: "
                f"checkpoint={saved_time_policy!r}, model={self.arg_model.time_policy!r}; use a fresh checkpoint"
            )
        saved_event_policy = metadata.get("model", {}).get("event_policy", "cwr")
        saved_time_head = metadata.get('model', {}).get('continuous_time_head', 'exponential')
        if saved_time_head != self.arg_model.continuous_time_head:
            raise ValueError('Cannot load across continuous time heads without explicit checkpoint migration')
        if saved_event_policy != self.arg_model.event_policy:
            raise ValueError(
                "Cannot load checkpoints across event_policy modes: "
                f"checkpoint={saved_event_policy!r}, model={self.arg_model.event_policy!r}"
            )
        if metadata.get("loss_type", "tb") != self.loss_type:
            raise ValueError("Cannot load checkpoints across loss_type objectives")
        if self.loss_type == "subtb":
            if metadata.get("flow_head_version") != self.flow_head_version:
                raise ValueError("Incompatible flow-head checkpoint version; construct the saved flow_head_version")
            if load_optimizer and metadata.get("subtb_lambda") != self.subtb_lambda:
                raise ValueError("Cannot restore optimizer with a different subtb_lambda")
        state_dict = checkpoint.get("generator_state_dict", checkpoint)
        saved_z = state_dict.get("_Z")
        legacy_vector_z = saved_z is not None and saved_z.shape == (256,)
        if legacy_vector_z:
            state_dict = state_dict.copy()
            # Match the old objective's reduction precision when preserving logZ.
            state_dict["_Z"] = (
                saved_z.double().sum() if self.loss_type == "subtb" else saved_z.sum()
            )
        if "metadata" in checkpoint:
            saved_policy = checkpoint["metadata"].get("model", {}).get("breakpoint_policy", "cnn")
            if saved_policy != self.model_kwargs["breakpoint_policy"]:
                raise ValueError("Cannot load weights across breakpoint policy types; create a new model for training")
        self.load_state_dict(state_dict)
        self.to(self.device)
        self.last_log_z_target = float(self.compute_log_Z().detach().cpu().item())

        if load_optimizer and "opt_state_dict" in checkpoint:
            optimizer_state = checkpoint["opt_state_dict"]
            if legacy_vector_z:
                # The scalar has different update semantics. Preserve all policy
                # and flow moments, but start its Adam history afresh.
                optimizer_state = optimizer_state.copy()
                optimizer_state["state"] = optimizer_state["state"].copy()
                z_id = optimizer_state["param_groups"][1]["params"][0]
                optimizer_state["state"].pop(z_id, None)
                warnings.warn(
                    "Converted legacy 256-value logZ to one scalar; reset only "
                    "logZ optimizer state. log_z_lr now controls one scalar update.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.opt.load_state_dict(optimizer_state)
            self._move_optimizer_state_to_device()
            self.arg_model_lr = self.opt.param_groups[0]["lr"]
            if not self.neural_source_flow:
                self.z_lr = self.opt.param_groups[1]["lr"]
            if self.loss_type == "subtb":
                self.flow_lr = self.opt.param_groups[1 if self.neural_source_flow else 2]["lr"]
            saved_scheduler = checkpoint.get('lr_scheduler_state_dict')
            self.scheduler = (WarmupCosineScheduler.from_state_dict(self.opt, saved_scheduler)
                              if saved_scheduler is not None else None)
            self._sync_learning_rates()
        if action_version == 1 and checkpoint.get('opt_state_dict', {}).get('state'):
            warnings.warn('Legacy checkpoint weights now use corrected single-candidate action probabilities; '
                          'previous density/loss/importance metrics may differ.', RuntimeWarning)
            self._action_probability_migration = dict(from_version=1,to_version=2,
                                                      retained_optimizer=bool(load_optimizer))
        elif 'action_probability_migration' in metadata:
            self._action_probability_migration = metadata['action_probability_migration']
        else:
            self.__dict__.pop('_action_probability_migration',None)
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
        if self.neural_source_flow:
            raise ValueError('Version 5 has no logZ parameter; inspect flow-head gradients')
        if self._Z.grad is None:
            return 0.0
        return float(self._Z.grad.detach().cpu().reshape(-1)[0].item())

    def log_z_grad_norm(self):
        if self.neural_source_flow:
            raise ValueError('Version 5 has no logZ parameter; inspect flow-head gradients')
        return self._grad_norm([self._Z])

    def compute_log_Z(self, scale_key=None):
        # Legacy reporting API. Version 5 evaluates F_theta(s0); it owns no Z.
        if self.neural_source_flow:
            return self.compute_source_log_flow()
        return self._Z

    def compute_source_log_flow(self):
        if not self.neural_source_flow:
            return self._Z
        state = self.env.get_initial_state()
        features = PackedLineageFeatures(
            torch.stack([self.env.evolution_model.normalize_partials(node.partials)
                         for node in state.active_lineages]),
            (0, self.env.num_sequences))
        summary = self.flow_encoder(features, torch.tensor([self.env.num_sequences], device=self.device))
        previous = getattr(self, '_last_flow_corrections', None)
        value = self.state_flows([state], summary)[0]
        self._last_flow_corrections = previous
        return value

    def state_flows(self, states, summary_reps):
        feature_rows = [
            [s.accumulated_log_prior / self.env.sequence_length,
             math.log1p(s.current_time), math.log1p(len(s.active_lineages)),
             s.total_active_blocks / self.env.num_blocks] for s in states
        ]
        if self.flow_head_version >= 2:
            initial_ll = self.env.flow_likelihood.initial_log_likelihood
            scale = self.flow_output_scale.item()
            remaining = []
            partial_likelihoods = []
            for state, row in zip(states, feature_rows):
                partial_ll = state.partial_log_likelihood
                if partial_ll is None and self.neural_source_flow and state.is_done:
                    # The exact terminal boundary also accepts inference states,
                    # where the optional intermediate likelihood tracker is off.
                    partial_ll = state.log_reward - self.env.reward_fn.C - state.accumulated_log_prior
                if partial_ll is None:
                    raise ValueError("Likelihood flow requires tracked partial likelihoods")
                partial_likelihoods.append(partial_ll)
                fraction = (state.total_active_blocks / self.env.num_blocks - 1) / max(self.env.num_sequences - 1, 1)
                ages = [max(0.0, state.current_time - node.time) for node in state.active_lineages]
                mean_age = sum(ages) / len(ages)
                age_std = math.sqrt(sum((age - mean_age)**2 for age in ages) / len(ages))
                remaining.append(fraction)
                row.extend([(partial_ll - initial_ll) / scale, fraction,
                            math.log1p(mean_age), math.log1p(age_std)])
        features = summary_reps.new_tensor(feature_rows)
        residual = self.flow_head(torch.cat((summary_reps, features), dim=-1)).squeeze(-1).double()
        prior = torch.tensor([s.accumulated_log_prior for s in states],
                             dtype=torch.float64, device=self.device)
        predicted = self.flow_init_offset + prior + residual
        if self.flow_head_version >= 2:
            partial_ll = prior.new_tensor(partial_likelihoods)
            source_potential = self.env.reward_fn.C + initial_ll
            baseline_log_z = (self.flow_init_offset if self.neural_source_flow else
                              self.flow_baseline_log_z if self.flow_head_version == 4
                              else self.compute_log_Z())
            predicted = (self.env.reward_fn.C + prior + partial_ll
                         + prior.new_tensor(remaining) * (baseline_log_z - source_potential)
                         + self.flow_output_scale * residual)
        self._last_flow_corrections = residual.detach() * (
            self.flow_output_scale if self.flow_head_version >= 2 else 1.0)
        # Each forward event allocates new node IDs, so this identifies the source in O(1).
        source = torch.tensor([s.max_node_idx == self.env.num_sequences - 1 for s in states], device=self.device)
        if self.neural_source_flow:
            terminal = torch.tensor([s.is_done for s in states], device=self.device)
            reward = prior.new_tensor([s.log_reward if s.is_done else 0. for s in states])
            return torch.where(terminal, reward, predicted)
        return torch.where(source, self.compute_log_Z().double(), predicted)

    def compute_event_probabilities(self, state):
        """Actual untempered forward event policy for evaluation."""
        if self.arg_model.event_policy == "cwr":
            return self.env.compute_event_probabilities(state)
        if state.is_done:
            return {event: 0.0 for event in self.env.event_types}
        inputs = self.env.prepare_state_rollout_inputs([state], event_policy="cwr_residual")
        _, summary, _, _ = self._encode_states([state])
        probs = self.arg_model.event_log_probs(
            [state], summary, inputs["event_actions"], inputs["event_prior_probs"],
        ).exp()[0]
        return dict(zip(self.env.event_types, probs.unbind()))

    def forward(self, input_dict, return_flows=False, forced_actions=None):
        if return_flows and self.loss_type != "subtb":
            raise ValueError("State flows require loss_type=subtb")

        states = input_dict.get("states")
        if forced_actions is not None:
            if len(forced_actions) != len(states):
                raise ValueError("Replay needs one action per state")
            if self.arg_model.event_policy != "cwr_residual":
                raise ValueError("Replay currently requires event_policy=cwr_residual")

        random_spec = input_dict.get("random_spec")
        if self.arg_model.time_policy == "cwr_exponential":
            validate_temperature(random_spec)
        

        lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts = self._encode_states(states)
        if self.arg_model.event_policy == "cwr_residual":
            event_actions = input_dict["event_actions"]
            event_log_probs = self.arg_model.event_log_probs(
                states, summary_reps, event_actions, input_dict["event_prior_probs"],
            )
            event_indices = (self.arg_model.sample(event_log_probs, random_spec) if forced_actions is None
                             else torch.tensor([int(isinstance(a, RecombinationChoice)) for a in forced_actions],
                                               dtype=torch.long, device=self.device))
            log_event_pf = event_log_probs.gather(1, event_indices[:, None]).squeeze(1)
            all_actions = [
                candidates[event_idx]
                for candidates, event_idx in zip(event_actions, event_indices.detach().cpu().tolist())
            ]
        else:
            event = input_dict["event"]
            event_probs = [float(event[idx]["probability"]) for idx in range(len(states))]
            log_event_pf = torch.log(
                torch.tensor(event_probs, dtype=(torch.float64 if self.arg_model.time_policy == "cwr_exponential"
                                                  else torch.float32), device=self.device)
            )
            all_actions = input_dict["input_actions"]
        # input_dict = self._move_input_to_device(input_dict)
        action_indices = None
        if forced_actions is not None:
            # Match the discrete candidate before adding its breakpoint/wait.
            candidates = [replace(a, time_action=None, delta_t=None,
                                  **({'breakpoint': None} if isinstance(a, RecombinationChoice) else {}))
                          for a in forced_actions]
            action_indices = [list(actions).index(action) for actions, action in zip(all_actions, candidates)]
        ret = self.arg_model(all_actions, lineage_reps, summary_reps, lineage_seq_features,
                             batch_active_lineage_counts, random_spec, action_indices=action_indices)

        log_action_pf, selected_action_indices, choosen_actions, choosen_action_features = ret

        log_p_breakpoints = []
        for idx, chosen_action in enumerate(choosen_actions):
            if isinstance(chosen_action, RecombinationChoice):
                lineage_idx = int(chosen_action.active_lineage_i)
                lineage_feature = lineage_seq_features.get_lineage(idx, lineage_idx)
                breakpoint, log_p_breakpoint = self.breakpoint_model(
                    chosen_action,
                    lineage_feature,
                    int(self.env.sequence_length),
                    int(self.env.num_blocks),
                    action_context=choosen_action_features[idx],
                    random_spec=random_spec,
                    **({'breakpoint': forced_actions[idx].breakpoint} if forced_actions is not None else {}),
                )
                choosen_actions[idx] = replace(chosen_action, breakpoint=breakpoint)
                log_p_breakpoints.append(log_p_breakpoint)
            else:
                log_p_breakpoints.append(torch.tensor(0.0, device=self.device))

        log_breakpoint_pf = torch.stack(log_p_breakpoints)

        selected_action_features = torch.stack(choosen_action_features, dim=0)  # shape: [B, F]
        if self.arg_model.time_policy == "cwr_exponential":
            time_features, baseline_rates = self.continuous_time_inputs(states, choosen_actions, selected_action_features)
            corrections = self.time_model(time_features)
            waits = (self.time_model.sample(corrections, baseline_rates, random_spec) if forced_actions is None
                     else torch.tensor([a.delta_t for a in forced_actions], dtype=torch.float64, device=self.device))
            for batch_idx, (action, wait) in enumerate(zip(choosen_actions, waits.cpu().tolist())):
                self.env.time_env.event_time(states[batch_idx].current_time, wait)
                choosen_actions[batch_idx] = replace(action, delta_t=wait)
            log_time_pf = self.time_model.compute_log_time_pf(corrections, waits, baseline_rates)
            total_log_pf = log_event_pf.double() + log_action_pf.double() + log_breakpoint_pf.double() + log_time_pf
            if not bool(torch.isfinite(total_log_pf).all()):
                raise ValueError("non-finite continuous joint forward score")
        else:
            time_logits = self.time_model(selected_action_features)
            time_actions = (self.time_model.sample(time_logits, random_spec) if forced_actions is None
                            else torch.tensor([a.time_action for a in forced_actions], device=self.device))

            for batch_idx, action in enumerate(choosen_actions):
                time = int(time_actions[batch_idx].detach().cpu().item())
                choosen_actions[batch_idx] = replace(action, time_action=time)

            log_time_pf = self.time_model.compute_log_time_pf(time_logits, time_actions)
            total_log_pf = log_event_pf + log_action_pf + log_breakpoint_pf + log_time_pf

        log_probs = torch.exp(total_log_pf)
        
        if return_flows:
            if self.flow_head_version >= 3:
                summary_reps = self.flow_encoder(lineage_seq_features, batch_active_lineage_counts)
            return total_log_pf, log_probs, choosen_actions, self.state_flows(states, summary_reps)
        return total_log_pf, log_probs, choosen_actions

    def continuous_time_inputs(self, states, actions, selected_action_features):
        rates = [self.env._total_event_rate(
            state.rates if state.rates is not None else self.env.enumerate_prior_options(state).rates
        ) for state in states]
        features = selected_action_features.new_tensor([
            [math.log1p(state.current_time), math.log(rate),
             float(isinstance(action, RecombinationChoice)),
             action.breakpoint / self.env.num_blocks if isinstance(action, RecombinationChoice) else 0.0]
            for state, action, rate in zip(states, actions, rates)
        ])
        if not bool(torch.isfinite(features).all()):
            raise ValueError("non-finite continuous timing features")
        return torch.cat((selected_action_features, features), dim=-1), torch.tensor(
            rates, dtype=torch.float64, device=selected_action_features.device)


    def update_model(self):
        
        info = {'grad_norm': self.grad_norm(self),
                # 'z_grad_norm': self._Z.grad.norm().item(),
                'param_norm': self.param_norm(self),
                'loss': self.loss.detach().cpu().numpy().tolist()}
        
        if self.loss_type == "subtb":
            info["subtb_loss"] = info["loss"]
            info["tb_loss"] = self.tb_reporting_loss
            self.tb_reporting_loss = 0.0
            info["policy_grad_norm"] = self.policy_grad_norm()
            info["flow_head_grad_norm"] = self._grad_norm(self.flow_params)
            if not self.neural_source_flow:
                info["log_z_grad"] = self.log_z_grad()
            info.update(self.accumulated_diagnostics)
            self.accumulated_diagnostics = {}
        if self.loss_type == "subtb" and self.flow_head_version >= 2:
            # The scaled flow head must not determine the policy clipping factor.
            torch.nn.utils.clip_grad_norm_(self.policy_params, self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.flow_params, self.grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        if self.scheduler is not None:
            info.update(policy_lr=self.opt.param_groups[0]['lr'],
                        lr_schedule_factor=self.scheduler.config.factor(self.scheduler.completed_updates))
            if self.loss_type == 'subtb':
                info['flow_lr'] = self.opt.param_groups[1 if self.neural_source_flow else 2]['lr']
            if not self.neural_source_flow:
                info['log_z_lr'] = self.opt.param_groups[1]['lr']
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()
            self._sync_learning_rates()
            info['lr_schedule_completed_updates'] = self.scheduler.completed_updates
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
        # Reversing a stored wait contributes no new time density. The triangular
        # map from waits to chronological event times has unit Jacobian.
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
        if self.env.time_policy == "cwr_exponential":
            expected = set(range(state.max_node_idx - len(node_ids) + 1, state.max_node_idx + 1))
            return set(node_ids) == expected and all(
                float(state.all_nodes[node_id].time) == current_time for node_id in node_ids
            )
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
        self._restore_backward_material(parent_state)

        active_idx_by_id = self._active_index_by_node_id(parent_state)
        forward_action = CoalescenceChoice(
            active_lineage_i=active_idx_by_id[child_ids[0]],
            active_lineage_j=active_idx_by_id[child_ids[1]],
        )
        parent_state.current_time = self._max_node_time(parent_state)
        delta_t = float(state.current_time) - float(parent_state.current_time)
        rates = self.env.enumerate_prior_options(parent_state).rates
        forward_action = replace(forward_action, **self.env.timing_for_delta(delta_t, rates))
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
        self._restore_backward_material(parent_state)

        active_idx_by_id = self._active_index_by_node_id(parent_state)
        forward_action = RecombinationChoice(
            active_lineage_i=active_idx_by_id[child_id],
            breakpoint=inverse_action["breakpoint"],
            span_start=child.material_span[0],
            span_end=child.material_span[1],
            material_count=child.material_span[2],
        )
        parent_state.current_time = self._max_node_time(parent_state)
        delta_t = float(state.current_time) - float(parent_state.current_time)
        rates = self.env.enumerate_prior_options(parent_state).rates
        forward_action = replace(forward_action, **self.env.timing_for_delta(delta_t, rates))
        self._finalize_backward_parent_state(parent_state, state, forward_action)
        return parent_state, forward_action

    def _restore_backward_material(self, state):
        state.total_active_blocks = sum(lineage.material_segments.count for lineage in state.active_lineages)
        state.action_options = state.rates = state.prior_options = None
        # Forward transitions discard inactive partials. Rebuild in allocation
        # order, which is topological, using the same block-resolution helpers.
        needed = {lineage.node_id for lineage in state.active_lineages if lineage.partials is None}
        pending = list(needed)
        while pending:
            node = state.all_nodes[pending.pop()]
            for child_id in node.children:
                if state.all_nodes[child_id].partials is None and child_id not in needed:
                    needed.add(child_id)
                    pending.append(child_id)
        for node_id in sorted(needed):
            node = state.all_nodes[node_id]
            if not node.children:
                node.partials = self.env._initial_lineage_partials(node_id, node.material_segments)
            elif node.event_type == "coal":
                node.partials = self.env._coalesced_parent_partials(
                    *(state.all_nodes[c] for c in node.children), node.material_segments, node.time)
            else:
                node.partials = self.env._recombined_parent_partials(
                    state.all_nodes[node.children[0]], node.material_segments, node.time)
        if self.env.flow_likelihood is not None:
            self.env.flow_likelihood.restore(state)

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
        if self.env.time_policy == "cwr_exponential" and not math.isfinite(parent_state.accumulated_log_prior):
            raise ValueError("non-finite reconstructed continuous prior score")
        parent_state.action_options = None
        parent_state.rates = None
        parent_state.prior_options = None

    def _active_index_by_node_id(self, state):
        return {lineage.node_id: idx for idx, lineage in enumerate(state.active_lineages)}

    def get_loss_from_rollout_outputs(self, rollout_outputs):
        if self.loss_type == "subtb":
            return geometric_subtb_loss(
                rollout_outputs["log_paths_pf"], rollout_outputs["log_paths_pb"],
                rollout_outputs["state_flows"], rollout_outputs["lengths"],
                rollout_outputs["log_rewards"], self.subtb_lambda,
            )
        return self.get_tb_loss_from_rollout_outputs(rollout_outputs)

    @torch.no_grad()
    def get_subtb_diagnostics(self, outputs, loss=None):
        if loss is None:
            loss = self.get_loss_from_rollout_outputs(outputs)
        metrics = subtb_diagnostics(outputs['log_paths_pf'], outputs['log_paths_pb'],
            outputs['state_flows'], outputs['lengths'], outputs['log_rewards'], loss, self.subtb_lambda)
        if 'flow_corrections' in outputs:
            corrections = outputs['flow_corrections']
            index = torch.arange(corrections.shape[1], device=corrections.device)[None, :]
            values = corrections[(index > 0) & (index < outputs['lengths'][:, None])]
            metrics['flow_correction_mean'] = values.mean().item() if values.numel() else 0.0
            metrics['flow_correction_std'] = values.std(unbiased=False).item() if values.numel() else 0.0
        return metrics

    def get_tb_loss_from_rollout_outputs(self, rollout_outputs):
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
        if self.loss_type == "subtb":
            with torch.no_grad():
                self.tb_reporting_loss += self.get_tb_loss_from_rollout_outputs(rollout_outputs).item() / factor
        loss = self.get_loss_from_rollout_outputs(rollout_outputs)
        if self.loss_type == "subtb":
            for key, value in self.get_subtb_diagnostics(rollout_outputs, loss).items():
                self.accumulated_diagnostics[key] = self.accumulated_diagnostics.get(key, 0.0) + value / factor
        loss = (loss / factor)
        loss.backward()
        self.loss = self.loss + loss.detach()
