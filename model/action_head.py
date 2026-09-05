import torch

from arg_environment import CoalescenceChoice, RecombinationChoice


class ActionPolicyMixin:
    def compute_event_probabilities(
        self,
        summary_reps,
        event_valid_mask,
        prior_event_probs,
        random_spec=None,
    ):
        """Return learned and CwR-mixed event distributions for each state."""
        event_valid_mask = event_valid_mask.to(device=summary_reps.device, dtype=torch.bool)
        prior_event_probs = prior_event_probs.to(
            device=summary_reps.device,
            dtype=summary_reps.dtype,
        )
        if event_valid_mask.shape != prior_event_probs.shape:
            raise ValueError("event mask and prior probabilities must have matching shapes")
        if event_valid_mask.shape != (summary_reps.shape[0], len(self.env.event_types)):
            raise ValueError("event inputs must have shape (batch, number of event types)")
        if not bool(event_valid_mask.any(dim=1).all()):
            raise ValueError("every state must have at least one valid event type")

        event_logits = self.event_scorer(summary_reps)
        if random_spec is not None:
            event_logits = event_logits / float(random_spec["T"])
        event_logits = event_logits.masked_fill(~event_valid_mask, float("-inf"))
        learned_event_probs = torch.softmax(event_logits, dim=1)

        prior_event_probs = prior_event_probs.masked_fill(~event_valid_mask, 0.0)
        prior_normalizer = prior_event_probs.sum(dim=1, keepdim=True)
        if not bool((prior_normalizer > 0).all()):
            raise ValueError("CwR event probabilities must have positive mass")
        prior_event_probs = prior_event_probs / prior_normalizer

        mixed_event_probs = (
            (1.0 - self.event_prior_weight) * learned_event_probs
            + self.event_prior_weight * prior_event_probs
        )
        return learned_event_probs, mixed_event_probs


    def compute_log_path_pf(self, logits, action_indices):
        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        log_p = self.logsoftmax(logits)
        return log_p[batch_idx, action_indices]

    def _batched_action_features(self, actions, batch_idx, lineage_reps, summary_reps):
        num_actions = len(actions)
        embedding_size = lineage_reps.shape[-1]

        primary_rep = lineage_reps.new_zeros(num_actions, embedding_size)
        secondary_rep = lineage_reps.new_zeros(num_actions, embedding_size)
        tertiary_rep = lineage_reps.new_zeros(num_actions, embedding_size)

        coal_rows = [(row_idx, action.active_lineage_i, action.active_lineage_j) for row_idx, action in enumerate(actions) if isinstance(action, CoalescenceChoice)]
        if coal_rows:
            rows, left_indices, right_indices = zip(*coal_rows)
            rows = torch.tensor(rows, dtype=torch.long, device=self.device)
            left_indices = torch.tensor(left_indices, dtype=torch.long, device=self.device)
            right_indices = torch.tensor(right_indices, dtype=torch.long, device=self.device)
            left_rep = lineage_reps[batch_idx, left_indices]
            right_rep = lineage_reps[batch_idx, right_indices]
            primary_rep[rows] = left_rep + right_rep
            secondary_rep[rows] = torch.abs(left_rep - right_rep)
            tertiary_rep[rows] = left_rep * right_rep

        recomb_rows = [(row_idx, action.active_lineage_i, action.breakpoint) for row_idx, action in enumerate(actions) if isinstance(action, RecombinationChoice)]

        if recomb_rows:
            rows, lineage_indices, _ = zip(*recomb_rows)
            rows = torch.tensor(rows, dtype=torch.long, device=self.device)
            lineage_indices = torch.tensor(lineage_indices, dtype=torch.long, device=self.device)
            primary_rep[rows] = lineage_reps[batch_idx, lineage_indices]

        summary_for_actions = summary_reps[batch_idx].expand(num_actions, -1)
        return torch.cat([
            primary_rep,
            secondary_rep,
            tertiary_rep,
            summary_for_actions,
        ], dim=-1)

    def _score_candidates(
        self,
        candidate_actions,
        lineage_reps,
        summary_reps
        ):
        batch_size = len(candidate_actions)
        max_candidates = max(len(actions) for actions in candidate_actions)
        feat_dim = self.seq_embedding.out_features * 4
        features = lineage_reps.new_zeros(batch_size, max_candidates, feat_dim)

        candidate_counts = []

        for batch_idx, actions in enumerate(candidate_actions):
            n = len(actions)
            candidate_counts.append(n)
            state_action_features  = self._batched_action_features(
                actions,
                batch_idx,
                lineage_reps,
                summary_reps
            )
            features[batch_idx, :n] = state_action_features
        logits = self.action_scorer(features.reshape(-1, feat_dim)).reshape(batch_size, max_candidates)

        counts = torch.tensor(candidate_counts, device=self.device)
        valid = torch.arange(max_candidates, device=self.device).unsqueeze(0) < counts.unsqueeze(1)
        masked_logits = logits.masked_fill(~valid, -1e9)
        return masked_logits, features

    def forward(self, all_actions, lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts, random_spec):
        
        if any(len(actions) == 0 for actions in all_actions):
            raise ValueError("ARGModel.forward received a batch item with no candidate actions.")

        logits, action_features = self._score_candidates(
            all_actions,
            lineage_reps,
            summary_reps,
        )
        
        sampled_action_indices = self.sample(logits, random_spec)
        selected_action_indices = sampled_action_indices.detach().cpu().tolist()
        chosen_actions = [actions[i] for actions, i in zip(all_actions, selected_action_indices)]
        chosen_features = [action_features[b, i] for b, i in enumerate(selected_action_indices)]
        log_action_pf = self.compute_log_path_pf(logits, selected_action_indices)
        return log_action_pf, selected_action_indices, chosen_actions, chosen_features

