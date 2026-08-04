"""Deterministic, policy-rescored evaluation banks for flow consistency."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch


FIXED_BANK_VERSION = 1


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _rng_snapshot(env):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "env": env.rng.getstate() if hasattr(env.rng, "getstate") else None,
    }


def _restore_rng(env, snapshot) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.random.set_rng_state(snapshot["torch"])
    if snapshot["cuda"] is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda"])
    if snapshot["env"] is not None:
        env.rng.setstate(snapshot["env"])


def _canonical_action(action: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "event_type",
        "active_lineage_i",
        "active_lineage_j",
        "material_count",
        "span_start",
        "span_end",
        "breakpoint",
        "event_time",
        "time_quantile",
        "delta_time",
        "waiting_rate",
        "fixed_horizon",
    )
    return {key: action.get(key) for key in fields if key in action}


def _state_to_cpu(state):
    cloned = state.clone(copy_partials=False)
    for lineage in cloned.all_nodes.values():
        if torch.is_tensor(lineage.partials):
            lineage.partials = lineage.partials.detach().cpu()
    return cloned


def fixed_bank_signature(bank: Mapping[str, Any]) -> str:
    payload = {
        "version": bank["version"],
        "seed": bank["seed"],
        "trajectories": [
            {
                "source": row["source"],
                "context_id": row.get("context_id"),
                "actions": [_canonical_action(action) for action in row["actions"]],
                "state_ids": [
                    hashlib.sha256(
                        repr(
                            state.structural_identity()
                            if hasattr(state, "structural_identity")
                            else state
                        ).encode("utf-8")
                    ).hexdigest()[:16]
                    for state in row["states"]
                ],
            }
            for row in bank["trajectories"]
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def fixed_bank_coverage(bank: Mapping[str, Any]) -> dict[str, Any]:
    trajectories = list(bank.get("trajectories", ()))
    lengths = [int(row.get("length", len(row.get("actions", ())))) for row in trajectories]
    return {
        "trajectory_count": len(trajectories),
        "sources": sorted({str(row.get("source")) for row in trajectories}),
        "complete_terminal_count": sum(
            bool(row.get("states"))
            and bool(
                getattr(
                    row["states"][-1],
                    "is_done",
                    row.get("terminal", True),
                )
            )
            for row in trajectories
        ),
        "coalescence_heavy_count": sum(
            int(row.get("coalescence_count", 0))
            >= int(row.get("recombination_count", 0))
            for row in trajectories
        ),
        "recombination_heavy_count": sum(
            int(row.get("recombination_count", 0))
            > int(row.get("coalescence_count", 0))
            for row in trajectories
        ),
        "fixed_attachment_count": sum(
            int(row.get("fixed_attachment_count", 0)) > 0
            for row in trajectories
        ),
        "max_trajectory_length": max(lengths, default=0),
    }


def validate_fixed_bank_coverage(
    bank: Mapping[str, Any],
    *,
    required_sources: Sequence[str] = (),
) -> dict[str, Any]:
    coverage = fixed_bank_coverage(bank)
    missing_sources = sorted(set(required_sources) - set(coverage["sources"]))
    missing = []
    if coverage["complete_terminal_count"] != coverage["trajectory_count"]:
        missing.append("complete terminal trajectories")
    if coverage["coalescence_heavy_count"] == 0:
        missing.append("a coalescence-heavy trajectory")
    if coverage["recombination_heavy_count"] == 0:
        missing.append("a recombination-heavy trajectory")
    if coverage["fixed_attachment_count"] == 0:
        missing.append("a fixed-attachment trajectory")
    if missing_sources:
        missing.append("trajectory sources " + ", ".join(missing_sources))
    if missing:
        raise ValueError("fixed evaluation bank lacks " + "; ".join(missing))
    return coverage


def generate_fixed_evaluation_bank(
    worker,
    generator,
    initial_states: Mapping[str, Any],
    *,
    episodes: int,
    seed: int,
    source: str,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Generate reproducible complete trajectories and restore all RNG state."""

    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("fixed evaluation bank episodes must be positive")
    if not initial_states:
        raise ValueError("fixed evaluation bank requires initial states")
    snapshot = _rng_snapshot(worker.env)
    was_training = generator.training
    context_ids = tuple(sorted(str(value) for value in initial_states))
    selector = random.Random(int(seed) + 17)
    trajectories = []
    try:
        generator.eval()
        _seed_everything(int(seed))
        if hasattr(worker.env.rng, "seed"):
            worker.env.rng.seed(int(seed))
        for index in range(episodes):
            context_id = selector.choice(context_ids)
            outputs, _ = worker.rollout(
                generator,
                episodes=1,
                start_states=[initial_states[context_id]],
                max_steps=max_steps,
            )
            if not bool(outputs["terminal_mask"][0].detach().cpu().item()):
                raise RuntimeError(
                    "fixed-bank generation hit max_steps before a terminal state"
                )
            actions = list(outputs["trajectory_actions"][0])
            action_types = [action.get("event_type") for action in actions]
            trajectories.append(
                {
                    "index": index,
                    "source": str(source),
                    "context_id": context_id,
                    "seed": int(seed),
                    "states": [
                        _state_to_cpu(state)
                        for state in outputs["trajectory_states"][0]
                    ],
                    "actions": actions,
                    "length": len(actions),
                    "coalescence_count": action_types.count("coal"),
                    "recombination_count": action_types.count("recomb"),
                    "fixed_attachment_count": action_types.count("fixed_attachment"),
                }
            )
    finally:
        _restore_rng(worker.env, snapshot)
        generator.train(was_training)
    bank = {
        "version": FIXED_BANK_VERSION,
        "seed": int(seed),
        "sources": [str(source)],
        "generation": {
            "episodes": episodes,
            "max_steps": max_steps,
            "context_ids": list(context_ids),
        },
        "trajectories": trajectories,
    }
    bank["coverage"] = fixed_bank_coverage(bank)
    bank["signature"] = fixed_bank_signature(bank)
    return bank


def merge_fixed_evaluation_banks(*banks: Mapping[str, Any]) -> dict[str, Any]:
    if not banks:
        raise ValueError("at least one fixed bank is required")
    versions = {int(bank["version"]) for bank in banks}
    if versions != {FIXED_BANK_VERSION}:
        raise ValueError("fixed evaluation bank versions are incompatible")
    trajectories = []
    for bank in banks:
        trajectories.extend(bank["trajectories"])
    merged = {
        "version": FIXED_BANK_VERSION,
        "seed": [bank["seed"] for bank in banks],
        "sources": sorted(
            {row["source"] for bank in banks for row in bank["trajectories"]}
        ),
        "generation": {"merged_signatures": [bank["signature"] for bank in banks]},
        "trajectories": trajectories,
    }
    merged["coverage"] = fixed_bank_coverage(merged)
    merged["signature"] = fixed_bank_signature(merged)
    return merged


def save_fixed_evaluation_bank(bank: Mapping[str, Any], path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(dict(bank), handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    manifest_path = f"{path}.json"
    with open(f"{manifest_path}.tmp", "w", encoding="utf-8") as handle:
        json.dump(
            {
                key: bank[key]
                for key in (
                    "version", "seed", "sources", "generation", "coverage", "signature"
                )
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(f"{manifest_path}.tmp", manifest_path)


def load_fixed_evaluation_bank(path: str) -> dict[str, Any]:
    with open(path, "rb") as handle:
        bank = pickle.load(handle)
    if int(bank.get("version", -1)) != FIXED_BANK_VERSION:
        raise ValueError("unsupported fixed evaluation bank version")
    signature = fixed_bank_signature(bank)
    if signature != bank.get("signature"):
        raise ValueError("fixed evaluation bank signature does not match its contents")
    return bank


def _pad_rows(rows: Sequence[torch.Tensor], device) -> torch.Tensor:
    width = max((int(row.numel()) for row in rows), default=0)
    dtype = rows[0].dtype if rows else torch.float32
    result = torch.zeros(len(rows), width, dtype=dtype, device=device)
    for index, row in enumerate(rows):
        result[index, : row.numel()] = row.to(device=device, dtype=dtype)
    return result


def _summary(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.detach().to(torch.float64)
    if values.numel() == 0:
        return {f"{prefix}_count": 0, f"{prefix}_mse": 0.0}
    return {
        f"{prefix}_count": int(values.numel()),
        f"{prefix}_mse": float(values.square().mean().cpu().item()),
        f"{prefix}_abs_mean": float(values.abs().mean().cpu().item()),
    }


def _selected_component_summary(
    values: Sequence[torch.Tensor],
    prefix: str,
    *,
    density: bool = False,
) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}/decision_count": 0,
            f"{prefix}/finite_rate": 0.0,
        }
    tensor = torch.stack(list(values)).detach().to(torch.float64)
    finite = tensor[torch.isfinite(tensor)]
    result = {
        f"{prefix}/decision_count": int(tensor.numel()),
        f"{prefix}/finite_rate": float(
            finite.numel() / max(tensor.numel(), 1)
        ),
    }
    if finite.numel():
        if density:
            result[f"{prefix}/selected_log_density_mean"] = float(
                finite.mean().cpu().item()
            )
        else:
            result[f"{prefix}/selected_nll_mean"] = float(
                (-finite).mean().cpu().item()
            )
    return result


def evaluate_fixed_bank(
    generator,
    bank: Mapping[str, Any],
    *,
    subtb_lambda: float = 0.9,
    subtb_max_span: int | None = 16,
) -> dict[str, float]:
    """Rescore a shared trajectory bank under ``generator`` without updates."""

    was_training = generator.training
    rows_pf = []
    rows_pb = []
    paths = []
    actions_by_path = []
    components_by_path = []
    diagnostics_by_path = []
    terminal_rewards = []
    sources = []
    try:
        generator.eval()
        with torch.no_grad():
            for trajectory in bank["trajectories"]:
                states = [
                    generator.env.clone_state_to_device(state)
                    if hasattr(generator.env, "clone_state_to_device")
                    else state
                    for state in trajectory["states"]
                ]
                actions = trajectory["actions"]
                scored = generator.score_local_transitions(states[:-1], actions)
                rows_pf.append(scored["total"])
                pb = torch.as_tensor(
                    [
                        generator.env.backward_log_probability(child)
                        for child in states[1:]
                    ],
                    dtype=scored["total"].dtype,
                    device=generator.device,
                )
                rows_pb.append(pb)
                paths.append(states)
                actions_by_path.append(actions)
                components_by_path.append(
                    [
                        {
                            name: float(scored[name][index].detach().cpu().item())
                            for name in ("gate", "atomic_action", "breakpoint", "time", "total")
                        }
                        for index in range(len(actions))
                    ]
                )
                diagnostics_by_path.append(scored["policy_diagnostics"])
                terminal_rewards.append(float(states[-1].log_reward))
                sources.append(str(trajectory["source"]))
            log_pf = _pad_rows(rows_pf, generator.device)
            log_pb = _pad_rows(rows_pb, generator.device)
            lengths = torch.as_tensor(
                [len(actions) for actions in actions_by_path],
                dtype=torch.long,
                device=generator.device,
            )
            outputs = {
                "log_paths_pf": log_pf,
                "log_paths_pb": log_pb,
                "log_rewards": torch.as_tensor(
                    terminal_rewards,
                    dtype=log_pf.dtype,
                    device=generator.device,
                ),
                "trajectory_states": paths,
                "trajectory_actions": actions_by_path,
                "trajectory_log_components": components_by_path,
                "trajectory_lengths": lengths,
                "terminal_mask": torch.ones(
                    len(paths), dtype=torch.bool, device=generator.device
                ),
            }
            flat_states = [state for path in paths for state in path]
            flat_flows = generator.compute_log_state_flows(flat_states)
            flows_by_path = []
            cursor = 0
            for path in paths:
                flows_by_path.append(flat_flows[cursor : cursor + len(path)])
                cursor += len(path)
            _, details = generator._subtb_loss_from_log_flows(
                flows_by_path,
                log_pf,
                log_pb,
                lengths,
                float(subtb_lambda),
                subtb_max_span,
                terminal_mask=outputs["terminal_mask"],
                terminal_loss_weight=1.0,
                residual_scale=1.0,
                trajectory_actions=actions_by_path,
                return_details=True,
            )
            records = details["records"]
            residuals = torch.stack([row["residual"] for row in records])
            weights = torch.stack([row["weight"] for row in records])
            one_step = torch.stack(
                [row["residual"] for row in records if row["span"] == 1]
            )
            terminal_one_step = torch.stack(
                [
                    row["residual"]
                    for row in records
                    if row["terminal"] and row["span"] == 1
                ]
            )
            tb_residuals = generator.compute_log_state_flows(
                [path[0] for path in paths]
            ) + log_pf.sum(1) - torch.as_tensor(
                terminal_rewards, dtype=log_pf.dtype, device=generator.device
            ) - log_pb.sum(1)
            metrics = {
                "flow_eval/fixed_bank_tb_mse": float(
                    tb_residuals.square().mean().cpu().item()
                ),
                "flow_eval/fixed_bank_subtb_mse": float(
                    (weights * residuals.square()).sum().div(weights.sum()).cpu().item()
                ),
                "flow_eval/fixed_bank_one_step_mse": float(
                    one_step.square().mean().cpu().item()
                ),
                "flow_eval/fixed_bank_terminal_mse": float(
                    terminal_one_step.square().mean().cpu().item()
                ),
                "flow_eval/fixed_bank_residual_p50": float(
                    residuals.abs().quantile(0.50).cpu().item()
                ),
                "flow_eval/fixed_bank_residual_p90": float(
                    residuals.abs().quantile(0.90).cpu().item()
                ),
                "flow_eval/fixed_bank_residual_p99": float(
                    residuals.abs().quantile(0.99).cpu().item()
                ),
                "flow_eval/fixed_bank_size": len(paths),
                "flow_eval/fixed_bank_subtb_lambda": float(subtb_lambda),
                "flow_eval/fixed_bank_subtb_max_span": (
                    0 if subtb_max_span is None else int(subtb_max_span)
                ),
            }
            coverage = fixed_bank_coverage(bank)
            for key, value in coverage.items():
                if isinstance(value, (int, float)):
                    metrics[f"flow_eval/coverage/{key}"] = value
            for action_type in ("coal", "recomb", "fixed_attachment", "terminal"):
                selected = [
                    row["residual"]
                    for row in records
                    if row["action_type"] == action_type
                ]
                tensor = (
                    torch.stack(selected)
                    if selected
                    else residuals.new_empty(0)
                )
                metrics.update(
                    _summary(tensor, f"flow_eval/action/{action_type}")
                )
            flat_actions = [action for path in actions_by_path for action in path]
            flat_components = [
                component for path in components_by_path for component in path
            ]
            flat_diagnostics = [
                diagnostic for path in diagnostics_by_path for diagnostic in path
            ]
            generated_indices = [
                index
                for index, action in enumerate(flat_actions)
                if action.get("event_type") != "fixed_attachment"
            ]
            recombination_indices = [
                index
                for index, action in enumerate(flat_actions)
                if action.get("event_type") == "recomb"
            ]
            metrics.update(
                _selected_component_summary(
                    [
                        torch.as_tensor(
                            flat_components[index]["atomic_action"],
                            device=generator.device,
                        )
                        for index in generated_indices
                    ],
                    "models/structural/fixed_bank",
                )
            )
            metrics.update(
                _selected_component_summary(
                    [
                        torch.as_tensor(
                            flat_components[index]["breakpoint"],
                            device=generator.device,
                        )
                        for index in recombination_indices
                    ],
                    "models/breakpoint/fixed_bank",
                )
            )
            metrics.update(
                _selected_component_summary(
                    [
                        torch.as_tensor(
                            flat_components[index]["time"],
                            device=generator.device,
                        )
                        for index in generated_indices
                    ],
                    "models/time/fixed_bank",
                    density=True,
                )
            )
            for model_name, diagnostic_name, indices in (
                (
                    "structural",
                    "structural_action_normalized_entropy",
                    generated_indices,
                ),
                (
                    "breakpoint",
                    "breakpoint_normalized_entropy",
                    recombination_indices,
                ),
            ):
                values = [
                    float(flat_diagnostics[index][diagnostic_name])
                    for index in indices
                    if diagnostic_name in flat_diagnostics[index]
                ]
                if values:
                    metrics[
                        f"models/{model_name}/fixed_bank/normalized_entropy_mean"
                    ] = float(np.mean(values))
            split_rows = [
                flat_diagnostics[index]
                for index in generated_indices
                if bool(
                    flat_diagnostics[index].get(
                        "recombination_split_bias_enabled",
                        False,
                    )
                )
            ]
            split_selected_rows = [
                row
                for row in split_rows
                if "recombination_split_selected_lineage_score" in row
            ]
            split_prefix = "models/recombination_split/fixed_bank"
            metrics[f"{split_prefix}/decision_count"] = int(len(split_rows))
            if split_rows:
                mass_errors = [
                    float(
                        row.get(
                            "recombination_split_mass_absolute_error",
                            0.0,
                        )
                    )
                    for row in split_rows
                ]
                metrics[f"{split_prefix}/mass_absolute_error_mean"] = float(
                    np.mean(mass_errors)
                )
                metrics[f"{split_prefix}/mass_absolute_error_max"] = float(
                    np.max(mass_errors)
                )
                for metric_name, diagnostic_name, reducer in (
                    ("candidate_score_min", "recombination_split_score_min", np.min),
                    ("candidate_score_mean", "recombination_split_score_mean", np.mean),
                    ("candidate_score_max", "recombination_split_score_max", np.max),
                ):
                    metrics[f"{split_prefix}/{metric_name}"] = float(
                        reducer(
                            [
                                float(row.get(diagnostic_name, 0.0))
                                for row in split_rows
                            ]
                        )
                    )
            if split_selected_rows:
                for metric_name, diagnostic_name in (
                    ("selected_lineage_score_mean", "recombination_split_selected_lineage_score"),
                    ("selected_breakpoint_score_mean", "recombination_split_selected_breakpoint_score"),
                    ("selected_atomic_logit_adjustment_mean", "recombination_split_selected_atomic_logit_adjustment"),
                ):
                    metrics[f"{split_prefix}/{metric_name}"] = float(
                        np.mean(
                            [
                                float(row.get(diagnostic_name, 0.0))
                                for row in split_selected_rows
                            ]
                        )
                    )
            cwr_rows = [
                flat_diagnostics[index]
                for index in generated_indices
                if bool(
                    flat_diagnostics[index].get(
                        "local_cwr_event_gate_enabled",
                        False,
                    )
                )
            ]
            cwr_prefix = "models/cwr_event_gate/fixed_bank"
            metrics[f"{cwr_prefix}/decision_count"] = int(len(cwr_rows))
            if cwr_rows:
                for metric_name, diagnostic_name, transform in (
                    (
                        "prior_recombination_probability_mean",
                        "local_cwr_prior_recombination_probability",
                        float,
                    ),
                    (
                        "policy_recombination_probability_mean",
                        "local_cwr_policy_recombination_probability",
                        float,
                    ),
                    (
                        "residual_mean",
                        "local_cwr_event_residual",
                        float,
                    ),
                    (
                        "residual_abs_mean",
                        "local_cwr_event_residual",
                        abs,
                    ),
                ):
                    metrics[f"{cwr_prefix}/{metric_name}"] = float(
                        np.mean(
                            [
                                transform(float(row.get(diagnostic_name, 0.0)))
                                for row in cwr_rows
                            ]
                        )
                    )
                metrics[f"{cwr_prefix}/residual_abs_max"] = float(
                    max(
                        abs(float(row.get("local_cwr_event_residual", 0.0)))
                        for row in cwr_rows
                    )
                )
                metrics[f"{cwr_prefix}/selected_recombination_rate"] = float(
                    np.mean(
                        [
                            row.get("local_cwr_selected_event") == "recombination"
                            for row in cwr_rows
                        ]
                    )
                )
            recombination_residuals = [
                row["residual"]
                for row in records
                if row["action_type"] == "recomb"
            ]
            metrics.update(
                _summary(
                    torch.stack(recombination_residuals)
                    if recombination_residuals
                    else residuals.new_empty(0),
                    (
                        "models/breakpoint/fixed_bank/"
                        "recombination_conditioned_residual"
                    ),
                )
            )
            for source in sorted(set(sources)):
                selected_indices = {
                    index for index, value in enumerate(sources) if value == source
                }
                selected = torch.stack(
                    [
                        row["residual"]
                        for row in records
                        if row["trajectory_index"] in selected_indices
                    ]
                )
                metrics.update(_summary(selected, f"flow_eval/source/{source}"))
            return metrics
    finally:
        generator.train(was_training)
