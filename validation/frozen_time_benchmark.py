#!/usr/bin/env python3
"""Frozen-structure diagnostic for the local Bernstein-beta time policy.

The benchmark replays sampled local histories with event types, lineage choices,
topology, and recombination breakpoints fixed.  For selected generated events it
changes only that event's time, keeps later generated event times fixed in
absolute time, and compares the resulting terminal local-score curve with one
or more learned time-policy densities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import infer
from env import CoalescenceChoice, FixedAttachmentChoice, RecombinationChoice
from tb_gfn import TBGFlowNetGenerator
from time_context import build_time_context


@dataclass
class Bundle:
    label: str
    checkpoint: Path
    env: Any
    generator: TBGFlowNetGenerator
    metadata: dict[str, Any]


def _load_bundle(label: str, checkpoint: Path, device: str) -> Bundle:
    data = infer.load_checkpoint(checkpoint, map_location="cpu")
    metadata = dict(data["metadata"])
    infer.validate_metadata(metadata)
    resolved = infer.resolve_device(device)
    base_env = infer.environment_from_metadata(
        metadata,
        seed=int(metadata["seed"]),
        device=resolved,
        dataset_path=metadata.get("dataset_path"),
    )
    env = infer.local_environment_from_metadata(metadata, base_env)
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=int(metadata["init_z_sample_count"]),
        cfg={
            "breakpoint_policy": metadata.get("breakpoint_policy", "continuous-bin"),
            "breakpoint_mixtures": int(metadata.get("breakpoint_mixtures", 4)),
        },
        device=resolved,
        verbose=False,
        log_z_lr=float(metadata.get("log_z_lr", 1e-3)),
        model_kwargs=dict(metadata.get("model", {})),
        initialize_z_from_prior=False,
        loss_mode=str(metadata.get("loss", "fl_subtb")),
        subtb_lambda=float(metadata.get("subtb_lambda", 0.9)),
    )
    generator.load(data, load_optimizer=False, map_location=resolved)
    generator.eval()
    return Bundle(label, checkpoint.resolve(), env, generator, metadata)


def _action(record: dict[str, Any]):
    event_type = record["event_type"]
    if event_type == "fixed_attachment":
        return FixedAttachmentChoice(event_time=float(record["event_time"]))
    if event_type == "coal":
        action = CoalescenceChoice.from_action(record)
    elif event_type == "recomb":
        action = RecombinationChoice.from_action(record)
    else:
        raise ValueError(f"unsupported event type {event_type!r}")
    if action is None:
        raise ValueError(f"invalid serialized action: {record}")
    return action


def _apply(env, state, action):
    if isinstance(action, FixedAttachmentChoice):
        log_prior = env.fixed_attachment_log_prior(state)
    else:
        log_prior = env.compute_cwr_event_log_prior(
            state, env.enumerate_actions(state), action
        )
    return env.apply_action(state, action, log_prior=log_prior)


def _base_replay(env, context_id: str, records: list[dict[str, Any]]):
    state = env.get_initial_state(context_id)
    states = []
    event_times = []
    for record in records:
        states.append(state)
        action = _action(record)
        if isinstance(action, FixedAttachmentChoice):
            event_time = float(action.event_time)
        else:
            event_time = float(state.current_time) + float(action.delta_time)
        event_times.append(event_time)
        state = _apply(env, state, action)
    if not state.is_done or state.log_reward is None:
        raise RuntimeError("manifest trajectory did not replay to a scored terminal state")
    return states, event_times, state


def _structural_key(action) -> tuple[Any, ...]:
    if isinstance(action, CoalescenceChoice):
        return ("coal", int(action.active_lineage_i), int(action.active_lineage_j))
    if isinstance(action, RecombinationChoice):
        return ("recomb", int(action.active_lineage_i))
    raise TypeError(type(action))


@torch.no_grad()
def _mixture_logits(bundle: Bundle, state, action):
    env = bundle.env
    generator = bundle.generator
    inputs = env.prepare_state_rollout_inputs([state])
    candidates = inputs["input_actions"][0]
    target_key = _structural_key(action)
    candidate_index = next(
        index
        for index, candidate in enumerate(candidates)
        if _structural_key(candidate) == target_key
    )
    lineage_reps, summary_reps, lineage_features, counts = generator._encode_states(
        [state]
    )
    _, action_features = generator.arg_model._score_candidates(
        [candidates],
        lineage_reps,
        summary_reps,
        state_contexts=(lineage_features if generator.arg_model.local_mode else None),
    )
    selected = action_features[:, candidate_index]
    rollout = inputs["rollout"][0]
    rate = float(rollout["total_rate"])
    max_delta = rollout["max_delta"]
    direct = generator.time_model.context_features(
        [rate], [max_delta], device=generator.device, dtype=selected.dtype
    )
    biological, _ = generator.arg_model.build_time_context(
        [state], [action], [max_delta], dtype=selected.dtype
    )
    logits = generator.time_model(torch.cat([selected, direct, biological], dim=-1))
    return logits, rate, max_delta


def _policy_log_u_density(bundle: Bundle, state, action, quantiles: np.ndarray):
    logits, _, _ = _mixture_logits(bundle, state, action)
    repeated = logits.expand(len(quantiles), -1)
    values = bundle.generator.time_model.log_quantile_density(
        repeated,
        torch.as_tensor(quantiles, dtype=torch.float64, device=bundle.generator.device),
    )
    return values.detach().cpu().numpy()


def _replay_with_time(
    env,
    target_state,
    records: list[dict[str, Any]],
    original_times: list[float],
    target_index: int,
    target_delta: float,
) -> float | None:
    state = target_state
    try:
        for index in range(target_index, len(records)):
            record = records[index]
            action = _action(record)
            if isinstance(action, FixedAttachmentChoice):
                action = FixedAttachmentChoice(event_time=float(record["event_time"]))
            else:
                delta = (
                    float(target_delta)
                    if index == target_index
                    else float(original_times[index]) - float(state.current_time)
                )
                if not math.isfinite(delta) or delta <= 0.0:
                    return None
                quantile = env.delta_to_time_quantile(state, delta)
                action = replace(
                    action,
                    delta_time=delta,
                    time_quantile=float(quantile),
                )
            state = _apply(env, state, action)
        if not state.is_done or state.log_reward is None:
            return None
        return float(state.log_reward)
    except (ValueError, RuntimeError, IndexError):
        return None


def _descriptor(env, state, action, max_delta) -> dict[str, Any]:
    context = build_time_context(
        state,
        action,
        env,
        max_delta=max_delta,
        mode="full",
        sampled_breakpoint=getattr(action, "breakpoint", None),
        device="cpu",
    )
    return dict(context.diagnostics)


def _select_events(candidates: list[dict[str, Any]], maximum: int):
    selected: list[dict[str, Any]] = []

    def add(rows, key, reverse=False):
        if rows:
            row = sorted(rows, key=lambda value: float(value.get(key, 0.0)), reverse=reverse)[0]
            if row not in selected:
                selected.append(row)

    coals = [row for row in candidates if row["event_type"] == "coal"]
    recombs = [row for row in candidates if row["event_type"] == "recomb"]
    add(coals, "pair_overlap_fraction")
    add(coals, "pair_overlap_fraction", True)
    add(recombs, "breakpoint_gap_fraction")
    add(recombs, "breakpoint_gap_fraction", True)
    add(candidates, "lineage_variant_count")
    add(candidates, "lineage_variant_count", True)
    bounded = [row for row in candidates if row.get("finite_upper_bound")]
    unbounded = [row for row in candidates if not row.get("finite_upper_bound")]
    add(bounded, "available_time_window")
    add(bounded, "available_time_window", True)
    add(unbounded, "lineage_variant_count", True)
    for row in candidates:
        if row not in selected:
            selected.append(row)
        if len(selected) >= maximum:
            break
    return selected[:maximum]


def _softmax(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.zeros_like(values, dtype=float)
    if not finite.any():
        return result
    shifted = values[finite] - np.max(values[finite])
    weights = np.exp(shifted)
    result[finite] = weights / weights.sum()
    return result


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    epsilon = 1e-300
    midpoint = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * np.log((left + epsilon) / (midpoint + epsilon)))
        + 0.5 * np.sum(right * np.log((right + epsilon) / (midpoint + epsilon)))
    )


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = [
        _load_bundle(label, checkpoint, args.device)
        for label, checkpoint in args.model
    ]
    reference = bundles[0]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    context_id = str(manifest["id"])
    trajectory_records = [
        row
        for row in manifest["trajectories"][: args.trajectories]
        if row.get("terminal") and not row.get("failure_diagnostics")
    ]
    candidates: list[dict[str, Any]] = []
    replay_cache: dict[int, tuple[Any, Any, Any]] = {}
    for trajectory in trajectory_records:
        records = list(trajectory["actions"])
        states, event_times, terminal = _base_replay(
            reference.env, context_id, records
        )
        replay_cache[int(trajectory["index"])] = (states, event_times, terminal)
        for index, (record, state) in enumerate(zip(records, states)):
            action = _action(record)
            if isinstance(action, FixedAttachmentChoice):
                continue
            rollout = reference.env.prepare_state_rollout_inputs([state])["rollout"][0]
            descriptor = _descriptor(
                reference.env, state, action, rollout["max_delta"]
            )
            candidates.append(
                {
                    **descriptor,
                    "trajectory_index": int(trajectory["index"]),
                    "action_index": index,
                    "event_type": record["event_type"],
                    "original_delta": float(record["delta_time"]),
                }
            )
    selected = _select_events(candidates, args.max_events)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for event_number, descriptor in enumerate(selected, start=1):
        trajectory = next(
            row
            for row in trajectory_records
            if int(row["index"]) == int(descriptor["trajectory_index"])
        )
        records = list(trajectory["actions"])
        states, original_times, _ = replay_cache[int(trajectory["index"])]
        index = int(descriptor["action_index"])
        state = states[index]
        action = _action(records[index])
        rollout = reference.env.prepare_state_rollout_inputs([state])["rollout"][0]
        rate = float(rollout["total_rate"])
        generated_mass = float(rollout["generated_prior_mass"])
        upper = rollout["max_delta"]
        next_time = original_times[index + 1] if index + 1 < len(original_times) else None
        order_upper = (
            None
            if next_time is None
            else float(next_time) - float(state.current_time)
        )
        effective_upper = upper
        if order_upper is not None:
            effective_upper = order_upper if effective_upper is None else min(float(effective_upper), order_upper)
        quantiles = np.linspace(args.quantile_min, args.quantile_max, args.grid_size)
        deltas = np.asarray(
            [
                reference.env.time_env.quantile_to_delta(
                    float(value), rate, max_delta=upper
                )
                for value in quantiles
            ]
        )
        valid = deltas > 0.0
        if effective_upper is not None:
            valid &= deltas < float(effective_upper) - 1e-12
        scores = np.full(len(deltas), np.nan)
        for grid_index in np.flatnonzero(valid):
            scores[grid_index] = _replay_with_time(
                reference.env,
                state,
                records,
                original_times,
                index,
                float(deltas[grid_index]),
            )
        valid &= np.isfinite(scores)
        if valid.sum() < 3:
            continue
        # Convert the terminal density in physical time to the same quantile
        # reference measure used by the Bernstein-beta policy.
        log_dt_du = (
            math.log(generated_mass)
            - math.log(rate)
            + rate * deltas
        )
        target = _softmax(np.where(valid, scores + log_dt_du, np.nan))
        event_id = f"event_{event_number:03d}"
        policy_by_label: dict[str, np.ndarray] = {}
        for bundle in bundles:
            policy_log = _policy_log_u_density(bundle, state, action, quantiles)
            policy = _softmax(np.where(valid, policy_log, np.nan))
            policy_by_label[bundle.label] = policy
            summaries.append(
                {
                    "event_id": event_id,
                    "model": bundle.label,
                    "event_type": descriptor["event_type"],
                    "trajectory_index": descriptor["trajectory_index"],
                    "action_index": index,
                    "finite_upper_bound": descriptor["finite_upper_bound"],
                    "lineage_variant_count": descriptor.get("lineage_variant_count", 0.0),
                    "pair_overlap_fraction": descriptor.get("pair_overlap_fraction", 0.0),
                    "breakpoint_gap_fraction": descriptor.get("breakpoint_gap_fraction", 0.0),
                    "js_divergence_to_score": _js_divergence(target, policy),
                    "mass_on_top_score_quartile": float(
                        policy[scores >= np.nanquantile(scores[valid], 0.75)].sum()
                    ),
                    "expected_centered_local_score": float(np.nansum(policy * scores)),
                }
            )
        for grid_index in np.flatnonzero(valid):
            row = {
                "event_id": event_id,
                "event_type": descriptor["event_type"],
                "trajectory_index": descriptor["trajectory_index"],
                "action_index": index,
                "quantile": quantiles[grid_index],
                "delta_time": deltas[grid_index],
                "event_time": float(state.current_time) + deltas[grid_index],
                "centered_local_score": scores[grid_index],
                "score_normalized_mass": target[grid_index],
            }
            for label, policy in policy_by_label.items():
                row[f"policy_mass_{label}"] = policy[grid_index]
            rows.append(row)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(quantiles[valid], target[valid], label="score-normalized target", lw=2)
        for label, policy in policy_by_label.items():
            ax.plot(quantiles[valid], policy[valid], label=label)
        ax.set_xlabel("CwR quantile u")
        ax.set_ylabel("Normalized grid mass")
        ax.set_title(
            f"{event_id}: {descriptor['event_type']}, variants={descriptor.get('lineage_variant_count', 0):.0f}"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{event_id}_density.png", dpi=180)
        plt.close(fig)
    frame = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    frame.to_csv(output_dir / "density_grid.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / "event_summary.tsv", sep="\t", index=False)
    aggregate = (
        summary.groupby("model", as_index=False)
        .agg(
            event_count=("event_id", "nunique"),
            mean_js_divergence=("js_divergence_to_score", "mean"),
            mean_top_quartile_mass=("mass_on_top_score_quartile", "mean"),
            mean_expected_score=("expected_centered_local_score", "mean"),
        )
        if not summary.empty
        else pd.DataFrame()
    )
    aggregate.to_csv(output_dir / "aggregate.tsv", sep="\t", index=False)
    report = {
        "status": "complete",
        "context_id": context_id,
        "manifest": str(args.manifest.resolve()),
        "grid_size": args.grid_size,
        "selected_event_count": int(summary["event_id"].nunique()) if not summary.empty else 0,
        "models": [
            {
                "label": bundle.label,
                "checkpoint": str(bundle.checkpoint),
                "time_context_mode": bundle.metadata.get("model", {}).get("time_context_mode", "baseline"),
            }
            for bundle in bundles
        ],
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def _model_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("models must be LABEL=CHECKPOINT")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("models must be LABEL=CHECKPOINT")
    return label, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=_model_argument, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--quantile-min", type=float, default=0.02)
    parser.add_argument("--quantile-max", type=float, default=0.98)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
