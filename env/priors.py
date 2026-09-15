"""ARG event prior calculations and sampling, independent of environment state updates."""

import math
from dataclasses import replace

from .actions import CoalescenceChoice, RecombinationChoice


def total_event_rate(rates, *, validate=True):
    """Sum event rates, requiring a valid waiting-time rate by default."""
    total_rate = float(rates["lambda_coal"] + rates["lambda_recomb"])
    if validate and (not math.isfinite(total_rate) or total_rate <= 0):
        raise ValueError("waiting-time rate must be finite and positive")
    return total_rate


def compute_event_rates(actions, *, rho, num_blocks):
    coal_actions, recomb_actions = actions

    lambda_coal = float(len(coal_actions))

    # In Hudson every link between the leftmost and rightmost ancestral
    # base is eligible, including links in trapped nonancestral gaps.
    # One pair has hazard 1 and one link has hazard 2 Ne r (in 2 Ne units).
    total_blocks = sum(choice.breakpoint_count for choice in recomb_actions)
    total_active_material_length = float(total_blocks) / float(num_blocks)
    lambda_recomb = rho / 2.0 * total_active_material_length

    return {
        "lambda_coal": lambda_coal,
        "lambda_recomb": lambda_recomb,
        "total_active_material_length": total_active_material_length,
    }


def compute_event_probabilities(rates):
    """Normalize coalescence and recombination rates into probabilities."""
    denom = total_event_rate(rates, validate=False)
    if denom <= 0:
        return {"coal": 0.0, "recomb": 0.0}
    return {
        "coal": rates["lambda_coal"] / denom,
        "recomb": rates["lambda_recomb"] / denom,
    }


def compute_cwr_event_log_prior(active_lineages, combined_actions, action, rates, *,
                                time_env, time_policy):
    """Score a validated timed action using its event and waiting-time prior."""
    coal_actions, recomb_actions = combined_actions

    total_rate = total_event_rate(rates)
    event_probs = compute_event_probabilities(rates)

    wait_log_prior = (time_env.log_density(action.delta_t, total_rate)
                      if time_policy == "cwr_exponential" else
                      time_env.time_action_log_probability(action.time_action, total_rate))

    if isinstance(action, CoalescenceChoice) and action.is_valid_for(
            active_lineages):
        action_log_prior = math.log(event_probs["coal"] / len(coal_actions))

    elif isinstance(action, RecombinationChoice) and RecombinationChoice.is_valid_for(action, active_lineages):
        canonical = next((choice for choice in recomb_actions
                          if choice.active_lineage_i == action.active_lineage_i), None)
        if (canonical is None or action.breakpoint not in range(canonical.span_start + 1, canonical.span_end + 1)
                or replace(action, breakpoint=None, time_action=None, delta_t=None) != canonical):
            raise ValueError('Invalid recombination lineage span or breakpoint')
        recomb_total_weight = sum(choice.breakpoint_count for choice in recomb_actions)
        action_log_prior = math.log(event_probs["recomb"] / recomb_total_weight)
    else:
        raise ValueError(f"Invalid action: {action}")

    score = action_log_prior + wait_log_prior
    if time_policy == "cwr_exponential" and not math.isfinite(score):
        raise ValueError("non-finite continuous event prior score")
    return score


def sample_event_type(event_probs, *, rng):
    """Sample the event type using a NumPy-compatible random generator."""
    event_types = ("coal", "recomb")
    index = rng.choice(2, p=[event_probs[event] for event in event_types])
    return event_types[index]


def sample_recombination_prior_action(recomb_actions, *, rng):
    """Sample a lineage weighted by its span, then a uniform breakpoint."""
    total_weight = sum(choice.breakpoint_count for choice in recomb_actions)
    if total_weight <= 0:
        raise ValueError("No valid recombination actions to sample")

    target = rng.random() * total_weight
    cumulative = 0
    selected = recomb_actions[-1]
    for choice in recomb_actions:
        cumulative += choice.breakpoint_count
        if target <= cumulative:
            selected = choice
            break

    if selected.breakpoint_count <= 0:
        raise ValueError("No valid recombination actions to sample")
    breakpoint = selected.span_start + 1 + rng.randrange(selected.breakpoint_count)
    return replace(selected, breakpoint=breakpoint)


def sample_prior_step(active_lineages, actions, rates, *, time_env, time_policy,
                      rng, event_rng):
    """Sample a complete timed action and return it with its log prior."""
    event_type = sample_event_type(compute_event_probabilities(rates), rng=event_rng)
    coal_actions, recomb_actions = actions
    if event_type == "coal":
        action = rng.choice(coal_actions)
    else:
        action = sample_recombination_prior_action(recomb_actions, rng=rng)

    rate = total_event_rate(rates)
    if time_policy == "cwr_exponential":
        action = replace(action, delta_t=time_env.sample_from_prior(rate, rng))
    else:
        action = replace(action, time_action=time_env.sample_action_from_prior(rate, rng))
    log_prior = compute_cwr_event_log_prior(
        active_lineages, actions, action, rates,
        time_env=time_env, time_policy=time_policy)
    return action, log_prior
