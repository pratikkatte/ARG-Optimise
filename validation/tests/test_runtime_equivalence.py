"""Measured CPU optimizations must retain proposal paths and topology identities."""
from unittest.mock import patch

import numpy as np
import pytest
import tskit

from env.env import SimpleTrajectory
from eval.posterior_summary import topology_signature
from training.trainer import sample_compatible_trajectories
from validation.tests.test_infinite_sites_environment import environment, assert_same


def reference_signature(tree, samples):
    indices = {int(node): i for i, node in enumerate(samples)}
    clades = set()
    for node in tree.nodes():
        leaves = list(tree.samples(node))
        if 1 < len(leaves) < len(samples):
            clades.add(sum(1 << indices[int(leaf)] for leaf in leaves))
    return tuple(sorted(clades))


@pytest.mark.parametrize('seed', range(16))
def test_topology_matches_reference_for_unary_forests_and_internal_samples(seed):
    rng = np.random.default_rng(seed)
    tables = tskit.TableCollection(1.)
    active = []
    for _ in range(8):
        tables.nodes.add_row(time=0)  # Noncontiguous sample identities.
        active.append(tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0))
    time = 1.
    while len(active) > (2 if seed % 3 == 0 else 1):
        rng.shuffle(active)
        children = [active.pop() for _ in range(min(2 + seed % 2, len(active)))]
        parent = tables.nodes.add_row(time=time,
            flags=tskit.NODE_IS_SAMPLE if seed % 4 == 0 else 0)
        time += 1
        for child in children:
            tables.edges.add_row(0, 1, parent, child)
        unary = tables.nodes.add_row(time=time)
        time += 1
        tables.edges.add_row(0, 1, unary, parent)
        active.append(unary)
    tables.sort()
    ts = tables.tree_sequence()
    samples = ts.samples()[::-1] if seed % 2 else ts.samples()
    assert topology_signature(ts.first(), samples) == reference_signature(ts.first(), samples)
    assert topology_signature(ts.first(), samples[:1]) == reference_signature(ts.first(), samples[:1])


def test_owned_compatible_proposals_preserve_actions_scores_and_rng():
    expected_env = environment(length=4, recombination_rate=.2)
    actual_env = environment(length=4, recombination_rate=.2)
    expected = []
    for _ in range(8):
        state, path = expected_env.get_initial_state(), SimpleTrajectory()
        while not state.is_done:
            step = expected_env.sample_compatible_step(state)
            state = expected_env.apply_action(state, step.action, step.log_prior)
            path.update(step.action, log_prior=step.log_prior, log_proposal=step.log_proposal,
                        log_reward=state.log_reward)
        expected.append(path)
    with patch.object(actual_env, 'apply_action', side_effect=AssertionError('Unexpected clone')):
        actual = sample_compatible_trajectories(actual_env, 8)
    for before, after in zip(expected, actual):
        assert before.__dict__ == after.__dict__
    assert expected_env.rng.getstate() == actual_env.rng.getstate()


def test_owned_step_still_rejects_a_wrong_supplied_prior_before_mutation():
    env = environment()
    state = env.get_initial_state()
    before = state.clone(copy_partials=True)
    step = env.sample_compatible_step(state)
    with pytest.raises(ValueError, match='supplied log_prior'):
        env.step_owned_state(state, step.action, step.log_prior + 1)
    assert_same(state, before)
