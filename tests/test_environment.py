"""Environment integration: independent scores, support, geometry, and exact replay."""
from collections import defaultdict, deque
from dataclasses import replace
from itertools import combinations
import math
from pathlib import Path

import msprime
import numpy as np
import pytest
import tskit

from env.actions import CoalescenceChoice, RecombinationChoice
from env.env import ARGReward, IncompatibleActionError, SimpleARGEnvironment
from env.infinite_sites import evaluate_infinite_sites
from env.snp_data import SNPData, load_snp_dataset
from env.states import DescendantSegments, MaterialSegments

REP0 = Path(__file__).resolve().parents[1] / 'paper/datasets/r1/rep0'


def data(g, positions, length=2):
    g = np.asarray(g, dtype=np.uint8)
    n, s = g.shape
    return SNPData(g, positions, length, tuple(range(s)), ('A',)*s, ('G',)*s,
                   tuple(f'h{i}' for i in range(n)))


def environment(g=None, positions=(0.25,), length=2, **kwargs):
    g = [[1], [1], [0]] if g is None else g
    defaults = dict(population_size=0.5, mutation_rate=0.1, recombination_rate=0.1)
    defaults.update(kwargs)
    return SimpleARGEnvironment(snp_data=data(g, positions, length), **defaults)


def coal(env, state, i=0, j=1, dt=1):
    return env.apply_action(state, CoalescenceChoice(i, j, delta_t=dt))


def split(env, state, i=0, breakpoint=1, dt=1):
    choice = next(a for a in env.enumerate_actions(state)[1] if a.active_lineage_i == i)
    return env.apply_action(state, replace(choice, breakpoint=breakpoint, delta_t=dt))


def assert_independent(env, state):
    result = env.evaluate_terminal(state)
    assert not result.zero_likelihood
    assert state.partial_log_likelihood == pytest.approx(result.log_likelihood, rel=0, abs=1e-9)
    np.testing.assert_allclose(state.completed_site_lengths * 2 * env.population_size,
                               result.compatible_branch_lengths, rtol=1e-12, atol=1e-10)
    assert state.exposure * 2 * env.population_size == pytest.approx(result.exposure, rel=1e-12)
    assert state.log_reward == pytest.approx(env.reward_fn.C + state.accumulated_log_prior
                                           + result.log_likelihood, rel=0, abs=1e-9)


def assert_same(a, b):
    assert a.current_time == b.current_time
    # Recovering waits by subtracting stored event times introduces float64 roundoff.
    assert a.accumulated_log_prior == pytest.approx(b.accumulated_log_prior, rel=0, abs=1e-9)
    assert a.exposure == pytest.approx(b.exposure, rel=1e-12)
    assert a.partial_log_likelihood == pytest.approx(b.partial_log_likelihood, abs=1e-10)
    np.testing.assert_allclose(a.completed_site_lengths, b.completed_site_lengths,
                               rtol=1e-12, atol=1e-12, equal_nan=True)
    assert [n.node_id for n in a.active_lineages] == [n.node_id for n in b.active_lineages]
    for x, y in zip(a.active_lineages, b.active_lineages):
        assert x.descendants == y.descendants
        assert x.material_segments == y.material_segments
        np.testing.assert_array_equal(x.snp_indices, y.snp_indices)
        np.testing.assert_allclose(x.messages, y.messages, rtol=1e-12, atol=1e-12)


def test_messages_mask_and_input_immutability():
    env = environment()
    start = env.get_initial_state()
    np.testing.assert_array_equal(start.active_lineages[0].messages, [[0, 1, 0]])
    assert len(env.enumerate_actions(start)[0]) == 3
    assert env.enumerate_policy_actions(start)[0] == [CoalescenceChoice(0, 1)]
    before = start.clone(copy_partials=True)
    with pytest.raises(IncompatibleActionError) as exc:
        coal(env, start, 0, 2)
    assert exc.value.site_ids == (0,)
    assert_same(start, before)
    assert start.active_lineages[0].parents == []
    with pytest.raises(ValueError):
        start.active_lineages[0].messages[0, 0] = 99
    middle = coal(env, start)
    # Both derived samples joined; the as-yet unbuilt shared stem can acquire the mutation.
    np.testing.assert_array_equal(middle.active_lineages[-1].messages, [[0, 1, 0]])
    assert middle.partial_log_likelihood == pytest.approx(-0.1 * 4)
    done = coal(env, middle)
    assert done.completed_site_lengths[0] == 1
    assert done.exposure == 10
    assert_independent(env, done)


def test_annotated_intervals_keep_local_identity():
    segments = DescendantSegments(((0, 1, 1), (1, 2, 2), (2, 3, 2)))
    assert segments.segments == ((0, 1, 1), (1, 3, 2))
    assert segments.material.segments == ((0, 3),)
    assert segments.at(1) == 2 and segments.at(3) == 0
    with pytest.raises(ValueError, match='disjoint'):
        DescendantSegments(((0, 2, 1), (1, 3, 2)))
    with pytest.raises(ValueError, match='disjoint'):
        segments.merge(segments)


def test_recombination_boundary_empty_parent_and_disjoint_merge():
    env = environment([[1], [0]], [1.0], length=2)
    start = env.get_initial_state()
    state = split(env, start)
    left, right = state.active_lineages[-2:]
    assert left.snp_indices.size == 0 and left.messages.shape == (0, 3)
    np.testing.assert_array_equal(right.snp_indices, [0])
    assert state.exposure == 2  # Child's 2 bp counted once, not twice.
    assert sum(n.material_count for n in state.active_lineages) == 4
    # Reunite disjoint material, preserving the eligible branch across unary edges.
    state = coal(env, state, 1, 2)
    assert state.active_lineages[-1].messages[0, 2] == 2
    state = coal(env, state)
    assert state.completed_site_lengths[0] == 3
    assert_independent(env, state)


def test_snp_free_dataset_and_invariant_parents():
    env = environment(np.empty((2, 0)), [], length=4)
    state = split(env, env.get_initial_state(), breakpoint=2)
    assert all(len(n.snp_indices) == 0 for n in state.active_lineages)
    state = coal(env, state, 1, 2)
    state = coal(env, state)
    assert state.partial_log_likelihood == pytest.approx(-env.kappa * 24)
    assert_independent(env, state)


def test_trapped_gap_remains_a_recombination_choice():
    env = environment(np.empty((2, 0)), [], length=4)
    state = split(env, env.get_initial_state(), 0, 1)
    state = split(env, state, 2, 3)
    state = coal(env, state, 1, 3)  # [0,1) united with [3,4).
    lineage = state.active_lineages[-1]
    assert lineage.material_segments.segments == ((0, 1), (3, 4))
    choice = next(a for a in env.enumerate_policy_actions(state)[1]
                  if a.active_lineage_i == len(state.active_lineages)-1)
    assert choice.breakpoint_count == 3
    state = env.apply_action(state, replace(choice, breakpoint=2, delta_t=1))
    assert state.active_lineages[-2].material_segments.segments == ((0, 1),)
    assert state.active_lineages[-1].material_segments.segments == ((3, 4),)


def test_local_completion_stems_and_full_genome_termination():
    env = environment([[1], [0]], [0.25], length=2)
    state = split(env, env.get_initial_state(), 0)
    state = split(env, state, 0)
    # Active: A-left, A-right, B-left, B-right.
    state = coal(env, state, 0, 2)
    assert not state.is_done and not np.isnan(state.completed_site_lengths[0])
    exposure = state.exposure
    # Merge completed left root with unresolved right A, then right B.
    state = coal(env, state, 0, 2)
    assert state.exposure == exposure + 3  # A-right extends from time 1 to 4; the root stem is excluded.
    state = coal(env, state)
    assert_independent(env, state)
    assert state.is_done
    # A hole offset by double coverage cannot qualify as terminal.
    broken = env.get_initial_state()
    for n in broken.active_lineages:
        n.material_segments = MaterialSegments(((0, 1),))
    broken.total_active_blocks = 2
    assert not env.is_terminal(broken)


def test_prior_is_unmasked_and_diagnostic_density_is_separate():
    env = environment(recombination_rate=0)
    state = env.get_initial_state()
    action = CoalescenceChoice(0, 1, delta_t=0.2)
    assert env.enumerate_prior_options(state).rates['lambda_coal'] == 3
    masked = env.enumerate_policy_actions(state)
    assert env.compute_cwr_event_log_prior(state, masked, action,
                                          rates={'lambda_coal': 1, 'lambda_recomb': 0}) == pytest.approx(-0.6)
    step = env.sample_compatible_step(state)
    assert step.log_proposal - step.log_prior == pytest.approx(math.log(3))
    assert step.log_prior == pytest.approx(-3 * step.action.delta_t)
    with pytest.raises(ValueError, match='unmasked'):
        env.apply_action(state, action, log_prior=-0.2)


@pytest.mark.parametrize('changes', [dict(sequences=['A','G']), dict(bp_per_blocks=2),
                                     dict(device='cuda'), dict(num_blocks=95), dict(rho=99)])
def test_rejected_interfaces(changes):
    with pytest.raises(ValueError):
        environment(**changes)


def test_geometry_support_and_zero_mutation():
    with pytest.raises(ValueError, match='inseparable'):
        environment([[1,1],[1,0],[0,1]], [0.1,0.2])
    with pytest.raises(ValueError, match='inseparable'):
        environment([[1,1],[1,0],[0,1]], [0.1,1.2], recombination_rate=0)
    env = environment([[1,1],[1,0],[0,1]], [0.1,1.2])
    assert not env.enumerate_policy_actions(env.get_initial_state())[0]
    assert env.enumerate_policy_actions(env.get_initial_state())[1]
    with pytest.raises(ValueError, match='zero support'):
        environment(mutation_rate=0)
    zero = environment(np.empty((2,0)), [], mutation_rate=0, recombination_rate=0)
    assert coal(zero, zero.get_initial_state()).partial_log_likelihood == 0


def test_true_zero_and_numerical_failures_have_different_semantics():
    env = environment()
    state = coal(env, coal(env, env.get_initial_state()))
    state.completed_site_lengths[0] = 0
    assert env.compute_terminal_log_reward(state) == -math.inf
    assert env.compute_terminal_log_reward(state, log_likelihood=-math.inf) == -math.inf
    state.completed_site_lengths[0] = math.inf
    with pytest.raises(FloatingPointError):
        env.compute_terminal_log_reward(state)
    with pytest.raises(FloatingPointError):
        ARGReward()(math.nan, 0)


def test_clone_restore_replay_and_batch():
    env = environment([[1], [0]], [1.25])
    initial = env.get_initial_state()
    middle = split(env, initial)
    done = coal(env, coal(env, middle, 1, 2))
    for state in (initial, middle, done):
        assert_same(state, env.replay(state.actions))
        broken = state.clone()
        broken.exposure = 999
        broken.accumulated_log_prior = 999
        broken.current_time = 999
        broken.actions = ()
        broken.partial_log_likelihood = None
        broken.completed_site_lengths[:] = np.nan
        for n in broken.all_nodes.values():
            n.messages = n.snp_indices = n.descendants = None
        assert_same(state, env.restore_state(broken))
    action = CoalescenceChoice(0,1,delta_t=1)
    batch = env.apply_actions([initial,initial], [action,action])
    assert_same(batch[0], batch[1])
    other = environment([[0],[1]],[1.25])
    with pytest.raises(ValueError, match='different'):
        other.restore_state(done)
    with pytest.raises(RuntimeError, match='exceeded'):
        environment().sample_compatible_trajectory(max_events=1)


def ranked_histories(k):
    if k == 1:
        yield ()
    else:
        for pair in combinations(range(k),2):
            for suffix in ranked_histories(k-1):
                yield (pair,)+suffix


def test_exhaustive_single_site_masks_against_independent_terminal_trees():
    n = 4
    for pattern in range(1, (1<<n)-1):
        obs = data([[(pattern>>i)&1] for i in range(n)], [0.25], 1)
        env = SimpleARGEnvironment(snp_data=obs, population_size=0.5, mutation_rate=0.1,
                                  recombination_rate=0)
        for history in ranked_histories(n):
            table = tskit.TableCollection(1); table.time_units = 'generations'
            active = [table.nodes.add_row(flags=1,time=0) for _ in range(n)]
            state = env.get_initial_state(); accepted = True
            for time,(i,j) in enumerate(history,1):
                parent = table.nodes.add_row(time=time)
                for idx in (i,j):
                    table.edges.add_row(0,1,parent,active[idx])
                active = [a for k,a in enumerate(active) if k not in (i,j)]+[parent]
                if accepted:
                    try:
                        state = coal(env,state,i,j)
                    except IncompatibleActionError:
                        accepted = False
            table.sort()
            reference = evaluate_infinite_sites(table.tree_sequence(),obs,mutation_rate=0.1)
            assert accepted == (not reference.zero_likelihood)
            if accepted:
                assert state.partial_log_likelihood == pytest.approx(reference.log_likelihood, abs=1e-12)


@pytest.mark.parametrize('targets', [(3,5),(0,3),(0,0)])
def test_two_interval_reachability_including_invariant_material(targets):
    n = 3
    present = [i for i,t in enumerate(targets) if t]
    obs = np.array([[(targets[j]>>i)&1 for j in present] for i in range(n)], dtype=np.uint8)
    env = environment(obs,[j+0.25 for j in present])
    def key(state):
        return tuple(sorted((x.descendants.at(0.25),x.descendants.at(1.25)) for x in state.active_lineages))
    initial = env.get_initial_state()
    queue = deque([initial]); seen = {key(initial)}; reverse = defaultdict(set); terminal = set()
    recomb_only = 0
    while queue:
        state = queue.popleft(); current = key(state)
        if state.is_done:
            terminal.add(current); continue
        coal_actions,recomb = env.enumerate_policy_actions(state)
        recomb_only += not coal_actions and bool(recomb)
        actions = [replace(a,delta_t=1) for a in coal_actions]
        actions += [replace(a,breakpoint=1,delta_t=1) for a in recomb]
        for action in actions:
            child = env.apply_action(state,action); child_key = key(child)
            reverse[child_key].add(current)
            if child_key not in seen:
                seen.add(child_key); queue.append(child)
    reachable = set(terminal); queue = deque(terminal)
    while queue:
        for parent in reverse[queue.popleft()]:
            if parent not in reachable:
                reachable.add(parent); queue.append(parent)
    assert seen == reachable
    if targets == (3,5):
        assert recomb_only > 0


def truth_replay(env, ts):
    """Validation only: translate full recorded ancestry into chronological events."""
    state = env.get_initial_state()
    source_to_env = dict(zip(map(int,ts.samples()),range(env.num_sequences)))
    children = defaultdict(set); spans = defaultdict(list)
    for edge in ts.edges():
        children[edge.parent].add(edge.child)
        spans[edge.parent].append((edge.left,edge.right))
    times = defaultdict(list)
    for node in ts.nodes():
        if node.id not in source_to_env:
            times[node.time].append(node.id)
    snapshots = []
    for time,ids in sorted(times.items()):
        active = {node.node_id:i for i,node in enumerate(state.active_lineages)}
        dt = time/(2*env.population_size)-state.current_time
        if len(ids)==2:
            assert children[ids[0]] == children[ids[1]] and len(children[ids[0]])==1
            source_child = next(iter(children[ids[0]]))
            ordered = sorted(ids,key=lambda i:min(l for l,r in spans[i]))
            boundary = max(r for l,r in spans[ordered[0]])
            assert int(boundary)==boundary
            choice = next(a for a in env.enumerate_actions(state)[1]
                          if a.active_lineage_i==active[source_to_env[source_child]])
            state = env.apply_action(state,replace(choice,breakpoint=int(boundary),delta_t=dt))
            for original,parent in zip(ordered,state.active_lineages[-2:]):
                source_to_env[original]=parent.node_id
        else:
            assert len(ids)==1 and len(children[ids[0]])==2
            i,j = sorted(active[source_to_env[c]] for c in children[ids[0]])
            state = coal(env,state,i,j,dt)
            source_to_env[ids[0]]=state.active_lineages[-1].node_id
        snapshots.append(state)
    return state,snapshots


def r1_environment(seed=None):
    return SimpleARGEnvironment(snp_data=load_snp_dataset(REP0), population_size=10000,
                                mutation_rate=1e-8, recombination_rate=1e-8, seed=seed)


def test_r1_full_ancestry_replay_and_cache_reconstruction():
    env = r1_environment()
    state,snapshots = truth_replay(env,tskit.load(REP0/'r1.full.trees'))
    reference = msprime.log_mutation_likelihood(tskit.load(REP0/'r1.trees'), mutation_rate=1e-8)
    assert state.is_done
    assert state.partial_log_likelihood == pytest.approx(reference,rel=0,abs=1e-9)
    assert_independent(env,state)
    for prefix in snapshots:
        assert_same(prefix,env.restore_state(prefix))
    assert_same(state,env.replay(state.actions))


@pytest.mark.parametrize('seed',[1,7,11])
def test_r1_generated_candidates(seed):
    env = r1_environment(seed)
    state,trajectory = env.sample_compatible_trajectory()
    assert len(trajectory)>0
    assert_independent(env,state)
    assert_same(state,env.restore_state(state))


def test_neural_entrypoints_reject_legacy_inputs(tmp_path):
    from generator import GFlowNetGenerator
    from train import train
    from infer import environment_from_metadata
    from eval.eval import run_evaluation
    model = GFlowNetGenerator(environment(), init_z_sample_count=1, initialize_z_from_policy=False)
    assert model.env.mutation_model == 'infinite_sites'
    calls = [lambda:train('missing.fa',str(tmp_path/'never-created'),'cpu'),
             lambda:environment_from_metadata({},7), lambda:run_evaluation({})]
    for call in calls:
        with pytest.raises(ValueError):
            call()
    assert not (tmp_path/'never-created').exists()


def test_invalid_actions_do_not_change_state():
    env = environment([[1],[0]], [0.25])
    state = env.get_initial_state()
    choice = env.enumerate_actions(state)[1][0]
    invalid = [CoalescenceChoice(-1,1,delta_t=1), CoalescenceChoice(0,0,delta_t=1),
               CoalescenceChoice(0,1,delta_t=math.nan), CoalescenceChoice(0,1,delta_t=0),
               replace(choice,breakpoint=0,delta_t=1), replace(choice,breakpoint=0.5,delta_t=1),
               replace(choice,material_count=99,breakpoint=1,delta_t=1)]
    for action in invalid:
        with pytest.raises(ValueError):
            env.apply_action(state,action)
        assert_same(state,env.get_initial_state())
    terminal=coal(env,state)
    assert env.enumerate_policy_actions(terminal)==([],[])
    with pytest.raises(ValueError,match='terminal'):
        coal(env,terminal)
