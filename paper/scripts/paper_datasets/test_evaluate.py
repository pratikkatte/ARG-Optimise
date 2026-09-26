"""Analytic and dense-reference checks for the paper metric runner."""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pytest
import tskit

from validation.scripts.evaluate_arginfer import Features, extract_features, topology_stats
from paper.scripts.paper_datasets.evaluate import time_metrics, topology_metrics


def feature(boundaries, times, clades):
    return Features(np.array(boundaries, dtype=float), np.array(times, dtype=float),
                    np.zeros(len(clades)), clades)


def test_time_metrics_analytic_spans_pairs_and_units():
    truth = feature([0, 2, 10], [[2, 6, 6], [4, 8, 8]], [(), ()])
    # Pair errors in generations are 1 and 2; aggregate RMSE is sqrt(2.5).
    draws = [feature([0, 2, 10], [[1, 6, 6], [3, 8, 8]], [(), ()]),
             feature([0, 2, 10], [[5, 10, 10], [7, 12, 12]], [(), ()])]
    result, pairs, _ = time_metrics(truth, draws, 2,
        ['pair_tmrca_rmse', 'tmrca_coverage', 'tmrca_interval_width'], chunk_size=1)
    assert result['pair_tmrca_rmse'] == pytest.approx(math.sqrt(2.5)/4)
    assert result['tmrca_coverage'] == pytest.approx(.5)
    assert result['tmrca_interval_width'] == pytest.approx(.9)
    np.testing.assert_allclose(pairs[:, 0], [1/16, 4/16])


def test_time_metrics_unequal_breakpoints_match_dense_and_chunk_invariant():
    truth = feature([0, 2, 10], [[2, 3], [5, 6]], [(), ()])
    draws = [feature([0, 5, 10], [[i+1, i+2], [i+3, i+4]], [(), ()]) for i in range(8)]
    expected = np.array([2, 5, 5])
    values = np.array([[i+1, i+1, i+3] for i in range(8)])
    weights = np.array([.2, .3, .5])
    low, high = np.quantile(values, [.05, .95], axis=0)
    for chunk in (1, 2, 64):
        result, _, intervals = time_metrics(truth, draws, .5,
            ['pair_tmrca_rmse', 'tmrca_coverage', 'tmrca_interval_width'], chunk_size=chunk)
        assert intervals == 3
        assert result['pair_tmrca_rmse'] == pytest.approx(np.sqrt(np.sum(weights*(values.mean(0)-expected)**2)))
        assert result['tmrca_coverage'] == pytest.approx(np.sum(weights*((low<=expected)&(expected<=high))))
        assert result['tmrca_interval_width'] == pytest.approx(np.sum(weights*(high-low)))


def test_topology_sweep_matches_brute_force_and_brier_denominators():
    a, b, c = (3, 12), (5, 10), (6, 9)
    truth = feature([0, 3, 10], [[1, 2], [1, 2]], [a, c])
    draws = [feature([0, 2, 10], [[1, 2], [1, 2]], [a, b]),
             feature([0, 7, 10], [[1, 2], [1, 2]], [b, c])]
    result = topology_metrics(truth, draws, 4)
    rf = brier = summed = 0
    for left, right in zip([0, 2, 3, 7], [2, 3, 7, 10]):
        actual = truth.topologies[truth.at(left)]
        sigs = [d.topologies[d.at(left)] for d in draws]
        reference = topology_stats(sigs, actual)
        union = set(actual).union(*map(set, sigs))
        weight = (right-left)/10
        rf += weight*reference['rf']
        brier += weight*reference['clade_brier']
        summed += weight*reference['clade_brier']*len(union)
    assert result['rooted_rf'] == pytest.approx(rf)
    assert result['clade_brier_observed_union'] == pytest.approx(brier)
    assert result['clade_brier_sum'] == pytest.approx(summed)
    assert result['clade_brier_fixed_universe'] == pytest.approx(summed/10)


def test_perfect_draws_and_metric_selection():
    truth = feature([0, 10], [[2, 3]], [(3,)])
    result, _, _ = time_metrics(truth, [truth], 1, ['tmrca_coverage'])
    assert result == {'tmrca_coverage': 1.0}
    result = topology_metrics(truth, [truth], 3)
    assert result['rooted_rf'] == result['clade_brier_observed_union'] == 0


def test_unary_scaffolding_and_sample_reordering_do_not_change_metrics():
    tables = tskit.TableCollection(10)
    tables.time_units = 'generations'
    for _ in range(3):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for time in (1, 2, 3):
        tables.nodes.add_row(time=time)
    for parent, child in ((3, 0), (3, 1), (4, 3), (4, 2), (5, 4)):
        tables.edges.add_row(0, 10, parent, child)
    tables.sort()
    raw = tables.tree_sequence()
    simple = raw.simplify(samples=[2, 0, 1])
    truth = extract_features(raw, [2, 0, 1])
    draw = extract_features(simple)
    assert truth.topologies == draw.topologies
    np.testing.assert_array_equal(truth.times, draw.times)
    assert truth.times[0, -1] == 2  # local MRCA, not unary ancestor at time 3
    assert topology_metrics(truth, [draw], 3)['rooted_rf'] == 0


@pytest.mark.parametrize('level', [0, 1, -1])
def test_invalid_credible_level(level):
    truth = feature([0, 10], [[2, 3]], [()])
    with pytest.raises(ValueError, match='Credible level'):
        time_metrics(truth, [truth], 1, ['tmrca_coverage'], level=level)


def test_draw_selection_preserves_order_and_checks_available_samples():
    from paper.scripts.paper_datasets.evaluate import select_draw_files
    files = [(i, Path(f'{i}.trees')) for i in (201000, 202000, 203000)]
    assert select_draw_files(files, 2, 'first') == files[:2]
    assert len(files) == 3
    assert select_draw_files(files, 3, 'all') == files
    with pytest.raises(ValueError, match='at least'):
        select_draw_files(files, 4, 'first')
    with pytest.raises(ValueError, match='Expected 2'):
        select_draw_files(files, 2, 'all')
    with pytest.raises(ValueError, match='Unknown'):
        select_draw_files(files, 2, 'random')
    with pytest.raises(ValueError, match='Invalid'):
        select_draw_files(files, 0, 'first')
