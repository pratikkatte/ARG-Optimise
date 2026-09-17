"""Independent likelihood controls, geometry, units, and reference comparisons."""
import math
from pathlib import Path

import msprime
import numpy as np
import pytest
import tskit

from env.infinite_sites import evaluate_infinite_sites
from env.snp_data import SNPData, load_snp_dataset


REP0 = Path(__file__).resolve().parents[1] / "datasets/sim_5k_mr20/rep0"


def observations(genotypes, positions, length=10):
    genotypes = np.asarray(genotypes)
    n, s = genotypes.shape
    return SNPData(genotypes, positions, length, tuple(range(s)), ("A",) * s, ("G",) * s,
                   tuple(f"h{i}" for i in range(n)))


def tables_with_samples(n, length=10):
    tables = tskit.TableCollection(length)
    tables.time_units = "generations"
    for _ in range(n):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    return tables


def two_tip(length=10, time=2, edge_end=None):
    tables = tables_with_samples(2, length)
    root = tables.nodes.add_row(time=time)
    for sample in range(2):
        tables.edges.add_row(0, length if edge_end is None else edge_end, root, sample)
    tables.sort()
    return tables.tree_sequence()


def quartet(unary=False, stem=False):
    tables = tables_with_samples(4)
    for time in (1, 1, 2):
        tables.nodes.add_row(time=time)
    for parent, child in ((4, 0), (4, 1), (5, 2), (5, 3), (6, 5)):
        tables.edges.add_row(0, 10, parent, child)
    if unary:
        node = tables.nodes.add_row(time=1.5)
        tables.edges.add_row(0, 10, node, 4)
        tables.edges.add_row(0, 10, 6, node)
    else:
        tables.edges.add_row(0, 10, 6, 4)
    if stem:
        root = tables.nodes.add_row(time=20)
        tables.edges.add_row(0, 10, root, 6)
    tables.sort()
    return tables.tree_sequence()


def test_two_tip_analytical_score_and_zero_coordinate():
    data = observations([[1], [0]], [0])
    result = evaluate_infinite_sites(two_tip(), data, mutation_rate=0.1)
    assert result.exposure == 40
    np.testing.assert_array_equal(result.compatible_branch_lengths, [2])
    assert result.log_likelihood == pytest.approx(-4 + math.log(0.2), abs=1e-12)
    assert result.incompatible_site_ids == () and not result.zero_likelihood


@pytest.mark.parametrize("unary,stem", [(False, False), (True, False), (False, True), (True, True)])
def test_unary_subdivisions_and_stems_preserve_likelihood(unary, stem):
    data = observations([[1], [1], [0], [0]], [2.5])
    result = evaluate_infinite_sites(quartet(unary, stem), data, mutation_rate=0.1)
    assert result.exposure == 60
    np.testing.assert_array_equal(result.compatible_branch_lengths, [1])
    assert result.log_likelihood == pytest.approx(-6 + math.log(0.1), abs=1e-12)


def test_incompatible_quartet_is_exactly_zero():
    data = observations([[1], [0], [1], [0]], [2.5])
    result = evaluate_infinite_sites(quartet(), data, mutation_rate=0.1)
    assert result.log_likelihood == -math.inf and result.zero_likelihood
    assert result.incompatible_site_ids == (0,)
    np.testing.assert_array_equal(result.compatible_branch_lengths, [0])


def test_recombination_half_open_boundaries_and_snp_free_exposure():
    tables = tables_with_samples(4, 15)
    for left, right, pairs, scale in ((0, 5, ((0, 1), (2, 3)), 1),
                                     (5, 10, ((0, 2), (1, 3)), 1),
                                     (10, 15, ((0, 1), (2, 3)), 3)):
        parents = [tables.nodes.add_row(time=scale) for _ in range(2)]
        root = tables.nodes.add_row(time=2 * scale)
        for parent, children in zip(parents, pairs):
            for child in children:
                tables.edges.add_row(left, right, parent, child)
            tables.edges.add_row(left, right, root, parent)
    tables.sort()
    ts = tables.tree_sequence()
    data = observations([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1]], [4.999, 5, 9.9], 15)
    result = evaluate_infinite_sites(ts, data, mutation_rate=0.1)
    assert result.exposure == 150  # Last 5 bp have no SNPs but contribute 90.
    np.testing.assert_array_equal(result.compatible_branch_lengths, [1, 1, 1])
    assert result.log_likelihood == pytest.approx(-15 + 3 * math.log(0.1), abs=1e-12)


@pytest.mark.parametrize("rate,sites,expected", [(0, False, 0), (0.1, False, -4), (0, True, -math.inf)])
def test_zero_snps_and_zero_rate(rate, sites, expected):
    data = observations([[1], [0]], [0.1]) if sites else observations(np.empty((2, 0)), [])
    result = evaluate_infinite_sites(two_tip(), data, mutation_rate=rate)
    assert result.log_likelihood == expected
    assert result.incompatible_site_ids == ()
    assert result.zero_likelihood == (expected == -math.inf)


def test_product_underflow_does_not_turn_compatible_score_into_zero():
    data = observations([[1], [0]], [0.1], 1)
    result = evaluate_infinite_sites(two_tip(length=1, time=1e-200), data, mutation_rate=1e-200)
    assert math.isfinite(result.log_likelihood)
    assert result.log_likelihood == pytest.approx(2 * math.log(1e-200), abs=1e-12)


def test_candidate_mutation_records_have_no_effect():
    data = observations([[1], [1], [0], [0]], [2.5])
    ts = quartet()
    expected = evaluate_infinite_sites(ts, data, mutation_rate=0.1)
    tables = ts.dump_tables()
    # Both the placement and labels deliberately disagree with observations.
    site = tables.sites.add_row(2.5, "T")
    tables.mutations.add_row(site=site, node=2, derived_state="C")
    site = tables.sites.add_row(3.5, "G")
    tables.mutations.add_row(site=site, node=6, derived_state="A")
    altered = evaluate_infinite_sites(tables.tree_sequence(), data, mutation_rate=0.1)
    tables.mutations.clear()
    tables.sites.clear()
    removed = evaluate_infinite_sites(tables.tree_sequence(), data, mutation_rate=0.1)
    assert altered.log_likelihood == removed.log_likelihood == expected.log_likelihood
    np.testing.assert_array_equal(altered.compatible_branch_lengths, expected.compatible_branch_lengths)


def test_explicit_mapping_preserves_score_after_sample_renumbering():
    ts = quartet()
    data = observations([[1], [1], [0], [0]], [2.5])
    permuted, mapping = ts.simplify([0, 2, 1, 3], map_nodes=True)
    assert evaluate_infinite_sites(permuted, data, mutation_rate=0.1).zero_likelihood
    correct = evaluate_infinite_sites(permuted, data, mutation_rate=0.1, sample_nodes=mapping[ts.samples()])
    original = evaluate_infinite_sites(ts, data, mutation_rate=0.1)
    assert correct.log_likelihood == original.log_likelihood


@pytest.mark.parametrize("mapping", [[0, 0], [0], [0, 2], [0, 1.0], [True, 0], 1])
def test_invalid_sample_mapping(mapping):
    with pytest.raises(ValueError, match="sample_nodes"):
        evaluate_infinite_sites(two_tip(), observations([[1], [0]], [0.1]), mutation_rate=0.1, sample_nodes=mapping)


@pytest.mark.parametrize("rate", [-1, math.inf, math.nan, True, "bad"])
def test_invalid_mutation_rate(rate):
    with pytest.raises(ValueError, match="mutation_rate"):
        evaluate_infinite_sites(two_tip(), observations([[1], [0]], [0.1]), mutation_rate=rate)


@pytest.mark.parametrize("units", ["unknown", "coalescent"])
def test_unknown_or_unconverted_units_are_rejected(units):
    tables = two_tip().dump_tables()
    tables.time_units = units
    with pytest.raises(ValueError, match="time_units"):
        evaluate_infinite_sites(tables.tree_sequence(), observations([[1], [0]], [0.1]), mutation_rate=0.1)


def test_noncontemporaneous_samples_are_rejected():
    tables = two_tip().dump_tables()
    times = tables.nodes.time
    times[0] = 1
    tables.nodes.time = times
    with pytest.raises(ValueError, match="contemporaneous"):
        evaluate_infinite_sites(tables.tree_sequence(), observations([[1], [0]], [0.1]), mutation_rate=0.1)


def test_span_and_sample_counts_must_match():
    with pytest.raises(ValueError, match="sequence_length"):
        evaluate_infinite_sites(two_tip(), observations([[1], [0]], [0.1], 11), mutation_rate=0.1)
    with pytest.raises(ValueError, match="sample count"):
        evaluate_infinite_sites(quartet(), observations([[1], [0]], [0.1]), mutation_rate=0.1)


def test_incomplete_ancestry_is_rejected_even_in_snp_free_interval():
    data = observations([[1], [0]], [0.1])
    with pytest.raises(ValueError, match="incomplete ancestry"):
        evaluate_infinite_sites(two_tip(edge_end=5), data, mutation_rate=0.1)


@pytest.mark.parametrize("seed", [7, 17, 27])
def test_simulated_fixtures_agree_with_msprime(seed):
    full = msprime.sim_ancestry(samples=3, population_size=10, sequence_length=20,
                               recombination_rate=0.01, record_full_arg=True, random_seed=seed)
    mutated = msprime.sim_mutations(full.simplify(), rate=0.01, discrete_genome=False,
                                   model=msprime.BinaryMutationModel(), random_seed=seed + 1)
    data = observations(mutated.genotype_matrix().T, mutated.tables.sites.position, 20)
    reference = msprime.log_mutation_likelihood(mutated, mutation_rate=0.01)
    for candidate in (full, mutated):
        result = evaluate_infinite_sites(candidate, data, mutation_rate=0.01)
        assert result.log_likelihood == pytest.approx(reference, rel=0, abs=1e-9)


def test_environment_export_applies_population_size_conversion_once():
    from env.actions import CoalescenceChoice
    from env.env import SimpleARGEnvironment

    data = observations([[0], [1]], [0.2], 2)
    env = SimpleARGEnvironment(snp_data=data, population_size=10,
                               mutation_rate=0.01, recombination_rate=0, device="cpu")
    state = env.get_initial_state()
    terminal = env.apply_coalescence(state, CoalescenceChoice(0, 1, delta_t=0.5))
    candidate = env.save_to_tree_sequence(terminal)
    data = observations([[0], [1]], [0.2], 2)
    result = evaluate_infinite_sites(candidate, data, mutation_rate=0.01)
    assert candidate.time_units == "generations" and candidate.first().time(candidate.first().root) == 10
    assert result.exposure == 40 and result.compatible_branch_lengths[0] == 10
    assert result.log_likelihood == pytest.approx(-0.4 + math.log(0.1), abs=1e-12)


@pytest.mark.skipif(not REP0.is_dir(), reason="local rep0 dataset is not installed")
def test_rep0_simplified_and_full_ancestry_regression():
    data = load_snp_dataset(REP0)
    simple = tskit.load(REP0 / "sim_5k_mr20.trees")
    reference = msprime.log_mutation_likelihood(simple, mutation_rate=2.5e-8)
    for filename in ("sim_5k_mr20.trees", "sim_5k_mr20.full.trees"):
        result = evaluate_infinite_sites(tskit.load(REP0 / filename), data, mutation_rate=2.5e-8)
        assert result.log_likelihood == pytest.approx(-733.476814024274, rel=0, abs=1e-9)
        assert result.log_likelihood == pytest.approx(reference, rel=0, abs=1e-9)
        assert result.exposure == pytest.approx(3432347945.6410666, rel=1e-14)
        assert not result.zero_likelihood and result.incompatible_site_ids == ()
