import numpy as np
import pytest
import tskit

from arg.new_rl.normal_ts_edge_closed_regions import (
    normal_edge_components_at_cut,
    scan_normal_ts_edge_closed_regions,
)


SOURCE_25KB = "arg/validation/output/tsinfer/l25kb_dated.trees"
SOURCE_1MB = "arg/validation/output/tsinfer/l1mb_dated.trees"


def _disjoint_half_open_tree_sequence():
    tables = tskit.TableCollection(sequence_length=10.0)
    child0 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    child1 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    parent0 = tables.nodes.add_row(time=5.0)
    parent1 = tables.nodes.add_row(time=5.0)
    tables.edges.add_row(2.0, 8.0, parent=parent0, child=child0)
    tables.edges.add_row(0.0, 2.0, parent=parent1, child=child1)
    tables.sort()
    return tables.tree_sequence()


def _overlapping_tree_sequence():
    tables = tskit.TableCollection(sequence_length=10.0)
    child0 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    child1 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    parent0 = tables.nodes.add_row(time=5.0)
    parent1 = tables.nodes.add_row(time=5.0)
    tables.edges.add_row(2.0, 8.0, parent=parent0, child=child0)
    tables.edges.add_row(0.0, 3.0, parent=parent1, child=child1)
    tables.sort()
    return tables.tree_sequence()


def test_direct_scan_accepts_adjacent_half_open_edge_regions():
    ts = _disjoint_half_open_tree_sequence()
    cut = normal_edge_components_at_cut(ts, 0.0, cut_index=0)
    closed = [
        component
        for component in cut["components"]
        if component["normal_edge_closed"]
    ]

    assert {(component["left"], component["right"]) for component in closed} == {
        (0.0, 2.0),
        (2.0, 8.0),
    }
    assert all(not component["outside_older_edge_overlap"] for component in closed)
    assert all(component["frontier_edge_count"] == 1 for component in closed)


def test_direct_scan_rejects_an_outside_edge_with_positive_overlap():
    cut = normal_edge_components_at_cut(_overlapping_tree_sequence(), 0.0)

    assert not any(
        component["normal_edge_closed"] for component in cut["components"]
    )
    assert all(
        "outside_older_edge_overlap" in component["rejection_reasons"]
        for component in cut["components"]
    )


def test_tied_parent_time_is_one_normal_cut_batch():
    result = scan_normal_ts_edge_closed_regions(_disjoint_half_open_tree_sequence())

    assert [item["cut_time"] for item in result.per_cut_summary] == [0.0, 5.0]
    assert len(result.regions) == 2
    assert all(region["valid_normal_cut_indices"] == (0,) for region in result.regions)


def test_25kb_direct_normal_edge_regions_are_structurally_closed():
    result = scan_normal_ts_edge_closed_regions(
        SOURCE_25KB,
        retain_per_cut_catalog=True,
    )

    assert len(result.regions) == 2
    assert {(region["left"], region["right"]) for region in result.regions} == {
        (386.0, 23963.0),
        (3440.0, 9543.0),
    }
    assert result.raw_closed_component_count == 9
    assert result.per_cut_component_catalog is not None
    for region in result.regions:
        assert region["contiguous"]
        assert region["proper_subregion"]
        assert region["frontier_edge_count"] > 0
        assert not region["outside_older_edge_overlap"]
        assert np.all(
            np.asarray(region["intervals"])[:, 0] < np.asarray(region["intervals"])[:, 1]
        )


@pytest.mark.slow
def test_1mb_direct_normal_edge_regions_are_stable():
    result = scan_normal_ts_edge_closed_regions(SOURCE_1MB)

    assert len(result.per_cut_summary) == 473
    assert result.raw_closed_component_count == 1335
    assert len(result.regions) == 32
    assert all(region["normal_edge_closed"] for region in result.regions)
    assert all(region["contiguous"] for region in result.regions)
    assert all(region["proper_subregion"] for region in result.regions)
