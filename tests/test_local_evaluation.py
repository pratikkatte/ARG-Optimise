from types import SimpleNamespace

import pytest

from refinement.evaluation import (
    LocalARGScore,
    _canonicalize_exterior_recombination,
    compare_scores,
    select_best_candidate_record,
)
from env import MaterialSegments
from validation.local_refinement_report import build_local_refinement_report


def score(*, likelihood, prior, valid=True, digest="x"):
    return LocalARGScore(
        whole_log_likelihood=float(likelihood),
        local_log_likelihood=float(likelihood),
        log_prior=float(prior),
        log_posterior=float(likelihood + prior),
        topology_digest=digest,
        splice_valid=valid,
    )


def test_comparison_reports_components_and_uses_strict_margin():
    source = score(likelihood=-10, prior=-2)
    candidate = score(likelihood=-8, prior=-3)
    comparison = compare_scores(source, candidate, margin=1e-6)

    assert comparison.likelihood_delta == pytest.approx(2.0)
    assert comparison.prior_delta == pytest.approx(-1.0)
    assert comparison.posterior_delta == pytest.approx(1.0)
    assert comparison.improves

    tied = compare_scores(source, score(likelihood=-9, prior=-3), margin=0.0)
    assert tied.posterior_delta == pytest.approx(0.0)
    assert not tied.improves


def test_invalid_candidate_never_improves():
    comparison = compare_scores(
        score(likelihood=-10, prior=-2),
        score(likelihood=10, prior=2, valid=False),
    )
    assert not comparison.improves


def test_candidate_selection_maximizes_delta_and_breaks_ties_by_index():
    records = [
        {"index": 3, "output_file": "three.trees", "comparison": {"improves": True, "posterior_delta": 2.0}},
        {"index": 1, "output_file": "one.trees", "comparison": {"improves": True, "posterior_delta": 2.0}},
        {"index": 2, "output_file": "two.trees", "comparison": {"improves": False, "posterior_delta": 4.0}},
    ]
    assert select_best_candidate_record(records)["index"] == 1
    assert select_best_candidate_record([records[2]]) is None


def test_exterior_mixed_recombination_becomes_local_identity_transition():
    trace = SimpleNamespace(
        edge_child=(291, 291),
        edge_parent=(2959, 2960),
        edge_left=(499874.0, 504721.0),
        edge_right=(504721.0, 506902.0),
    )
    selected_event = SimpleNamespace(
        mode="mixed_boundary",
        authorized_edge_ids=(0,),
    )
    state = SimpleNamespace(
        block_boundaries=(400000.0, 499874.0, 499962.0, 500000.0),
        active_lineages=(
            SimpleNamespace(
                node_id=291,
                material_segments=MaterialSegments(((1, 3),)),
            ),
        ),
    )
    source_to_live = {291: 291}

    collapsed = _canonicalize_exterior_recombination(
        selected_event=selected_event,
        trace=trace,
        state=state,
        source_child_id=291,
        live_child_id=291,
        parent_ids=(2959, 2960),
        source_to_live=source_to_live,
    )

    assert collapsed
    assert source_to_live[2959] == 291
    assert 2960 not in source_to_live


def test_exterior_recombination_canonicalization_fails_closed_on_partial_coverage():
    trace = SimpleNamespace(
        edge_child=(291,),
        edge_parent=(2959,),
        edge_left=(499900.0,),
        edge_right=(504721.0,),
    )
    state = SimpleNamespace(
        block_boundaries=(400000.0, 499874.0, 499962.0, 500000.0),
        active_lineages=(
            SimpleNamespace(
                node_id=291,
                material_segments=MaterialSegments(((1, 3),)),
            ),
        ),
    )

    assert not _canonicalize_exterior_recombination(
        selected_event=SimpleNamespace(
            mode="mixed_boundary",
            authorized_edge_ids=(0,),
        ),
        trace=trace,
        state=state,
        source_child_id=291,
        live_child_id=291,
        parent_ids=(2959, 2960),
        source_to_live={291: 291},
    )


def test_offline_report_aggregates_manifest_without_truth(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"source_arg":{"path":"source.trees"},"requests":['
        '{"id":"r1","request":{"genomic_range":[1,2]},'
        '"selected_source":true,"selected_output_file":"selected.trees",'
        '"evaluation":{"valid_splice_rate":0.5}}]}',
        encoding="utf-8",
    )
    report = build_local_refinement_report(manifest)
    assert report["requests"][0]["request_id"] == "r1"
    assert report["requests"][0]["selected_source"] is True
    assert report["requests"][0]["valid_splice_rate"] == pytest.approx(0.5)
