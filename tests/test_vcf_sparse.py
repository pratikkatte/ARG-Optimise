import math

import pytest
import torch

from env import (
    ARGLineage,
    ARGState,
    CoalescenceChoice,
    MaterialSegments,
    RecombinationChoice,
    SimpleARGEnvironment,
)
from models import ARGModel
from utils import load_vcf_variants


def write_vcf(tmp_path, body_rows, header_extra=""):
    path = tmp_path / "tiny.vcf"
    path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1,length=100>",
                header_extra,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0",
                *body_rows,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def valid_rows():
    return [
        "1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0|1",
        "1\t40\t.\tG\tT\t.\tPASS\t.\tGT\t1|0",
        "1\t80\t.\tC\tA\t.\tPASS\t.\tGT\t0|1",
    ]


def scalar_vcf_candidate_features(env, lineage, breakpoints, device, dtype):
    partials = lineage.partials.to(device=device, dtype=dtype)
    blocks = lineage.material_segments.to_block_list()
    lookup = {int(block): idx for idx, block in enumerate(blocks)}
    seq_len = max(float(env.sequence_length), 1.0)
    num_blocks = max(int(env.num_blocks), 1)
    span_start = lineage.material_segments.span_start
    span_end = lineage.material_segments.span_end
    if span_start is None or span_end is None:
        interval_start = 0.0
        interval_end = float(env.sequence_length)
    else:
        interval_start = env._block_to_sequence_coordinate(span_start)
        interval_end = env._block_to_sequence_coordinate(span_end + 1)
    interval_width = max(float(interval_end - interval_start), 1.0)

    features = []
    for breakpoint in breakpoints:
        breakpoint = int(breakpoint)
        left = torch.zeros(4, dtype=dtype, device=device)
        right = torch.zeros(4, dtype=dtype, device=device)
        left_idx = lookup.get(breakpoint - 1)
        right_idx = lookup.get(breakpoint)
        if left_idx is not None:
            left = partials[left_idx]
        if right_idx is not None:
            right = partials[right_idx]
        scalar = torch.tensor(
            [
                float(breakpoint) / float(num_blocks),
                float(env._breakpoint_gap_length(breakpoint)) / seq_len,
                (env._block_to_sequence_coordinate(breakpoint) - interval_start) / interval_width,
            ],
            dtype=dtype,
            device=device,
        )
        features.append(torch.cat([scalar, left, right], dim=0))
    return torch.stack(features, dim=0)


def test_load_vcf_haploidizes_phased_diploid_samples(tmp_path):
    data = load_vcf_variants(write_vcf(tmp_path, valid_rows()))

    assert data.input_mode == "vcf"
    assert data.sample_ids == ["s0"]
    assert data.haplotype_ids == ["s0_h0", "s0_h1"]
    assert data.sequence_length == 100
    assert data.positions0.tolist() == [9, 39, 79]
    assert data.haplotype_partials.shape == (2, 3, 4)
    assert data.haplotype_partials.sum(axis=-1).tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


@pytest.mark.parametrize(
    "bad_row,match",
    [
        ("1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0/1", "Unphased"),
        ("1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0|.", "Missing or unsupported"),
        ("1\t10\t.\tA\tC,G\t.\tPASS\t.\tGT\t0|1", "Multiallelic"),
        ("1\t10\t.\tA\tCT\t.\tPASS\t.\tGT\t0|1", "Only biallelic"),
        ("1\t10\t.\tN\tC\t.\tPASS\t.\tGT\t0|1", "Only biallelic"),
    ],
)
def test_load_vcf_rejects_unsupported_v1_records(tmp_path, bad_row, match):
    with pytest.raises(ValueError, match=match):
        load_vcf_variants(write_vcf(tmp_path, [bad_row]))


def test_material_segments_and_lineage_block_tensor_cache():
    contiguous = MaterialSegments(((0, 3),))
    disjoint = MaterialSegments(((0, 2), (4, 6)))
    empty = MaterialSegments(())

    assert contiguous.to_block_tensor("cpu").tolist() == [0, 1, 2]
    assert disjoint.to_block_tensor("cpu").tolist() == [0, 1, 4, 5]
    assert empty.to_block_tensor("cpu").numel() == 0

    lineage = ARGLineage(
        node_id=1,
        material_segments=disjoint,
        num_blocks=6,
        partials=torch.ones(4, 4),
    )
    first = lineage.block_indices_tensor("cpu")
    second = lineage.block_indices_tensor(torch.device("cpu"))

    assert first.tolist() == [0, 1, 4, 5]
    assert first.data_ptr() == second.data_ptr()

    lineage.material_mask = [False, True, False, True, False, False]
    refreshed = lineage.block_indices_tensor("cpu")
    assert refreshed.tolist() == [1, 3]
    assert refreshed.data_ptr() != first.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_lineage_block_tensor_cache_can_live_on_cuda():
    lineage = ARGLineage(
        node_id=1,
        material_segments=MaterialSegments(((0, 2), (4, 5))),
        num_blocks=5,
        partials=torch.ones(3, 4),
    )

    blocks = lineage.block_indices_tensor("cuda")
    assert blocks.device.type == "cuda"
    assert blocks.tolist() == [0, 1, 4]


def test_vcf_environment_uses_compact_variant_partials_and_transitions(tmp_path):
    variant_data = load_vcf_variants(write_vcf(tmp_path, valid_rows()))
    env = SimpleARGEnvironment(variant_data=variant_data, seed=1)
    state = env.get_initial_state()

    assert env.input_mode == "vcf"
    assert env.num_sequences == 2
    assert env.num_blocks == 3
    assert state.active_lineages[0].partials.shape == (3, 4)

    recomb = RecombinationChoice(
        active_lineage_i=0,
        material_count=3,
        span_start=0,
        span_end=2,
        breakpoint=1,
        time_quantile=0.5,
    )
    next_state = env.apply_recombination(state, recomb)
    left, right = next_state.active_lineages[-2:]

    assert left.material_segments.segments == ((0, 1),)
    assert right.material_segments.segments == ((1, 3),)
    assert left.partials.shape == (1, 4)
    assert right.partials.shape == (2, 4)


def test_vcf_encoder_can_encode_visible_lineage_subset(tmp_path):
    variant_data = load_vcf_variants(write_vcf(tmp_path, valid_rows()))
    env = SimpleARGEnvironment(variant_data=variant_data, seed=1)
    state = env.get_initial_state()
    model = ARGModel(
        env,
        embedding_size=16,
        hidden_size=32,
        transformer_depth=1,
        transformer_heads=4,
        breakpoint_gap_layers=1,
        time_layers=1,
    )
    model.eval()

    with torch.no_grad():
        lineage_reps, summary_reps, _, active_counts = model._encode_states(
            [state],
            visible_lineage_indices_by_state=[(0,)],
        )

    assert lineage_reps.shape == (1, len(state.active_lineages), model.embedding_size)
    assert summary_reps.shape == (1, model.embedding_size)
    assert active_counts.tolist() == [len(state.active_lineages)]
    assert not torch.allclose(lineage_reps[0, 0], torch.zeros_like(lineage_reps[0, 0]))
    assert torch.allclose(lineage_reps[0, 1], torch.zeros_like(lineage_reps[0, 1]))


def test_vcf_breakpoint_candidate_features_match_scalar_reference(tmp_path):
    variant_data = load_vcf_variants(write_vcf(tmp_path, valid_rows()))
    env = SimpleARGEnvironment(variant_data=variant_data, seed=1)
    state = env.get_initial_state()
    model = ARGModel(
        env,
        embedding_size=16,
        hidden_size=32,
        transformer_depth=1,
        transformer_heads=4,
        breakpoint_gap_layers=1,
        breakpoint_gap_dropout=0.0,
        time_layers=1,
    )
    model.eval()

    action = env.compute_recombination_actions(state)[0]
    breakpoints = list(range(action.span_start + 1, action.span_end + 1))
    scorer = model.breakpoint_scorer
    device = next(scorer.parameters()).device
    dtype = next(scorer.parameters()).dtype
    breakpoint_tensor = scorer._valid_breakpoints_tensor(action, device)

    actual_features = scorer._candidate_features(
        breakpoint_tensor,
        state.active_lineages[0],
        device,
        dtype,
        env.sequence_length,
        env.num_blocks,
    )
    expected_features = scalar_vcf_candidate_features(
        env,
        state.active_lineages[0],
        breakpoints,
        device,
        dtype,
    )
    assert torch.allclose(actual_features, expected_features)

    action_context = torch.randn(model.embedding_size * 4, device=device, dtype=dtype)
    actual_logits = scorer.valid_breakpoint_logits(
        action,
        state.active_lineages[0],
        env.sequence_length,
        env.num_blocks,
        action_context,
    )
    expected_logits = scorer.gap_scorer(
        torch.cat([action_context.expand(len(breakpoints), -1), expected_features], dim=1)
    ).squeeze(-1)
    assert torch.allclose(actual_logits, expected_logits)


def test_vcf_breakpoint_candidate_features_zero_missing_disjoint_neighbors(tmp_path):
    variant_data = load_vcf_variants(write_vcf(tmp_path, valid_rows()))
    env = SimpleARGEnvironment(variant_data=variant_data, seed=1)
    state = env.get_initial_state()
    model = ARGModel(
        env,
        embedding_size=16,
        hidden_size=32,
        transformer_depth=1,
        transformer_heads=4,
        breakpoint_gap_layers=1,
        time_layers=1,
    )

    partials = state.active_lineages[0].partials.index_select(
        0,
        torch.tensor([0, 2], dtype=torch.long),
    )
    lineage = ARGLineage(
        node_id=99,
        material_segments=MaterialSegments(((0, 1), (2, 3))),
        num_blocks=env.num_blocks,
        partials=partials,
    )
    action = RecombinationChoice(
        active_lineage_i=0,
        material_count=2,
        span_start=0,
        span_end=2,
    )
    scorer = model.breakpoint_scorer
    device = next(scorer.parameters()).device
    dtype = next(scorer.parameters()).dtype
    features = scorer._candidate_features(
        scorer._valid_breakpoints_tensor(action, device),
        lineage,
        device,
        dtype,
        env.sequence_length,
        env.num_blocks,
    )

    assert torch.allclose(features[0, 3:7], partials[0].to(device=device, dtype=dtype))
    assert torch.count_nonzero(features[0, 7:11]) == 0
    assert torch.count_nonzero(features[1, 3:7]) == 0
    assert torch.allclose(features[1, 7:11], partials[1].to(device=device, dtype=dtype))


def test_sparse_vcf_batch_feature_builder_matches_scalar_reference(tmp_path):
    variant_data = load_vcf_variants(write_vcf(tmp_path, valid_rows()))
    env = SimpleARGEnvironment(variant_data=variant_data, seed=1)
    state = env.get_initial_state()
    model = ARGModel(
        env,
        embedding_size=16,
        hidden_size=32,
        transformer_depth=1,
        transformer_heads=4,
        breakpoint_gap_layers=1,
        time_layers=1,
    )

    full_partials = state.active_lineages[0].partials
    lineages = [
        ARGLineage(
            node_id=10,
            material_segments=MaterialSegments(((0, 1),)),
            num_blocks=env.num_blocks,
            partials=full_partials[:1],
        ),
        ARGLineage(
            node_id=11,
            material_segments=MaterialSegments(((1, 3),)),
            num_blocks=env.num_blocks,
            partials=full_partials[1:],
        ),
        state.active_lineages[0],
        ARGLineage(
            node_id=12,
            material_segments=MaterialSegments(((0, 1), (2, 3))),
            num_blocks=env.num_blocks,
            partials=full_partials.index_select(0, torch.tensor([0, 2], dtype=torch.long)),
        ),
        ARGLineage(
            node_id=13,
            material_segments=MaterialSegments(()),
            num_blocks=env.num_blocks,
            partials=torch.zeros(0, 4),
        ),
    ]
    max_tokens = max(max(int(lineage.material_segments.count), 1) for lineage in lineages)

    expected = [model._sparse_lineage_features(lineage, max_tokens) for lineage in lineages]
    expected_token_features = torch.stack([item[0] for item in expected], dim=0)
    expected_token_mask = torch.stack([item[1] for item in expected], dim=0)
    expected_lineage_features = torch.stack([item[2] for item in expected], dim=0)
    actual_token_features, actual_token_mask, actual_lineage_features = (
        model._sparse_lineage_batch_features(lineages)
    )

    assert torch.allclose(actual_token_features, expected_token_features)
    assert torch.equal(actual_token_mask, expected_token_mask)
    assert torch.allclose(actual_lineage_features, expected_lineage_features)
    assert actual_token_mask.sum(dim=1).tolist() == [1, 2, 3, 2, 0]
    assert torch.isfinite(actual_lineage_features).all()


def test_sparse_vcf_encode_states_handles_multiple_states(tmp_path):
    variant_data = load_vcf_variants(write_vcf(tmp_path, valid_rows()))
    env = SimpleARGEnvironment(variant_data=variant_data, seed=1)
    state = env.get_initial_state()
    recomb = RecombinationChoice(
        active_lineage_i=0,
        material_count=3,
        span_start=0,
        span_end=2,
        breakpoint=1,
        time_quantile=0.5,
    )
    recombined_state = env.apply_recombination(state, recomb)
    model = ARGModel(
        env,
        embedding_size=16,
        hidden_size=32,
        transformer_depth=1,
        transformer_heads=4,
        breakpoint_gap_layers=1,
        time_layers=1,
    )

    lineage_reps, summary_reps, lineage_features, active_counts = model._encode_states(
        [state, recombined_state]
    )

    assert active_counts.tolist() == [2, 3]
    assert lineage_reps.shape[:2] == (2, 3)
    assert torch.isfinite(lineage_reps).all()
    assert torch.isfinite(summary_reps).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_sparse_vcf_batch_feature_builder_can_run_on_cuda(tmp_path):
    variant_data = load_vcf_variants(write_vcf(tmp_path, valid_rows()))
    env = SimpleARGEnvironment(
        variant_data=variant_data,
        seed=1,
        device="cuda",
    )
    state = env.get_initial_state()
    model = ARGModel(
        env,
        embedding_size=16,
        hidden_size=32,
        transformer_depth=1,
        transformer_heads=4,
        breakpoint_gap_layers=1,
        time_layers=1,
    ).to("cuda")

    token_features, token_mask, lineage_features = model._sparse_lineage_batch_features(
        state.active_lineages
    )
    lineage_reps, summary_reps, _, _ = model._encode_states([state])

    assert token_features.device.type == "cuda"
    assert token_mask.device.type == "cuda"
    assert lineage_features.device.type == "cuda"
    assert lineage_reps.device.type == "cuda"
    assert summary_reps.device.type == "cuda"


def test_vcf_terminal_state_exports_tree_sequence(tmp_path):
    variant_data = load_vcf_variants(write_vcf(tmp_path, valid_rows()))
    env = SimpleARGEnvironment(variant_data=variant_data, seed=1)
    state = env.get_initial_state()
    action = CoalescenceChoice(
        active_lineage_i=0,
        active_lineage_j=1,
        time_quantile=0.5,
    )

    terminal = env.apply_coalescence(state, action)
    assert terminal.is_done
    assert math.isfinite(terminal.log_reward)

    ts = env.save_to_tree_sequence(terminal)
    assert int(ts.sequence_length) == 100
    assert ts.num_edges > 0


def test_terminal_likelihood_handles_deep_arg_without_recursion_error():
    env = SimpleARGEnvironment(
        sequences=["A"],
        bp_per_blocks=1,
        population_size=10000,
        mutation_rate=1e-8,
        recombination_rate=0.0,
        reward_C=0.0,
        seed=1,
        device="cpu",
    )
    material = MaterialSegments.full(env.num_blocks)
    all_nodes = {
        0: ARGLineage(
            node_id=0,
            children=[],
            parents=[1],
            material_segments=material,
            num_blocks=env.num_blocks,
            time=0.0,
        )
    }
    previous_id = 0
    depth = 1500
    for node_id in range(1, depth + 1):
        all_nodes[node_id] = ARGLineage(
            node_id=node_id,
            children=[previous_id],
            parents=[node_id + 1] if node_id < depth else [],
            material_segments=material,
            num_blocks=env.num_blocks,
            time=float(node_id) * 1e-6,
        )
        previous_id = node_id

    state = ARGState(
        active_lineages=[all_nodes[depth]],
        all_nodes=all_nodes,
        max_node_idx=depth,
        is_done=True,
        total_active_blocks=env.num_blocks,
        current_time=float(depth) * 1e-6,
    )

    log_likelihood = env.evolution_model.compute_arg_log_likelihood(state)

    assert math.isfinite(log_likelihood)


def test_sparse_vcf_model_outputs_finite_action_and_breakpoint_logits(tmp_path):
    variant_data = load_vcf_variants(write_vcf(tmp_path, valid_rows()))
    env = SimpleARGEnvironment(variant_data=variant_data, seed=1)
    state = env.get_initial_state()
    model = ARGModel(
        env,
        embedding_size=16,
        hidden_size=32,
        transformer_depth=1,
        transformer_heads=4,
        breakpoint_gap_layers=1,
        time_layers=1,
    )

    lineage_reps, summary_reps, lineage_features, active_counts = model._encode_states([state])
    assert torch.isfinite(lineage_reps).all()
    assert torch.isfinite(summary_reps).all()
    assert active_counts.tolist() == [2]

    recomb_action = env.compute_recombination_actions(state)[0]
    action_context = model._batched_action_features(
        [recomb_action],
        0,
        lineage_reps,
        summary_reps,
    )[0]
    logits = model.breakpoint_scorer.valid_breakpoint_logits(
        recomb_action,
        state.active_lineages[0],
        env.sequence_length,
        env.num_blocks,
        action_context,
    )
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
