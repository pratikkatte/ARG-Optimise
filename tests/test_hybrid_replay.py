from types import SimpleNamespace

import pytest

from refinement.replay import (
    FractionalQuotaAllocator,
    HybridReplayBuffer,
    normalize_hybrid_replay_config,
    structural_topology_signature,
)


def _state(*, reward=0.0, node_time=1.0, breakpoint=None, context_id="ctx"):
    material = SimpleNamespace(segments=((0, 2),))
    child = SimpleNamespace(
        event_type="sample",
        children=[],
        parents=[1],
        material_segments=material,
        breakpoint=None,
        recombination_side=None,
        time=0.0,
        node_id=0,
    )
    parent = SimpleNamespace(
        event_type="coalescence",
        children=[0],
        parents=[],
        material_segments=material,
        breakpoint=breakpoint,
        recombination_side=None,
        time=float(node_time),
        node_id=1,
    )
    return SimpleNamespace(
        all_nodes={0: child, 1: parent},
        active_lineages=[parent],
        local_context_id=context_id,
        log_reward=float(reward),
    )


def _actions(index):
    return [
        {
            "event_type": "coal",
            "active_lineage_i": 0,
            "active_lineage_j": 1,
            "time_quantile": index / 100.0,
            "delta_time": float(index + 1),
            "time_policy_entropy": 123.0,
        }
    ]


def test_hybrid_replay_config_and_fractional_quotas():
    config = normalize_hybrid_replay_config({"enabled": True})
    assert config["capacity_per_context"] == 200
    assert sum(config["fractions"].values()) == pytest.approx(1.0)

    allocator = FractionalQuotaAllocator(config["fractions"])
    totals = {name: 0 for name in config["fractions"]}
    for _ in range(100):
        allocation = allocator.allocate(4)
        assert sum(allocation.values()) == 4
        for name, count in allocation.items():
            totals[name] += count
    assert totals == {
        "fresh": 200,
        "residual": 100,
        "reward": 60,
        "topology": 40,
    }


def test_hybrid_replay_config_rejects_invalid_contract():
    with pytest.raises(ValueError, match="sum to 1"):
        normalize_hybrid_replay_config(
            {
                "enabled": True,
                "fractions": {
                    "fresh": 0.5,
                    "residual": 0.5,
                    "reward": 0.5,
                    "topology": 0.0,
                },
            }
        )
    with pytest.raises(ValueError, match="max_abs_subtb"):
        normalize_hybrid_replay_config({"residual_priority": "rms"})


def test_structural_topology_signature_ignores_time_but_not_structure():
    baseline = structural_topology_signature(_state(node_time=1.0))
    assert structural_topology_signature(_state(node_time=99.0)) == baseline
    assert (
        structural_topology_signature(_state(node_time=1.0, breakpoint=1))
        != baseline
    )


def test_buffer_prioritizes_top_tiers_and_deduplicates():
    buffer = HybridReplayBuffer(
        ["ctx"],
        capacity_per_context=20,
        top_fraction=0.2,
        seed=17,
    )
    for index in range(10):
        entry, status = buffer.add(
            "ctx",
            _actions(index),
            _state(reward=index),
            residual_priority=index * 10,
            step=index,
        )
        assert entry is not None
        assert status == "inserted"

    assert buffer.context_size("ctx") == 10
    assert {
        buffer.sample("residual", "ctx").residual_priority for _ in range(20)
    } <= {80.0, 90.0}
    assert {
        buffer.sample("reward", "ctx").log_reward for _ in range(20)
    } <= {8.0, 9.0}

    duplicate, status = buffer.add(
        "ctx",
        _actions(9),
        _state(reward=9),
        residual_priority=999,
        step=20,
    )
    assert status == "updated"
    assert duplicate.residual_priority == 999
    assert buffer.context_size("ctx") == 10


def test_buffer_capacity_and_topology_uniform_sampling():
    buffer = HybridReplayBuffer(
        ["ctx"],
        capacity_per_context=3,
        top_fraction=1.0,
        seed=3,
    )
    for index in range(20):
        buffer.add(
            "ctx",
            _actions(index),
            _state(reward=index, breakpoint=index % 2),
            residual_priority=index,
            step=index,
        )
    assert len(buffer) == 3
    selected = buffer.sample("topology", "ctx")
    assert selected is not None
    assert selected.context_id == "ctx"
    metrics = buffer.metrics(current_step=25)
    assert metrics["replay/buffer_size"] == 3
    assert metrics["replay/unique_topology_count"] >= 1

