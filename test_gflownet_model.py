import torch
import torch.nn as nn
from env import ThreadingConfig, ARGweaverThreadEnv, MultiLeafThreadEnv
from utils import _canonical_choice
from models import TreeMPNN
import numpy as np

def get_dummy_config():
    return ThreadingConfig(
        sequence_length=10,
        geno=torch.tensor([
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], # leaf 0
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], # leaf 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # leaf 2
        ]),
        time_grid=torch.linspace(0, 10, 5),
        mutation_rate=1e-8,
        recomb_rate=1e-8,
        reward_temperature=1.0
    )


def test_encoding():
    print("Testing PyG Encoding Shape & Models")
    config = get_dummy_config()
    
    from utils import SiteBackboneTree, BackboneSegment
    
    # Edge index is [parent, child]. Root is 2, leaves are 0 and 1
    # So edge_index should be [[2, 0], [2, 1]]. Let's arrange as shape [2, E]
    # utils.py _parent_from_edge_index expects parent = u, child = v from u,v in edge_index.t()
    
    edge_index = torch.tensor([[2, 2], [0, 1]], dtype=torch.long)
    
    backbone_segments = [
        BackboneSegment(
            start=0,
            end=10,
            edge_index=edge_index,
            num_nodes=3,
            root=2,
            node_times=torch.tensor([0.0, 0.0, 5.0]),
            node_sample_ids=(0, 1, -1),
            leaf_ids=(0, 1),
        )
    ]
    
    env = ARGweaverThreadEnv(config, focal_leaf=2, backbone_segments=backbone_segments)
    
    st = env.reset()
    
    # Test Encoding
    window_size = 2
    data = env.encode(st, window_size=window_size)
    
    print("--- Encoding Test ---")
    print("Data object keys:", dict(data).keys())
    print("Node features shape:", data.x.shape) # num_nodes x (1 + 3 + window_size*2+1)
    
    num_nodes = 3
    expected_dim = 1 + 3 + (window_size * 2 + 1)
    assert data.x.shape == (num_nodes, expected_dim), f"Expected node features shape {(num_nodes, expected_dim)} but got {data.x.shape}"
    assert data.focal_seq.shape == (1, expected_dim), f"Expected focal_seq shape {(1, expected_dim)} but got {data.focal_seq.shape}"
    
    num_valid_actions = len(data.valid_action_indices)
    print(f"Number of valid actions: {num_valid_actions}")
    
    # Test Model Forward
    model = TreeMPNN(node_feature_dim=expected_dim, hidden_dim=32, num_time_pts=len(config.time_grid))
    logits = model(data.x, data.edge_index, data.focal_seq, data.valid_action_info)
    
    print("--- Model Test ---")
    print("Output logits shape:", logits.shape)
    assert logits.shape == (num_valid_actions, 1), f"Expected logits shape {(num_valid_actions, 1)} but got {logits.shape}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_encoding()
