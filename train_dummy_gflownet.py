import torch
import torch.nn as nn
from torch.optim import Adam

from env import ThreadingConfig, MultiLeafThreadEnv
from models import TreeMPNN

def train_dummy():
    REFERENCE_FULL_TREES = [
        {"sites": (0, 4), "tree": ("n", 2.0, ("n", 0.25, 0, 1), ("n", 0.25, 2, 3))},
        {"sites": (4, 8), "tree": ("n", 2.0, ("n", 0.25, 0, 2), ("n", 0.25, 1, 3))},
    ]

    GENO = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    ALL_LEAF_IDS = [0, 1, 2, 3]
    TIME_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
    
    multi_env_cfg = ThreadingConfig.from_raw(GENO, TIME_GRID, 0.4, 0.35, 0.15)
    env = MultiLeafThreadEnv(multi_env_cfg, ALL_LEAF_IDS, REFERENCE_FULL_TREES)

    window_size = 2
    expected_dim = 1 + 3 + (window_size * 2 + 1)
    model = TreeMPNN(node_feature_dim=expected_dim, hidden_dim=32, num_time_pts=len(TIME_GRID))
    
    logZ = nn.Parameter(torch.tensor(0.0))
    optimizer = Adam(list(model.parameters()) + [logZ], lr=5e-3)
    
    n_episodes = 200
    losses = []
    
    print("Starting training...")
    for ep in range(n_episodes):
        st = env.reset()
        log_pf_sum = 0.0
        log_pb_sum = 0.0
        
        while not env.is_terminal(st):
            # Encode
            data = env.encode(st, window_size=2)
            valid_acts = env.valid_actions(st)
            
            # Forward
            logits = model(data.x, data.edge_index, data.focal_seq, data.valid_action_info).squeeze(-1)
            dist = torch.distributions.Categorical(logits=logits)
            
            # Sample
            action_idx = dist.sample()
            log_pf_sum += dist.log_prob(action_idx)
            
            # For backward probability, we assume a uniform distribution over parents.
            # In a general graph this requires knowing the in-degree of the next state.
            # But for simplicity, we treat PB as a uniform constant (e.g. log(1/N)). 
            # In a tree, if PB=1, log_pb = 0.0 (unique path backwards). Let's assume trees are generated via a unique sequence of operations.
            
            # Step
            act = valid_acts[action_idx.item()]
            st, reward, done = env.step(st, act)

        # Let's say we want to penalize recombinations. The default reward is 1.0. 
        # A more complex reward could scale with the number of recombinations or mutations if tracked.
        # For a dummy, we use the default 1.0 reward returned by the env.
        
        loss = (logZ + log_pf_sum - log_pb_sum - torch.log(torch.tensor(reward)))**2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if ep % 20 == 0:
            print(f"Ep {ep}, Loss: {loss.item():.4f}, logZ: {logZ.item():.4f}")
            
    print("Training finished.")

if __name__ == '__main__':
    train_dummy()
