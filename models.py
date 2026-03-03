import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

class TreeMPNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_time_pts):
        super().__init__()
        self.node_embed = nn.Linear(node_feature_dim, hidden_dim)
        
        # Message passing layers
        self.bottom_up_mp = GATv2Conv(hidden_dim, hidden_dim, add_self_loops=False)
        self.top_down_mp = GATv2Conv(hidden_dim, hidden_dim, add_self_loops=False)
        
        # Time point embeddings
        self.time_embed = nn.Embedding(num_time_pts, hidden_dim)

        self.branch_combiner = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Action head
        # We concatenate branch_embed + time_embed + extra context to get logits
        self.action_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim), # branch_embed + time_embed + focal_seq_embed
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Sequence Embedder (very simple linear for now)
        self.seq_embed = nn.Linear(node_feature_dim, hidden_dim)

    def forward(self, x, edge_index, focal_seq, valid_action_info):
        """
        x: [N, D_node]
        edge_index: [2, E] (Top-down directed)
        focal_seq: [1, D_node]
        valid_action_info: list of dicts or tuples containing (branch_child, branch_parent, time_idx)
        """
        if len(valid_action_info) == 0:
            return torch.empty((0, 1), device=x.device)
            
        # Node Embedding
        h = self.node_embed(x)
        
        # Create bottom-up edges (child -> parent)
        edge_index_bu = edge_index.flip([0])
        
        # Bottom-up passing
        h_bu = self.bottom_up_mp(h, edge_index_bu)
        h = h + torch.relu(h_bu)
        
        # Top-down passing
        h_td = self.top_down_mp(h, edge_index)
        h = h + torch.relu(h_td)
        
        # Context embedding
        context = self.seq_embed(focal_seq) # [1, hidden_dim]
        
        # Compute action logits
        logits = []
        for action in valid_action_info:
            child, parent, t_idx = action['branch_child'], action['branch_parent'], action['time_idx']
            h_child = h[child]
            
            # Root branch case: parent might be -1
            if parent == -1:
                # Use just the child embedding or a special root embedding
                h_parent = torch.zeros_like(h_child)
            else:
                h_parent = h[parent]
                
            branch_embed = self.branch_combiner(torch.cat([h_child, h_parent], dim=-1))
            
            t_emb = self.time_embed(torch.tensor(t_idx, device=h.device))
            
            # Combine all
            combined = torch.cat([branch_embed, t_emb, context.squeeze(0)], dim=-1)
            logit = self.action_mlp(combined)
            logits.append(logit)
            
        return torch.cat(logits, dim=0)

