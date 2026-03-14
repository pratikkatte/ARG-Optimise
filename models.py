import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TreeAwareAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Learnable scalars for structural matrices, one per head
        self.w_d = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        self.w_a = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        
    def forward(self, x, D, A):
        """
        x: (Batch, N, d_model)
        D: (Batch, N, N) - Path Distance matrix
        A: (Batch, N, N) - Ancestry matrix
        """
        B, N, _ = x.size()
        
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Incorporate structural biases by broadcasting them
        # D and A are (B, N, N), so we add the head dimension: (B, 1, N, N)
        scores = scores + self.w_d * D.unsqueeze(1) + self.w_a * A.unsqueeze(1)
        
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        
        return self.out_proj(out)


class TreeAwareTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = TreeAwareAttention(d_model, n_heads)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, D, A):
        attn_out = self.self_attn(x, D, A)
        x = self.norm1(x + self.dropout1(attn_out))
        
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class GFlowNetForwardPolicy(nn.Module):
    def __init__(self, input_dim=4, d_model=64, n_heads=4, num_layers=3, num_time_bins=5):
        super().__init__()
        self.d_model = d_model
        self.num_time_bins = num_time_bins
        
        self.node_embed = nn.Linear(input_dim, d_model)
        
        self.layers = nn.ModuleList([
            TreeAwareTransformerLayer(d_model, n_heads) 
            for _ in range(num_layers)
        ])
        
        # Head 1: Prune Node (v_p)
        self.vp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Head 2: Regraft Node (v_r)
        # Conditioned on v_p, so the input concatenates two d_model representations
        self.vr_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Head 3: Time (t_r)
        # Conditioned on v_p and v_r
        self.time_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_time_bins)
        )
        
    def forward(self, X, D, A, action_targets=None):
        """
        X: (B, N, 4) - [Time, Branch Length, Num Descendants, Mutation State]
        D: (B, N, N) - Path Distance matrix
        A: (B, N, N) - Ancestry matrix 
                       A[i, j] = 1 if node i is an ancestor of node j
        action_targets: Optional tuple of (v_p, v_r, t_r) for teacher-forcing evaluation
                        during loss calculation, bypassing the sampling step.
                        
        Returns a dictionary containing sampled actions and corresponding masked logits.
        """
        is_unbatched = X.dim() == 2
        if is_unbatched:
            X = X.unsqueeze(0)
            D = D.unsqueeze(0)
            A = A.unsqueeze(0)
            if action_targets is not None:
                action_targets = tuple(t.unsqueeze(0) for t in action_targets)
                
        B, N, _ = X.size()
        
        # 1. Base network representation
        h = self.node_embed(X)
        for layer in self.layers:
            h = layer(h, D, A)
        
        # ==========================
        # Head 1: Prune Node (v_p)
        # ==========================
        vp_logits = self.vp_head(h).squeeze(-1) # (B, N)
        
        # Mask the root node (the very last node in the post-order sequence)
        mask_vp = torch.zeros_like(vp_logits, dtype=torch.bool)
        mask_vp[:, -1] = True
        vp_logits = vp_logits.masked_fill(mask_vp, -1e9)
        
        if action_targets is not None:
            vp_sampled = action_targets[0]
        else:
            vp_dist = torch.distributions.Categorical(logits=vp_logits)
            vp_sampled = vp_dist.sample() # (B,)
        
        # Extract embeddings for v_p
        batch_indices = torch.arange(B, device=X.device)
        vp_embed = h[batch_indices, vp_sampled] # (B, d_model)
        vp_embed_exp = vp_embed.unsqueeze(1).expand(-1, N, -1) # (B, N, d_model)
        
        # ==========================
        # Head 2: Regraft Node (v_r)
        # ==========================
        h_vr_input = torch.cat([h, vp_embed_exp], dim=-1) # (B, N, 2*d_model)
        vr_logits = self.vr_head(h_vr_input).squeeze(-1)  # (B, N)
        
        # Masking: We cannot regraft a node onto its own descendants or itself.
        # A[v_p, :] grabs the ancestry vector for v_p, matching A[v_p, j] == 1
        is_descendant = A[batch_indices, vp_sampled, :] == 1 # (B, N)
        is_vp = F.one_hot(vp_sampled, num_classes=N).bool()  # (B, N)
        
        mask_vr = is_descendant | is_vp
        vr_logits = vr_logits.masked_fill(mask_vr, -1e9)
        
        if action_targets is not None:
            vr_sampled = action_targets[1]
        else:
            vr_dist = torch.distributions.Categorical(logits=vr_logits)
            vr_sampled = vr_dist.sample() # (B,)
        
        # Extract embeddings for v_r
        vr_embed = h[batch_indices, vr_sampled] # (B, d_model)
        
        # ==========================
        # Head 3: Regraft Time (t_r)
        # ==========================
        h_time_input = torch.cat([vp_embed, vr_embed], dim=-1) # (B, 2*d_model)
        time_logits = self.time_head(h_time_input) # (B, T)
        
        # Extract time features from the input X for both v_p and v_r.
        # X[..., 0] corresponds to the Time feature.
        t_vp = X[batch_indices, vp_sampled, 0] # (B,)
        t_vr = X[batch_indices, vr_sampled, 0] # (B,)
        
        # Represent discrete time bins assuming bins roughly map sequentially 0..T-1.
        # Adjusting the values of `time_bins` might be needed if your Time features scale
        # differently, but conceptually, the bins follow this shape scale.
        time_bins = torch.arange(self.num_time_bins, device=X.device, dtype=X.dtype).unsqueeze(0).expand(B, -1)
        
        # The regraft time must be strictly older than the pruned node's time (time_bins > t_vp)
        # and older than/equal to the regraft node's time (time_bins >= t_vr)
        valid_time_mask = (time_bins > t_vp.unsqueeze(-1)) & (time_bins >= t_vr.unsqueeze(-1))
        
        time_logits = time_logits.masked_fill(~valid_time_mask, -1e9)
        
        if action_targets is not None:
            time_sampled = action_targets[2]
        else:
            time_dist = torch.distributions.Categorical(logits=time_logits)
            time_sampled = time_dist.sample() # (B,)
            
        outputs = {
            "v_p": vp_sampled,
            "v_r": vr_sampled,
            "t_r": time_sampled,
            "vp_logits": vp_logits,
            "vr_logits": vr_logits,
            "time_logits": time_logits
        }
        
        if is_unbatched:
            for k in outputs:
                outputs[k] = outputs[k].squeeze(0)
                
        return outputs
