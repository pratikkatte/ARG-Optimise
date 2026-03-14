import torch
import torch.nn as nn
import math

# Assuming calc_mutation_log_likelihood is available via utils
# In this environment, we'll import if available or provide the wrapper implementation.
from utils import _binary_site_log_likelihood

def calc_mutation_log_likelihood(tree: dict, site_alleles: torch.Tensor, theta: float = 1.0) -> torch.Tensor:
    """
    Given a local site tree and observed alleles (sample states), returns the log-likelihood
    of the alleles using Felsenstein's pruning algorithm.

    Args:
        tree: A dictionary containing at minimum:
            - 'edge_index': (2, E) LongTensor of directional (parent, child) edges
            - 'num_nodes': integer count of all nodes
            - 'root': integer ID of the root node
            - 'node_times': (num_nodes,) FloatTensor with ages
            - 'node_sample_ids': Tuple or list indicating the sample ID for leaf nodes (-1 for internals)
        site_alleles: (num_samples,) tensor of observed binary alleles for the current site.
        theta: Population scaled mutation rate (default 1.0; adjust as needed per your prior).
        
    Returns:
        log_likelihood: A scalar torch tensor representing the log likelihood of the site.
    """
    # Assuming the underlying `_binary_site_log_likelihood` returns a standard python float or scalar math
    log_li = _binary_site_log_likelihood(
        edge_index=tree["edge_index"],
        num_nodes=tree["num_nodes"],
        root=tree["root"],
        node_times=tree["node_times"],
        node_sample_ids=tree["node_sample_ids"],
        site_observation=site_alleles,
        theta=theta,
    )
    
    # We return it as a tensor (on same device as the alleles) so it integrates with PyTorch.
    # Note: If Felsenstein's requires gradient flow *through the tree times*, 
    # _binary_site_log_likelihood would need to be pure PyTorch differentiable ops. 
    # The current utils.py implementation uses `math.exp` locally, breaking backprop 
    # through time! However, for GFlowNet TB Loss context, the reward R = p(x|G) does not 
    # receive gradients directly to update the model policy anyway. The loss only backprops
    # through the policy log_probs! So returning a detached tensor is perfectly fine here.
    return torch.tensor(log_li, dtype=torch.float32, device=site_alleles.device)


class TBLoss(nn.Module):
    """
    Trajectory Balance (TB) Loss Module for GFlowNet.
    
    This matches the specific objective:
    Loss = (log Z + sum(log P_F) - sum(log P_B) - log R)^2
    
    It maintains the learnable parameter `log_Z`.
    """
    def __init__(self, init_log_z: float = 0.0):
        super().__init__()
        # log Z is the log-partition function, learned as a scalar parameter.
        self.log_Z = nn.Parameter(torch.tensor([init_log_z], dtype=torch.float32))
        
    def forward(self, sum_log_pf: torch.Tensor, sum_log_pb: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
        """
        Computes the TB loss. Note that GFlowNets can be trained on batches of trajectories,
        so inputs can be batched (shape: [B]) or single scalar tensors.
        
        Args:
            sum_log_pf: Accumulated forward log-probabilities of actions in trajectory (requires_grad)
            sum_log_pb: Accumulated backward log-probabilities of actions in trajectory
            log_r: Log-reward (log mutation likelihood) of the terminal trajectory
            
        Returns:
            The squared trajectory balance error scalar loss.
        """
        # (log Z + sum_log_PF - sum(log P_B) - log R)^2
        diff = self.log_Z + sum_log_pf - sum_log_pb - log_r
        return (diff ** 2).mean()


def train_gflownet(model, tb_loss_module, optimizers, dataloader, env, epochs: int, save_path: str = "gflownet_model.pt"):
    """
    Main training loop utilizing Trajectory Balance Loss for the ARG GFlowNet.
    
    Args:
        model: The forward policy (Transformer model returning log-probs/logits).
        tb_loss_module: Instance of TBLoss containing the learnable log_Z.
        optimizers: An iterable or single optimizer. E.g., `[policy_opt, z_opt]`.
                    *Justification:* It is best practice to allow passing optimizers separately,
                    because sometimes `log_Z` requires a wildly different learning rate (e.g., 10x
                    larger) than the Transformer weights for stable Trajectory Balance training. 
                    Therefore, the training loop takes a list, steps all provided optimizers.
        dataloader: A loader yielding genetic data windows (site_alleles over sequence L).
        env: The environment, abstracted to provide an `env.step(tree, action)` interface.
        epochs: Number of epochs to train.
        save_path: Path to save the trained model weights.
    """
    # Ensure optimizers is a list for easy iteration
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
        
    model.train()
    tb_loss_module.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, window_alleles in enumerate(dataloader):
            # window_alleles shape = [L, num_samples]. 
            # We assume a single trajectory (batch size 1) loop for clarity as described.
            # L is the number of variant sites.
            
            L = window_alleles.shape[0]
            
            # Step 1. Initialize
            # Start with an initial local tree s_0 (e.g. at the first site)
            # and zero out the accumulators.
            s_t = env.reset() # Using abstracted reset getting initial tree
            
            sum_log_pf = torch.tensor(0.0, device=window_alleles.device, requires_grad=True).clone()
            sum_log_pb = torch.tensor(0.0, device=window_alleles.device)
            log_R = torch.tensor(0.0, device=window_alleles.device)
            
            # Step 2. Step Through the Window
            for t in range(L - 1): # L sites means L-1 steps (actions) between them
                # Get site alleles. 
                current_alleles = window_alleles[t]
                
                # Forward pass: dynamically encode current state
                X, D, A = env.encode(s_t)
                
                # Add batch dimension for the model
                X_batch = X.unsqueeze(0)
                D_batch = D.unsqueeze(0)
                A_batch = A.unsqueeze(0)
                
                outputs = model(X_batch, D_batch, A_batch)
                
                vp_logits = outputs["vp_logits"].squeeze(0)
                vr_logits = outputs["vr_logits"].squeeze(0)
                time_logits = outputs["time_logits"].squeeze(0)
                
                # Get valid target nodes
                masks = env.get_action_masks(s_t)
                if not masks:
                    break
                target_node_mask = masks["target_node_mask"].to(vr_logits.device)
                
                # Mask vr_logits
                vr_logits[~target_node_mask] = -float('inf')
                
                # Sample the actions
                vp_dist = torch.distributions.Categorical(logits=vp_logits)
                vp = vp_dist.sample()
                
                vr_dist = torch.distributions.Categorical(logits=vr_logits)
                vr = vr_dist.sample()
                
                # Get valid times for this target node
                time_masks = env.get_action_masks(s_t, int(vr.item()))
                time_mask = time_masks["time_mask"].to(time_logits.device)
                
                # Mask time_logits
                time_logits[~time_mask] = -float('inf')
                
                time_dist = torch.distributions.Categorical(logits=time_logits)
                t_r = time_dist.sample()
                
                action_tuple = (int(vr.item()), int(t_r.item()))
                
                # Calculate the exact specific log probability for the selected action
                log_p_vp = vp_dist.log_prob(vp)
                log_p_vr = vr_dist.log_prob(vr)
                log_p_time = time_dist.log_prob(t_r)
                
                log_pf_step = log_p_vp + log_p_vr + log_p_time
                sum_log_pf = sum_log_pf + log_pf_step
                
                s_next, reward, done = env.step(s_t, action_tuple)
                
                # Backward policy (Uniform over reverse valid actions)
                # Since we don't naturally have reverse actions available yet, dummy uniform.
                num_valid_reverse = int(target_node_mask.sum().item() * time_mask.sum().item())
                log_pb_step = -math.log(max(num_valid_reverse, 1))
                sum_log_pb = sum_log_pb + log_pb_step
                
                s_t = s_next
                if done:
                    break
                
            # After traversing the window, calculate the total or terminal Reward.
            # In your formulation, you noted calculating reward step-by-step:
            # log_r_step = calc_mutation_log_likelihood(s_{t+1}, alleles[t+1]) -> sum to log_R.
            
            total_log_likelihood = 0.0
            
            # Use the environment's method to reconstruct the final trees
            final_trees_info = env.reconstruct_all_local_trees(s_t)
            
            # final_trees_info is a tuple of dicts: {"sites": (start, end), "tree": TimedTree_tuple}
            # We need to evaluate the likelihood of each site.
            for tree_dict in final_trees_info:
                start_site, end_site = tree_dict["sites"]
                timed_tree = tree_dict["tree"]
                
                # We need to format `timed_tree` into the graph dict format expected by _binary_site_log_likelihood
                # utils.py provides `_timed_tree_to_graph_full`
                from utils import _timed_tree_to_graph_full
                edge_index, num_nodes, root, node_times, node_sample_ids = _timed_tree_to_graph_full(timed_tree)
                
                graph_format_tree = {
                    "edge_index": edge_index,
                    "num_nodes": num_nodes,
                    "root": root,
                    "node_times": node_times,
                    "node_sample_ids": node_sample_ids
                }
                
                # Calculate likelihood for each site covered by this local tree
                for t in range(start_site, min(end_site, L)):
                    if t >= L: break
                    ll_t = calc_mutation_log_likelihood(graph_format_tree, window_alleles[t])
                    total_log_likelihood += ll_t.item()
            loss = tb_loss_module(sum_log_pf, sum_log_pb, log_R)
            
            for opt in optimizers:
                opt.zero_grad()
                
            loss.backward()
            
            for opt in optimizers:
                opt.step()
                
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - TB Loss: {epoch_loss / len(dataloader):.4f}")

    # Save the model and tb_loss module
    torch.save({
        'model_state_dict': model.state_dict(),
        'tb_loss_state_dict': tb_loss_module.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")

