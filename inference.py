import torch

def sample_posterior(model, env, window_alleles, num_samples: int = 100, device: str = 'cpu'):
    """
    Samples ARGs from the posterior distribution using the trained GFlowNet policy.
    
    Args:
        model: The trained Forward Policy (Transformer model).
        env: The ARG environment.
        window_alleles: (L, num_samples) tensor of alleles for the genomic window.
        num_samples: Number of ARGs to sample.
        device: Device to run the sampling on.
        
    Returns:
        List of generated terminal states (ARGs).
    """
    model.eval()
    sampled_args = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Same environment initialization as in training
            s_t = env.reset()
            done = False
            
            # Autoregressively generate the sequence
            while not done:
                # Encode current state
                X, D, A = env.encode(s_t)
                
                # Add batch dimension and move to device
                X_batch = X.unsqueeze(0).to(device)
                D_batch = D.unsqueeze(0).to(device)
                A_batch = A.unsqueeze(0).to(device)
                
                # Forward pass
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
                if not time_masks:
                    break
                time_mask = time_masks["time_mask"].to(time_logits.device)
                
                # Mask time_logits
                time_logits[~time_mask] = -float('inf')
                
                time_dist = torch.distributions.Categorical(logits=time_logits)
                t_r = time_dist.sample()
                
                action_tuple = (int(vr.item()), int(t_r.item()))
                
                # Step the environment
                s_next, reward, done = env.step(s_t, action_tuple)
                s_t = s_next
                
            sampled_args.append(s_t)
            
    return sampled_args

def load_and_sample(model_class, model_kwargs, model_path: str, env, window_alleles, num_samples: int = 100, device: str = 'cpu'):
    """
    Utility function to load a saved model, environment, and sample from the posterior.
    
    Args:
        model_class: The PyTorch class for the Forward Policy.
        model_kwargs: Dictionary of initialization arguments for model_class.
        model_path: Path where the model is saved (.pt).
        env: Instantiated ARG environment.
        window_alleles: Tensor of site alleles.
        num_samples: Total number of ARGs to sample.
        device: Device representation (e.g. 'cuda:0', 'cpu')
    
    Returns:
        List of terminal state ARGs natively sampled proportional to their rewards.
    """
    # Initialize model
    model = model_class(**model_kwargs).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Successfully loaded model from {model_path}")
    
    # Sample
    print(f"Sampling {num_samples} ARGs from the posterior...")
    samples = sample_posterior(model, env, window_alleles, num_samples, device)
    print(f"Completed sampling {len(samples)} ARGs.")
    
    return samples

if __name__ == "__main__":
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description="Inference for ARG GFlowNet")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of ARGs to sample")
    parser.add_argument("--output_file", type=str, default="sampled_args.pkl", help="File to save sampled ARG states outputs")
    
    args = parser.parse_args()
    
    print(f"Ready to run inference using {args.model_path} and sample {args.num_samples} ARGs.")
    print("WARNING: You still must import your specific dataloader/env instantiation and model definitions before calling load_and_sample in a real script or notebook context.")
