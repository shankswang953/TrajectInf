import torch
import math
import os
def precompute_gaussian_params(point_cloud_a, sigma, device):
    """
    Compute GMM parameters where each point has its own Gaussian component.
    
    Args:
        point_cloud_a: Reference point cloud [N, D]
        sigma: Initial scale for covariance matrices
        device: torch device
    
    Returns:
        weights: [N, ] the weights of different components (equal weights)
        means: [N, D] each point becomes a component mean
        precisions: [N, D, D] the inverse of the covariance matrices
    """
    N, D = point_cloud_a.shape
    
    # Each point becomes a component with equal weight
    weights = torch.ones(N, device=device) / N
    
    for i in range(point_cloud_a.shape[0]):
        dists = torch.sum((point_cloud_a - point_cloud_a[i])**2, dim=1)

        neighbors = torch.sum(dists < (4 * sigma * sigma))
        weights[i] = 1.0 / (neighbors.float() + 1.0) 
        
    # normalize weights
    weights = weights / torch.sum(weights)
    
    # Each point is a component mean
    means = point_cloud_a.clone()
    
    adaptive_sigma = sigma * (1.0 + 0.1 * math.log(max(D, 1)))
    
    # Create precision matrices (inverse of covariance matrices)
    precision_scalar = 1.0 / (adaptive_sigma ** 2)
    precisions = precision_scalar * torch.eye(D, device=device).repeat(N, 1, 1)
    
    return weights, means, precisions

def save_checkpoint(func, optimizer, scheduler, itr, loss_meter, args, filename='checkpoint.pth'):
    """Save model checkpoint"""
    checkpoint = {
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'iteration': itr,
        'loss_meter': loss_meter.avg,
        'args': args
    }
    torch.save(checkpoint, os.path.join(args.train_dir, filename))

def load_checkpoint(func, optimizer, scheduler, args, ckpt_path='ckpt_latest.pth'):
    """Load model checkpoint"""
    ckpt_path = os.path.join(args.train_dir, ckpt_path)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        func.load_state_dict(checkpoint['func_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_itr = checkpoint['iteration']
        loss_meter_avg = checkpoint['loss']
        print(f'Loaded checkpoint from {ckpt_path}')
        print(f'Resuming from iteration {start_itr} with loss {loss_meter_avg:.4f}')
        return start_itr
    else:
        print(f'No checkpoint found at {ckpt_path}')
        return 1
