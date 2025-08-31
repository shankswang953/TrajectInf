from heapq import nlargest
from itertools import tee
import torch
import torch.nn.functional as F

def map_whole_trajectory2another_manifold(
    traj_points,
    support_points_main,
    support_points_aux,
    k=20,
    alpha_main=100.0,
    sigma=0.05,
    type = "accrossSpace"
):
    if type == "time_terminalMatching":
        return _map_whole_trajectory2another_manifold_time_terminalMatching(traj_points, support_points_main, support_points_aux, k, alpha_main, sigma)
    elif type == "attention":
        return _map_whole_trajectory2another_manifold_attention(traj_points, support_points_main, support_points_aux, k, alpha_main, sigma)
    elif type == "accrossSpace":
        return _map_whole_trajectory2another_manifold_accrossSpace(traj_points, support_points_main, support_points_aux, k, sigma)
    elif type == "Simple":
        return _map_whole_trajectory2another_manifold_Simple(traj_points, support_points_main, support_points_aux, k, sigma)
    else:
        raise ValueError(f"Invalid type: {type}")




def map_to_nearest_manifold(sample_points, manifold_points, k=10, sigma=0.05, max_distance=0.05):
    """
    Map trajectory points to manifold points
    
    Args:
        sample_points: tensor of shape (n_samples, dim) - points from trajectory
        manifold_points: tensor of shape (n_manifold, dim) - points on manifold
        k: number of nearest neighbors (default: 5)
        sigma: kernel bandwidth (default: 0.1)
        max_distance: maximum allowed distance for mapping (default: 0.1)
    
    Returns:
        mapped_points: tensor of shape (n_samples, dim)
        indices: tensor of shape (n_samples, k)
        kernel_weights: tensor of shape (n_samples, k)
    """
    # Calculate pairwise distances
    dist_matrix = torch.cdist(sample_points, manifold_points)
    
    # Get k-nearest neighbors
    distances, indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)
    
    # Apply distance threshold
    if max_distance is not None:
        # Create a mask for distances within threshold
        valid_mask = distances <= max_distance
        # Set large weights for invalid distances
        distances = torch.where(valid_mask, distances, torch.ones_like(distances) * 1e6)
    
    # Compute Gaussian kernel weights
    kernel_weights = torch.exp(-distances**2 / (2 * sigma**2))
    
    # Normalize weights
    kernel_weights = kernel_weights / (kernel_weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # Get corresponding manifold points
    nearest_manifold_points = manifold_points[indices]
    
    # Compute weighted average
    mapped_points = torch.sum(nearest_manifold_points * kernel_weights.unsqueeze(-1), dim=1)
    
    return mapped_points, indices, kernel_weights


def _map_whole_trajectory2another_manifold_attention(
    traj_points, support_points_main, support_points_aux, sigma=0.1, device=None, aux_weight=1, distance_threshold=0.05
):
    """
    Attention-based projection mapping.
    
    traj_points: (N, Sample, dim_1)
    support_points_main: (n_support, dim_1)
    support_points_aux: (n_support, dim_2)
    
    Returns:
        mapped_indices: (N, Sample, n_support)
        weights: (N, Sample, n_support)
    """
    N, Sample, dim_1 = traj_points.shape
    n_support = support_points_main.shape[0]

    mapped_indices = []
    weights = []

    # Optional normalization for stability
    support_points_main_norm = F.normalize(support_points_main, dim=1)
    support_points_aux_norm = F.normalize(support_points_aux, dim=1)

    for t in range(N):
        traj_points_t = F.normalize(traj_points[t], dim=1)

        # Attention scores: dot product similarity
        attention_scores_main = torch.matmul(traj_points_t, support_points_main_norm.T)  # (Sample, n_support)

        if t == 0:
            # Apply Gaussian mask to enforce locality
            dists_main = torch.cdist(traj_points[t], support_points_main)
            mask = dists_main > 0.002
            attention_scores_main = attention_scores_main.masked_fill(mask, -1e6)

            # Soft selection with attention
            soft_weights = torch.softmax(attention_scores_main, dim=1)

            combined_weights = soft_weights

            mean_main = (support_points_main * combined_weights.unsqueeze(-1)).sum(dim=1)
            mean_aux = (support_points_aux * combined_weights.unsqueeze(-1)).sum(dim=1)

        else:
            # Compute attention scores between last mapped mean and support points
            mean_main_norm = F.normalize(mean_main, dim=1)
            mean_aux_norm = F.normalize(mean_aux, dim=1)

            attention_scores_main = torch.matmul(mean_main_norm, support_points_main_norm.T)
            attention_scores_aux = torch.matmul(mean_aux_norm, support_points_aux_norm.T)

            # Multi-space attention: weighted sum
            total_scores = attention_scores_main + aux_weight * attention_scores_aux

            dists_main = torch.cdist(traj_points[t], support_points_main)
            mask = dists_main > distance_threshold
            total_scores = total_scores.masked_fill(mask, -1e6)

            soft_weights = torch.softmax(total_scores, dim=1)

            combined_weights = soft_weights

            # Update mean positions
            mean_main = (support_points_main * combined_weights.unsqueeze(-1)).sum(dim=1)
            mean_aux = (support_points_aux * combined_weights.unsqueeze(-1)).sum(dim=1)

        indices = torch.arange(n_support, device=device).unsqueeze(0).repeat(Sample, 1)

        mapped_indices.append(indices)
        weights.append(combined_weights)

    mapped_indices = torch.stack(mapped_indices)
    weights = torch.stack(weights)

    return mapped_indices, weights


def _map_whole_trajectory2another_manifold_time_terminalMatching(
    traj_points,
    support_points_main,
    support_points_aux,
    k=10,
    alpha_main=100.0,
    sigma_aux=0.05
):
    """
    Multi-space KNN mapping with softmin weights and time continuity.

    Args:
        traj_points: (T, N, D)
        support_points_main: (M, D)
        support_points_aux: (M, D)
        k: number of neighbors
        alpha_main: softmin temperature for main space
        sigma_aux: bandwidth for auxiliary space

    Returns:
        indices: (T, N, k)
        joint_weights: (T, N, k)
    """
    traj_points = traj_points.to(dtype=support_points_main.dtype)
    T, N, D = traj_points.shape
    M = support_points_main.shape[0]

    traj_points_flat = traj_points.reshape(-1, D)

    # ============================
    # Step 1: path index (considering temporal continuity)
    # ============================
    prev_indices = None
    best_indices_list = []

    for t in range(T):
        traj_t = traj_points[t].reshape(N, D)

        # calculate distance between current time step and support
        dist_matrix_t = torch.cdist(traj_t, support_points_main)  # (N, M)
        distances_t, knn_indices_t = torch.topk(dist_matrix_t, k=k, dim=1, largest=False)  # (N, k)

        # main space Gaussian weights
        kernel_main = torch.exp(-distances_t ** 2 / (2 * 0.05 ** 2))
        kernel_main = kernel_main / (kernel_main.sum(dim=1, keepdim=True) + 1e-8)

        if t == 0:
            # first frame: directly take the point with the largest weight
            best_indices = knn_indices_t[torch.arange(N), torch.argmax(kernel_main, dim=1)]
            best_indices_list.append(best_indices)
            prev_indices = best_indices
            continue

        # temporal continuity auxiliary space weights
        prev_mapped_points = support_points_aux[prev_indices]  # (N, D)
        neighbor_points_aux = support_points_aux[knn_indices_t]  # (N, k, D)
        prev_expanded = prev_mapped_points.unsqueeze(1)  # (N, 1, D)
        distances_aux = torch.norm(neighbor_points_aux - prev_expanded, dim=-1)  # (N, k)

        kernel_aux = torch.exp(-distances_aux ** 2 / (2 * 0.025 ** 2))  # sigma_time fixed to 0.05
        kernel_aux = torch.clamp(kernel_aux, min=1e-4)

        joint_kernel = kernel_main * kernel_aux
        joint_kernel = joint_kernel / (joint_kernel.sum(dim=1, keepdim=True) + 1e-8)

        best_indices = knn_indices_t[torch.arange(N), torch.argmax(joint_kernel, dim=1)]
        best_indices_list.append(best_indices)

        prev_indices = best_indices

    # path index, continue batch processing
    path_indices = torch.stack(best_indices_list, dim=0)  # (T, N)

    # ============================
    # Step 2: efficient batch computation (batch processing)
    # ============================
    dist_matrix_support = torch.cdist(support_points_main, support_points_main)  # (M, M)
    distances_main_support, support_knn_indices = torch.topk(dist_matrix_support, k=k, dim=1, largest=False)

    mapped_support_indices = path_indices.view(T * N)
    neighbor_indices = support_knn_indices[mapped_support_indices]  # (T*N, k)

    neighbor_points_main = support_points_main[neighbor_indices]  # (T*N, k, D)
    neighbor_points_aux = support_points_aux[neighbor_indices]    # (T*N, k, D)

    traj_points_expanded = traj_points_flat.unsqueeze(1)  # (T*N, 1, D)

    distances_main = torch.norm(traj_points_expanded - neighbor_points_main, dim=-1)  # (T*N, k)

    # Softmin weights
    kernel_main = F.softmin(distances_main * alpha_main, dim=1)

    path_index_flat = path_indices.view(T * N)
    aux_path = support_points_aux[path_index_flat] # (T*N, D)
    aux_path_expanded = aux_path.unsqueeze(1) # (T*N, 1, D)
    #barycenter_aux = neighbor_points_aux.mean(dim=1, keepdim=True)  # (T*N, 1, D)
    distances_aux = torch.norm(neighbor_points_aux - aux_path_expanded, dim=-1)  # (T*N, k)

    kernel_aux = torch.exp(-distances_aux ** 2 / (2 * sigma_aux ** 2))
    kernel_aux = torch.clamp(kernel_aux, min=1e-4)

    joint_kernel = kernel_main * kernel_aux
    joint_kernel = joint_kernel / (joint_kernel.sum(dim=1, keepdim=True) + 1e-8)

    indices = neighbor_indices.view(T, N, k)
    joint_weights = joint_kernel.view(T, N, k)
    
    del dist_matrix_support, distances_main_support, support_knn_indices, mapped_support_indices, neighbor_indices, neighbor_points_main, neighbor_points_aux, traj_points_expanded, path_index_flat, aux_path, aux_path_expanded, distances_aux, kernel_aux, joint_kernel
    torch.cuda.empty_cache()

    return indices, joint_weights, path_indices

def _map_whole_trajectory2another_manifold_accrossSpace(
    traj_points,
    support_points_main,
    support_points_aux,
    k=10,
    alpha_main=100.0,   # kept for API compatibility; not used
    sigma_aux=0.05
):
    """
    Multi-space KNN mapping (time-flattened, no temporal continuity).

    Args:
        traj_points: (T, N, D) tensor
        support_points_main: (M, D) tensor
        support_points_aux: (M, D) tensor (one-to-one aligned with main)
        k: number of output neighbors per sample (use 20 if you want top-20)
        alpha_main: kept for API compatibility (not used)
        sigma_aux: bandwidth for Gaussian weights in BOTH spaces

    Returns:
        indices: (T, N, k) long tensor of selected support indices
        joint_weights: (T, N, k) float tensor of normalized weights
        path_indices: (T, N) long tensor of nearest (main-space) support per sample
    """
    # Do NOT use in-place .to_() etc.; make aligned copies (out-of-place)
    device = support_points_main.device
    dtype  = support_points_main.dtype
    traj_points = traj_points.to(device=device, dtype=dtype)
    support_points_aux = support_points_aux.to(device=device, dtype=dtype)

    T, N, D = traj_points.shape
    M = support_points_main.shape[0]
    TN = T * N

    # Flatten (T, N, D) -> (TN, D)
    X = traj_points.reshape(TN, D)

    # Candidate pool size from main space
    K_MAIN = min(50, M)
    K_OUT  = min(k, K_MAIN)

    # Preallocate outputs (these buffers don't require grad)
    out_indices = X.new_empty((TN, K_OUT), dtype=torch.long, device=device)
    out_weights = X.new_empty((TN, K_OUT), dtype=dtype, device=device)
    nearest_main_idx = X.new_empty((TN,), dtype=torch.long, device=device)

    # Chunk to control memory; keep everything out-of-place
    CHUNK = 4096 if TN >= 4096 else TN
    eps = torch.as_tensor(1e-12, dtype=dtype, device=device)
    two_sigma2 = torch.as_tensor(2.0 * (sigma_aux ** 2), dtype=dtype, device=device)

    for start in range(0, TN, CHUNK):
        end = min(start + CHUNK, TN)
        X_chunk = X[start:end]                         # (B, D)
        B = X_chunk.shape[0]

        # --- Main space distances & KNN (out-of-place) ---
        dist_main_full = torch.cdist(X_chunk, support_points_main)              # (B, M)
        dist_main_vals, dist_main_idx = torch.topk(dist_main_full, k=K_MAIN, dim=1, largest=False)  # (B, K_MAIN)

        # Anchor (nearest in main)
        anchor_idx = dist_main_idx[:, 0]                                        # (B,)
        nearest_main_idx[start:end] = anchor_idx

        # --- Aux distances between candidates and anchor's aux point (out-of-place) ---
        candidates_aux = support_points_aux[dist_main_idx]                       # (B, K_MAIN, D)
        anchor_aux = support_points_aux[anchor_idx].unsqueeze(1)                 # (B, 1, D)
        dist_aux_vals = torch.norm(candidates_aux - anchor_aux, dim=-1)          # (B, K_MAIN)

        # --- Gaussian weights in BOTH spaces (NO in-place clamp) ---
        w_main = torch.exp(-(dist_main_vals ** 2) / two_sigma2)                  # (B, K_MAIN)
        w_aux  = torch.exp(-(dist_aux_vals  ** 2) / two_sigma2)                  # (B, K_MAIN)

        # Multiply then normalize (out-of-place)
        w_joint = w_main * w_aux                                                 # (B, K_MAIN)
        w_sum = w_joint.sum(dim=1, keepdim=True)
        w_joint = w_joint / (w_sum + eps)

        # Top-K selection (indices are integer, not part of grad)
        topw, topi = torch.topk(w_joint, k=K_OUT, dim=1, largest=True)          # (B, K_OUT)
        top_indices = torch.gather(dist_main_idx, 1, topi)                       # (B, K_OUT)

        out_indices[start:end] = top_indices
        out_weights[start:end] = topw

        
        # Free temporaries early
        del dist_main_full, dist_main_vals, dist_main_idx, candidates_aux, anchor_aux, dist_aux_vals, w_main, w_aux, w_joint, topw, topi, top_indices
        torch.cuda.empty_cache() if X.is_cuda else None


    # Reshape to (T, N, ...)
    indices = out_indices.view(T, N, K_OUT)
    joint_weights = out_weights.view(T, N, K_OUT)
    path_indices = nearest_main_idx.view(T, N)

    return indices, joint_weights, path_indices


def _map_whole_trajectory2another_manifold_Simple(
    traj_points,
    support_points_main,
    support_points_aux,
    k=20,
    sigma=0.05
):
    """
    Simplest variant: time-flattened KNN in MAIN space only.
    Weights come from MAIN-space Gaussian; AUX is only used for mapping by indices.

    Args:
        traj_points: (T, N, D) tensor
        support_points_main: (M, D) tensor
        support_points_aux: (M, D) tensor (one-to-one with main)
        k: number of neighbors to return
        alpha_main: kept for API compatibility (not used)
        sigma_aux: Gaussian bandwidth for MAIN-space distances

    Returns:
        indices: (T, N, k) long tensor of selected support indices (by MAIN KNN)
        joint_weights: (T, N, k) float tensor of normalized MAIN Gaussian weights
        path_indices: (T, N) long tensor of nearest MAIN support per sample
    """
    # Device / dtype alignment (out-of-place)
    device = support_points_main.device
    dtype  = support_points_main.dtype
    traj_points = traj_points.to(device=device, dtype=dtype)
    support_points_aux = support_points_aux.to(device=device, dtype=dtype)

    T, N, D = traj_points.shape
    M = support_points_main.shape[0]
    TN = T * N

    # Flatten time
    X = traj_points.reshape(TN, D)

    # Effective k
    K = min(int(k), M)
    if K <= 0:
        raise ValueError("k must be >= 1 and <= number of support points.")

    # Outputs (do not require grad)
    out_indices = torch.empty((TN, K), dtype=torch.long, device=device)
    out_weights = torch.empty((TN, K), dtype=dtype, device=device)
    nearest_idx = torch.empty((TN,), dtype=torch.long, device=device)

    # Chunking to control memory
    CHUNK = 4096 if TN >= 4096 else TN
    eps = torch.as_tensor(1e-12, dtype=dtype, device=device)
    two_sigma2 = torch.as_tensor(2.0 * (sigma ** 2), dtype=dtype, device=device)

    for start in range(0, TN, CHUNK):
        end = min(start + CHUNK, TN)
        Xc = X[start:end]  # (B, D)

        # MAIN-space distances and KNN
        dist_full = torch.cdist(Xc, support_points_main)                     # (B, M)
        dists, idxs = torch.topk(dist_full, k=K, dim=1, largest=False)       # (B, K)

        # Nearest (for path_indices)
        nearest_idx[start:end] = idxs[:, 0]

        # Gaussian weights in MAIN space (no in-place)
        w = torch.exp(-(dists ** 2) / two_sigma2)                            # (B, K)
        wsum = w.sum(dim=1, keepdim=True)
        w = w / (wsum + eps)

        out_indices[start:end] = idxs
        out_weights[start:end] = w


    # Reshape back to (T, N, K)
    indices = out_indices.view(T, N, K)
    joint_weights = out_weights.view(T, N, K)
    path_indices = nearest_idx.view(T, N)

    return indices, joint_weights, path_indices

def knn_aux_mean_distance_loss(traj, support_main, support_aux, k=20, sigma=1.0):
    """
    traj: (N, Sample, dim)
    support_main: (n_support, dim)
    support_aux: (n_support, dim)
    k: number of nearest neighbors
    sigma: softmax temperature
    alpha: consistency loss weight
    Returns:
        loss: scalar
    """
    N, Sample, dim = traj.shape

    # Reshape traj to (N*Sample, dim)
    traj_flat = traj.reshape(-1, dim)  # (B, dim), B = N*Sample

    # Compute pairwise distances in main space: (B, n_support)
    dists_main = torch.cdist(traj_flat, support_main)

    # Find top-k neighbors (B, k)
    knn_dists, knn_indices = torch.topk(-dists_main, k=k, dim=1)
    knn_dists = -knn_dists

    # Softmax weights: (B, k)
    weights = torch.softmax(-knn_dists / sigma, dim=1)

    # Gather aux neighbors: (B, k, dim)
    aux_knn = support_aux[knn_indices]

    # Compute weighted mean in aux space: (B, dim)
    weights_expanded = weights.unsqueeze(-1)  # (B, k, 1)
    mean_aux_soft = torch.sum(aux_knn * weights_expanded, dim=1)

    # Variance loss: compute distances to weighted center
    aux_diff = aux_knn - mean_aux_soft[:, None, :]  # (B, k, dim)
    aux_dist = torch.norm(aux_diff, dim=-1, p=2)  # (B, k)
   

    '''
    # Consistency loss: adjacent time steps
    mean_aux_soft_time = mean_aux_soft.view(N, Sample, dim)  # (N, Sample, dim)

    # Shifted difference: (N, Sample-1, dim)
    consistency_diff = mean_aux_soft_time[:, 1:, :] - mean_aux_soft_time[:, :-1, :]
    loss_consistency = torch.norm(consistency_diff, dim=-1).mean()
    '''
    # Total loss
    threshold = 0.1
    sharpness = 50
    
    aux_smooth = 1/(1 + torch.exp(-sharpness * (aux_dist - threshold)))
    
    loss = aux_smooth.mean()

    return loss


def gaussian_kernel(window_size=5, std=1.0, device=torch.device("cuda:0")):
    center = window_size // 2
    x = torch.arange(window_size, device=device) - center
    kernel = torch.exp(-0.5 * (x / std)**2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)  # shape (1, 1, window_size)

def smooth_trajectory(trajectory, window_size=5):
    """
    smooth the trajectory (keep the first and last frame unchanged), support efficient vectorization.
    Args:
        trajectory: (N_time, Sample, dim)
        window_size: odd, window size
    Returns:
        smoothed: (N_time, Sample, dim)
    """
    assert window_size % 2 == 1, "window_size must be odd."
    N_time, Sample, dim = trajectory.shape
    half_w = window_size // 2

    # 1. reshape: (Sample * dim, N_time)
    traj = trajectory.permute(1, 2, 0).reshape(Sample * dim, N_time).unsqueeze(1)  # (Batch, 1, Time)

    # 2. padding: keep boundary
    traj_padded = F.pad(traj, pad=(half_w, half_w), mode='replicate')  # (Batch, 1, Time + pad)

    # 3. kernel: same kernel for all batches
    #kernel = torch.ones(1, 1, window_size, device=trajectory.device) / window_size
    kernel = gaussian_kernel(window_size=window_size, std=2.0, device=trajectory.device)
    # 4. apply same kernel to each sequence
    smoothed = F.conv1d(traj_padded, kernel, groups=1)  # (Batch, 1, Time)

    # 5. reshape back
    smoothed = smoothed.squeeze(1).reshape(Sample, dim, N_time).permute(2, 0, 1).contiguous()  # (N_time, Sample, dim)

    return smoothed


import numpy as np

def project_trajectory(secondary_trajectory, total_secondary, threshold=1e-2):
    """
    PyTorch version: Project secondary_trajectory to (D+1) dim, last dim is distance to manifold (tanh mapped).

    Args:
        secondary_trajectory: (T, N, D) torch tensor (can be on GPU)
        total_secondary: (N, D) torch tensor (can be on GPU)
        threshold: distance threshold

    Returns:
        projected_trajectory: (T, N, D+1) torch tensor
    """
    T, N, D = secondary_trajectory.shape

    # Expand dimensions for broadcasting
    traj_expanded = secondary_trajectory.unsqueeze(2)  # (T, N, 1, D)
    manifold_expanded = total_secondary.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)

    # Compute pairwise distances: (T, N, N)
    distances = torch.norm(traj_expanded - manifold_expanded, dim=-1)

    # Get min distance to the manifold for each point: (T, N)
    min_distances, _ = distances.min(dim=-1)

    # Apply threshold and tanh mapping
    distance_values = torch.where(min_distances < threshold, torch.zeros_like(min_distances), torch.tanh(min_distances * 20)) * 1000

    # Concatenate the distance as the (D+1)-th dimension
    projected_trajectory = torch.cat([secondary_trajectory, distance_values.unsqueeze(-1)], dim=-1)

    return projected_trajectory