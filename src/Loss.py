
import torch
import torch.nn as nn
import numpy as np
import math
from torch.distributions import MultivariateNormal
from torchdiffeq import odeint
from geomloss import SamplesLoss
from src.DataLoad import Sampling
from src.MapSpace import map_to_nearest_manifold

def calculate_density_loss(samples, full_data, k=10, temperature=3.0, hinge_value=0.01):
    
    full_data = full_data.detach()
    cdist = torch.cdist(samples, full_data)

    values, _ = torch.topk(cdist, k=k, dim=1, largest=False, sorted=False)
    # Softmin to approximate closest distances smoothly
    softmin_weights = torch.softmax(-values * temperature, dim=1)
    softmin_distance = torch.sum(softmin_weights * values, dim=1)
    values = torch.clamp(softmin_distance - hinge_value, min=0)
    loss = torch.mean(values)
    return loss
'''
def calculate_density_loss(samples, full_data, k=20, hinge_value=0.01):
    cdist = torch.cdist(samples, full_data)

    values, _ = torch.topk(cdist, k, dim=1, largest=False, sorted=False)

    values -= hinge_value
    values[values < 0] = 0
    loss = torch.mean(values)
    return loss
'''

'''
def calculate_density_loss(samples, full_data, k=10, hinge_value=0.25, sigma=0.0002):
    
    Calculate the density loss between the samples and the full data.

    parameters:
        samples: tensor of shape (n_samples, dim) - points from trajectory
        full_data: tensor of shape (n_data, dim) - points on manifold
        k: number of nearest neighbors (default: 10)
        hinge_value: value of the hinge function (default: 0.25)

    returns:
        density_loss: density loss
    
    cdist = torch.cdist(samples, full_data)
    values, _ = torch.topk(cdist, k, dim=1, largest=False, sorted=False)
    # print min max before hinge
    #print(f"min: {values.min().item()}, max: {values.max().item()}")
    values -= hinge_value
    #print(f"min: {values.min().item()}, max: {values.max().item()}")
    values[values < 0] = 0
    
    kernel_values = torch.exp(-values**2 / (2 * sigma**2))
    
    # Convert to loss: points within hinge_value will have zero loss
    # points outside will have exponentially increasing loss
    loss = torch.mean(1 - kernel_values)
    return loss
'''


def get_geodesic_loss(z0_sample, z0, blur=0.05, device='cuda'):
    # Calculate geodesic distance between two points
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, scaling=0.9)
    geodesic_distance = loss(z0_sample, z0)
    return geodesic_distance


def Manifold_reconstruction_loss(
    trajectory: torch.Tensor,
    manifold: torch.Tensor,
    device: torch.device,
    top_percent: float = 0.05,
    k: int = 5,
    sigma: float = 0.5,
    max_distance: float = 0.3
) -> torch.Tensor:
    """
    For a given trajectory and manifold, reconstruct the trajectory by mapping each time step's samples
    to the manifold, then compute the MSE for each sample (averaged over time and dim), and return the mean
    of the top_percent largest errors.

    Args:
        trajectory: Tensor of shape (time, n_sample, dim)
        manifold: Tensor of shape (n_manifold, dim)
        device: torch device
        top_percent: The percentage of samples with the largest error to average
        k: Number of neighbors for mapping
        sigma: Sigma for mapping
        max_distance: Max distance for mapping

    Returns:
        Mean loss of the top_percent largest error samples
    """
    time_length = trajectory.shape[0]
    n_sample = trajectory.shape[1]
    mapped_mainfold_list = []
    dim = trajectory.shape[2]

    flat_samples = trajectory.reshape(-1, dim)  # (time * n_sample, dim)
    
    # Batch mapping
    mapped_points, indices, weights = map_to_nearest_manifold(
        sample_points=flat_samples,
        manifold_points=manifold,
        k=k,
        sigma=sigma,
        max_distance=max_distance
    )
    # Reshape back to (time, n_sample, dim)
    mapped_mainfold_trajectory = mapped_points.reshape(time_length, n_sample, dim)

    # Compute MSE for each sample (mean over time and dim)
    mse_per_trajectory = ((trajectory - mapped_mainfold_trajectory) ** 2).mean(dim=-1).mean(dim=0)  # (n_sample,)
    mse_per_trajectory = mse_per_trajectory * 10000

    # Select top_percent largest errors
    top_k = int(top_percent * mse_per_trajectory.shape[0])
    if top_k < 1:
        top_k = 1
        print("Warning: top_k is 1 !!!")
        breakpoint()
    topk_values, _ = torch.topk(mse_per_trajectory, top_k, largest=True)
    mean_topk = topk_values.mean()
    return mean_topk



class MMD_loss(nn.Module):
    '''
    https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
    Memory-optimized: chunked computation without building full kernel matrices.
    '''
    def __init__(self, kernel_mul=1.0, kernel_num=10):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        # internal chunk size (tune if needed); not exposed to caller
        self._chunk = 2048
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """Original reference implementation (allocates full (n+m)^2 matrix).
        Kept for compatibility; NOT used in forward to avoid peak memory spikes.
        """
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    @staticmethod
    def _pairwise_sq_dists(X, Y):
        """Compute squared Euclidean distances between two blocks X:[a,D], Y:[b,D] without sqrt."""
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        x2 = (X * X).sum(dim=1, keepdim=True)          # [a,1]
        y2 = (Y * Y).sum(dim=1, keepdim=True).T        # [1,b]
        # matmul is memory-efficient vs. expanding to [a,b,D]
        xy = X @ Y.T                                   # [a,b]
        dist2 = x2 + y2 - 2.0 * xy
        # numerical floor at 0
        return dist2.clamp_min_(0.0)

    def _estimate_bandwidth(self, total, kernel_mul, kernel_num, fix_sigma):
        """Estimate bandwidth (sigma^2) like the reference code, but in chunks to reduce memory."""
        if fix_sigma is not None:
            return fix_sigma

        n = total.size(0)
        if n <= 1:
            # fallback to small positive to avoid divide-by-zero
            return total.new_tensor(1.0)

        chunk = self._chunk
        # Accumulate sum of all pairwise squared distances (including diagonal zeros),
        # then divide by n^2 - n as in the reference.
        sum_dist2 = total.new_zeros(())
        with torch.no_grad():  # bandwidth is a constant; no gradient needed
            for i in range(0, n, chunk):
                Xi = total[i:i+chunk]
                for j in range(0, n, chunk):
                    Yj = total[j:j+chunk]
                    d2 = self._pairwise_sq_dists(Xi, Yj)  # [ci, cj]
                    sum_dist2 = sum_dist2 + d2.sum()

            bandwidth = sum_dist2 / (n * n - n)

        # Scale to create the bandwidth list center, consistent with the reference logic
        bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
        # ensure positive scalar tensor
        return bandwidth.clamp_min(1e-12)

    def forward(self, source, target):
        """
        Chunked MMD with multi-kernel RBF (sum of exponentials over bandwidth list).
        Does NOT materialize the full kernel matrices.

        Returns:
            loss (Tensor): scalar MMD estimate using the same mean formula as the reference.
        """
        X = source
        Y = target
        nx = int(X.size(0))
        ny = int(Y.size(0))
        device = X.device
        dtype = X.dtype

        # Build the bandwidth list (sigma^2) using the same statistic as the reference code
        total = torch.cat([X, Y], dim=0)
        base_bw = self._estimate_bandwidth(total, self.kernel_mul, self.kernel_num, self.fix_sigma)  # scalar tensor
        # create list of bandwidths like: bw * kernel_mul**i
        bw_list = [base_bw * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        # Accumulators for sums of kernel values (we will divide by counts to get means)
        sum_xx = X.new_zeros(())   # sum over all entries of Kxx
        sum_yy = X.new_zeros(())   # sum over all entries of Kyy
        sum_xy = X.new_zeros(())   # sum over all entries of Kxy

        cx = self._chunk
        cy = self._chunk

        # --- Kxx ---
        for i in range(0, nx, cx):
            Xi = X[i:i+cx]  # [ci, D]
            for j in range(0, nx, cx):
                Xj = X[j:j+cx]  # [cj, D]
                d2 = self._pairwise_sq_dists(Xi, Xj)  # [ci, cj]
                # sum over multiple bandwidths without storing matrices
                ksum = X.new_zeros(())
                for bw in bw_list:
                    ksum = ksum + torch.exp(-d2 / bw)
                sum_xx = sum_xx + ksum.sum()

        # --- Kyy ---
        for i in range(0, ny, cy):
            Yi = Y[i:i+cy]
            for j in range(0, ny, cy):
                Yj = Y[j:j+cy]
                d2 = self._pairwise_sq_dists(Yi, Yj)
                ksum = Y.new_zeros(())
                for bw in bw_list:
                    ksum = ksum + torch.exp(-d2 / bw)
                sum_yy = sum_yy + ksum.sum()

        # --- Kxy ---
        for i in range(0, nx, cx):
            Xi = X[i:i+cx]
            for j in range(0, ny, cy):
                Yj = Y[j:j+cy]
                d2 = self._pairwise_sq_dists(Xi, Yj)
                ksum = X.new_zeros(())
                for bw in bw_list:
                    ksum = ksum + torch.exp(-d2 / bw)
                sum_xy = sum_xy + ksum.sum()

        # Convert sums to means exactly like the reference:
        # mean(XX) + mean(YY) - mean(XY) - mean(YX)  == mean(XX) + mean(YY) - 2*mean(XY)
        mean_xx = sum_xx / (nx * nx)
        mean_yy = sum_yy / (ny * ny)
        mean_xy = sum_xy / (nx * ny)
        loss = mean_xx + mean_yy - 2.0 * mean_xy
        return loss

def compute_kl_divergence(z0_sample, z0, bandwidth=0.02, grid_size=40):
    """
    Compute KL divergence between two distributions in arbitrary dimensions
    with guaranteed non-negativity
    
    Args:
        z0_sample: Sample points from first distribution
        z0: Sample points from second distribution 
        bandwidth: Bandwidth for KDE
        grid_size: Number of grid points per dimension (smaller for high dims)
    """
    # Get dimension of the data
    dim = z0.shape[1]
    
    
    # Ensure inputs are PyTorch tensors on same device
    device = z0.device
    z0_sample = z0_sample.to(device)
    
    # Calculate adaptive bandwidth
    n_sample = z0_sample.shape[0]
    n_true = z0.shape[0]
    

    # Calculate data scales using robust statistics
    def get_adaptive_bandwidth(data):
        # Use median absolute deviation instead of std for robustness
        median_per_dim = torch.median(data, dim=0)[0]
        abs_dev = torch.abs(data - median_per_dim.unsqueeze(0))
        mad_per_dim = torch.median(abs_dev, dim=0)[0] * 1.4826  # scale factor for normal distribution
        
        # Use max of MAD and a minimum value to avoid zero bandwidth
        scales = torch.maximum(mad_per_dim, torch.ones_like(mad_per_dim) * 1e-3)
        
        # Scott's rule with dimension adjustment
        h = scales * (data.shape[0] ** (-1.0 / (dim + 4)))
        
        # Scale up for higher dimensions
        if dim > 2:
            h = h * (1.0 + 0.1 * (dim - 2))
        
        return h
    
    bandwidth_sample = get_adaptive_bandwidth(z0_sample)
    bandwidth_true = get_adaptive_bandwidth(z0)
    
    # set the bandwidth to the maximum of the two
    bandwidth_vec = torch.maximum(bandwidth_sample, bandwidth_true)
    
    # Additional manual adjustment based on input bandwidth parameter
    bandwidth_vec = bandwidth_vec * (bandwidth / 0.05)
    
    # set the minimum bandwidth
    bandwidth_vec = torch.maximum(bandwidth_vec, torch.tensor(1e-2, device=device))
    
    # create evaluation points
    n_eval = min(grid_size**min(dim, 2), 2000)  # limit the number of evaluation points
    
    # use mixed point sampling to improve stability
    # draw samples from two distributions
    idx_sample = torch.randperm(n_sample)[:n_eval//4]
    idx_true = torch.randperm(n_true)[:n_eval//4]
    
    points_from_sample = z0_sample[idx_sample]
    points_from_true = z0[idx_true]
    
    # determine the boundary of the region
    min_vals = torch.min(torch.cat([z0_sample, z0]), dim=0)[0] - 2.0 * bandwidth_vec
    max_vals = torch.max(torch.cat([z0_sample, z0]), dim=0)[0] + 2.0 * bandwidth_vec
    
    # create random points
    n_random = n_eval - points_from_sample.shape[0] - points_from_true.shape[0]
    random_points = min_vals.unsqueeze(0) + torch.rand(n_random, dim, device=device) * (max_vals - min_vals).unsqueeze(0)
    
    # merge evaluation points
    grid_points = torch.cat([points_from_sample, points_from_true, random_points], dim=0)
    
    # improved KDE implementation
    def robust_kde(x, data, h_vec):
        n_data = data.shape[0]
        results = torch.zeros(x.shape[0], device=device)
        
        # batch calculation to avoid memory problem
        batch_size = min(1000, x.shape[0])
        
        for i in range(0, x.shape[0], batch_size):
            end_idx = min(i + batch_size, x.shape[0])
            batch_points = x[i:end_idx]
            
            # calculate the kernel distance
            diff = batch_points.unsqueeze(1) - data.unsqueeze(0)  # [batch, n_data, dim]
            scaled_diff = diff / h_vec.unsqueeze(0).unsqueeze(0)  # Scale by bandwidth
            sq_dist = torch.sum(scaled_diff**2, dim=2)  # [batch, n_data]
            
            # Gaussian kernel
            kernel = torch.exp(-0.5 * sq_dist)
            
            # correct normalization
            normalization = torch.prod(h_vec) * torch.sqrt(torch.tensor(2 * torch.pi, device=device)) ** dim
            results[i:end_idx] = torch.sum(kernel / normalization, dim=1) / n_data
        
        return results
    
    # calculate the density
    p_sample = robust_kde(grid_points, z0_sample, bandwidth_vec)
    p_true = robust_kde(grid_points, z0, bandwidth_vec)
    
    # ensure the density is positive
    eps = 1e-10
    p_sample = torch.maximum(p_sample, torch.tensor(eps, device=device))
    p_true = torch.maximum(p_true, torch.tensor(eps, device=device))
    
    # use direct MC estimation
    # KL(p_sample || p_true) ≈ E_p_sample[log(p_sample/p_true)]
    weight_sum = torch.sum(p_sample)
    norm_weights = p_sample / weight_sum  # normalize the weights
    
    # use the variant of Jensen's inequality to avoid numerical problems
    log_ratio = torch.log(p_sample / p_true)
    
    # use weighted average to calculate KL divergence
    kl_div = torch.sum(norm_weights * log_ratio)
    
    # positive kl
    kl_div = torch.maximum(kl_div, torch.tensor(0.0, device=device))
    
    del p_sample, p_true, norm_weights, log_ratio, weight_sum, grid_points, random_points, points_from_sample, points_from_true
    
    return kl_div


def kde_relative_log_pdf(real_data: torch.Tensor, query_data: torch.Tensor, bandwidth: float):
    """
    Calculate the relative log probability density of query_data with respect to real_data.
    
    Args:
        real_data: Reference point cloud [N, D]
        query_data: Query point cloud [M, D]
        bandwidth: Bandwidth parameter for KDE
        
    Returns:
        log_pdf: Log probability density values for each point in query_data [M]
    """
    N, D = real_data.shape
    M, D_query = query_data.shape
    assert D == D_query, "The dimension D of real_data and query_data must match"
    
    # Convert bandwidth to tensor
    bandwidth_tensor = torch.tensor(bandwidth, dtype=torch.float32)
    pi_tensor = torch.tensor(np.pi, dtype=torch.float32)
    
    # Calculate normalization constant
    denom = (2 * pi_tensor) ** (D / 2) * (bandwidth_tensor ** D)
    log_const = -torch.log(denom)
    
    # Calculate squared distances between query_data and real_data
    diff = query_data.unsqueeze(1) - real_data.unsqueeze(0)  # [M, N, D]
    dist_sq = diff.pow(2).sum(dim=2)  # [M, N]
    
    # Calculate kernel values
    kernel_vals = torch.exp(-dist_sq / (2 * bandwidth_tensor ** 2))  # [M, N]
    
    # Calculate log probability density
    density = kernel_vals.mean(dim=1)  # [M]
    log_pdf = torch.log(density) + log_const
    
    return log_pdf

def MultiNormal_density(point_cloud_a, point_cloud_b, sigma=5, device='cuda'):
    """
    Calculate the probability density of point_cloud_b with respect to point_cloud_a.
    
    Args:
        point_cloud_a (torch.Tensor): point cloud a (num_points_a, dimension)。
        point_cloud_b (torch.Tensor): point cloud b (num_points_b, dimension)。
        sigma (float): the value of the diagonal elements of the covariance matrix.
        device (torch.device): the device for calculation (CPU or GPU).
        
    Returns:
        torch.Tensor: the probability density of point cloud b with respect to point cloud a, shape (num_points_b,)。
    """
   
    point_cloud_a = point_cloud_a.to(device)
    point_cloud_b = point_cloud_b.to(device)
    
    dimension = point_cloud_a.shape[1]
    sigma_matrix = sigma * torch.eye(dimension).to(device)
    
    log_densities = torch.full((point_cloud_b.shape[0],), float('-inf'), device=device)
    
    for i in range(point_cloud_a.shape[0]):
        gaussian = MultivariateNormal(point_cloud_a[i], sigma_matrix)
        log_probs = gaussian.log_prob(point_cloud_b)  
        log_densities = torch.logaddexp(log_densities, log_probs)  
    
    log_densities -= torch.log(torch.tensor(point_cloud_a.shape[0], device=device, dtype=torch.float32))
    
    return log_densities


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

def compute_log_density_with_precomputation(point_cloud_a, point_cloud_b, weights, means, precisions, device, batch_size=1024, eps=1e-16):
    """
    Compute log density of points with respect to GMM with gradient support and numerical stability.
    Uses memory-efficient batching for large point clouds.
    
    Args:
        point_cloud_a: Reference point cloud [N_a, D]
        point_cloud_b: Query point cloud [N_b, D]
        weights, means, precisions: GMM parameters
        device: Computation device
        batch_size: Batch size for memory-efficient processing
        eps: Small constant for numerical stability
        
    Returns:
        log_density: Log probability density values [N_b]
    """
    B, D = point_cloud_b.shape
    K = weights.shape[0]
    
    # Ensure weights sum to 1 and are positive
    weights = torch.maximum(weights, torch.tensor(eps, device=device))
    weights = weights / weights.sum()
    
    # Ensure precision matrices are well-conditioned
    min_eigenval = 1e-6
    identity = torch.eye(D, device=device).unsqueeze(0).repeat(K, 1, 1)
    precisions = precisions + min_eigenval * identity
    
    # Process in batches to save memory
    log_densities = []
    
    for i in range(0, B, batch_size):
        end_idx = min(i + batch_size, B)
        batch_points = point_cloud_b[i:end_idx]
        
        # Reshape for broadcasting
        x = batch_points.unsqueeze(1)  # [batch, 1, D]
        mu = means.unsqueeze(0)        # [1, K, D]
        prec = precisions.unsqueeze(0) # [1, K, D, D]
        
        # Calculate difference vectors
        diff = x - mu  # [batch, K, D]
        
        # Compute Mahalanobis distance with numerical stability
        maha_dist = torch.sum(
            torch.matmul(diff.unsqueeze(2), prec) * diff.unsqueeze(2),
            dim=-1
        ).squeeze(-1)  # [batch, K]
        
        # Clip extremely large distances
        max_dist = 1e3
        maha_dist = torch.minimum(maha_dist, torch.tensor(max_dist, device=device))
        
        # Compute log determinant with stability check
        log_det = torch.logdet(precisions)  # [K]
        # Handle negative determinants (should not happen with the regularization above)
        log_det = torch.where(log_det > -1e10, log_det, 
                            torch.tensor(-1e10, device=device))
        
        # Compute log probabilities
        log_probs = (
            torch.log(weights + eps)[None, :] +  # [1, K]
            0.5 * log_det[None, :] -            # [1, K] 
            0.5 * D * np.log(2*np.pi) -         # scalar
            0.5 * maha_dist                     # [batch, K]
        )
        
        # Numerical stability for logsumexp
        max_log_prob = torch.max(log_probs, dim=1, keepdim=True)[0]
        log_probs = log_probs - max_log_prob
        
        # Use numerically stable logsumexp
        batch_log_density = max_log_prob.squeeze(1) + \
                          torch.log(torch.sum(torch.exp(log_probs), dim=1) + eps)
        
        # Handle extreme values
        min_log_density = torch.tensor(-1e3, device=device)
        batch_log_density = torch.maximum(batch_log_density, min_log_density)
        
        log_densities.append(batch_log_density)
    
    # Combine all batch results
    log_density = torch.cat(log_densities, dim=0)  # [B]
    
    return log_density

def compute_pdf_lossMSE(z_initial_primal_sample: torch.Tensor,
                    z_terminal_primal_sample: torch.Tensor,
                    logp_diff_initial: torch.Tensor,
                    logp_diff_terminal: torch.Tensor,
                    initial_mixture: torch.Tensor,
                    terminal_mixture: torch.Tensor,
                    mass_ratio: torch.Tensor):
    """
    Compute density-based loss between two distributions with gradient support.
    
    backward
    
    initial_time (t1) -----------------------> terminal_time (t0)
                          neural network
    z_terminal_sample , logp_diff_terminal are output of neural network
    
    logp_predict = terminal_mixture.log_prob(z_terminal_primal_sample).to(device) - logp_diff_terminal.view(-1)
    logp_true = initial_mixture.log_prob(z_initial_primal_sample).type(torch.float32).to(device) + torch.log(mass_ratio)

    Args:
        z_intial_primal_sample: Initial time query points
        z_terminal_primal_sample: Terminal time query points
        logp_diff_initial: Log of Jacobian determinant for initial time
        initial_mixture: GMM parameters for initial time
        terminal_mixture: GMM parameters for terminal time
        mass_ratio: Mass ratio between initial and terminal time
    Returns:
        pdf_diff: mean loss between predicted and true log densities, which is kl divergence
    """
    mse = nn.MSELoss()
    device = z_initial_primal_sample.device
    
    # Ensure data is on the correct device
    z_initial_primal_sample = z_initial_primal_sample.to(device)
    logp_diff_terminal = logp_diff_terminal.to(device)

        
    logp_initial_predicted = terminal_mixture.log_prob(z_terminal_primal_sample).to(device) - logp_diff_terminal.view(-1)
    logp_initial = initial_mixture.log_prob(z_initial_primal_sample).type(torch.float32).to(device) + torch.log(mass_ratio)
        
    # Compute MSE between log densities
    pdf_diff = mse(logp_initial, logp_initial_predicted)
    del z_initial_primal_sample, logp_diff_terminal, logp_initial, logp_initial_predicted
    
    return pdf_diff


def compute_pdf_lossNegLog(z_initial_primal_sample: torch.Tensor,
                    z_terminal_primal_sample: torch.Tensor,
                    logp_diff_initial: torch.Tensor,
                    logp_diff_terminal: torch.Tensor,
                    initial_mixture: torch.Tensor,
                    terminal_mixture: torch.Tensor,
                    mass_ratio: torch.Tensor,
                    ):
    """
    Compute density-based loss between two distributions with gradient support.
    
    Args:
        z_intial_primal_sample: Initial time query points
        z_terminal_primal_sample: Terminal time query points
        logp_diff_initial: Log of Jacobian determinant for initial time
        initial_mixture: GMM parameters for initial time
        terminal_mixture: GMM parameters for terminal time
        mass_ratio: Mass ratio between initial and terminal time
    Returns:
        pdf_diff: mean loss between predicted and true log densities, which is kl divergence
    """
    device = z_initial_primal_sample.device
    
    # Ensure data is on the correct device
    z_initial_primal_sample = z_initial_primal_sample.to(device)
    logp_diff_terminal = logp_diff_terminal.to(device)

    logp_initial_predicted = terminal_mixture.log_prob(z_terminal_primal_sample).to(device) - logp_diff_terminal.view(-1)
    logp_initial_predicted = logp_initial_predicted + torch.log(mass_ratio)

    pdf_loss = -logp_initial_predicted.mean()
    
    del z_initial_primal_sample, logp_diff_terminal, logp_initial_predicted
    
    return pdf_loss


def relative_log_pdf(mixture: torch.Tensor,
                     query_points: torch.Tensor,
                     device: torch.device,
                    ):
    """
    Compute relative log probability density of query_points with respect to mixture.
    
    Args:
        mixture: GMM parameters
        query_points: Query points
        device: Computation device
    Returns:
        log_pdf: Log probability density values for each point in query_points [M]
    """
    
    # Ensure data is on the correct device
    query_points = query_points.to(device)

    log_pdf = mixture.log_prob(query_points)

    pdf_loss = -log_pdf.mean()
    
    return pdf_loss