import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path

# =====================================================
# Helper functions
# =====================================================

def build_geodesic_distance_graph(support_points: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Build approximate geodesic distance matrix using k-NN graph (undirected).

    Parameters
    ----------
    support_points : torch.Tensor (M, D)
        Discrete points lying on the manifold (auxiliary space).
    k : int, optional
        Number of neighbours in the k-NN graph, by default 20.

    Returns
    -------
    torch.Tensor (M, M)
        Symmetric matrix with estimated geodesic distances (float32).
    """
    support_np = support_points.detach().cpu().numpy()

    # k-NN graph in Euclidean metric
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(support_np)
    distances, indices = nbrs.kneighbors(support_np)

    M = support_np.shape[0]
    adjacency = np.full((M, M), np.inf, dtype=np.float64)

    for i in range(M):
        for j in range(1, k):  # skip self-loop (j == 0)
            nb = indices[i, j]
            d = distances[i, j]
            adjacency[i, nb] = d
            adjacency[nb, i] = d

    # Shortest path on sparse graph ≈ geodesic distance
    geodesic_np = shortest_path(adjacency, directed=False)

    return torch.from_numpy(geodesic_np).to(support_points.device).float()


def soft_map_to_support(traj_points: torch.Tensor, support_points: torch.Tensor, sigma: float = 0.05):
    """Softly assign every trajectory point to support set using Gaussian kernel.

    Parameters
    ----------
    traj_points : torch.Tensor (T, N, D)
    support_points : torch.Tensor (M, D)
    sigma : float, optional
        Bandwidth of Gaussian, by default 0.05.

    Returns
    -------
    mapped : torch.Tensor (T, N, D)
        Softly mapped coordinates in the same ambient space.
    weights : torch.Tensor (T, N, M)
        Soft assignment weights over support points (row-stochastic).
    """
    T, N, D = traj_points.shape
    M = support_points.shape[0]

    traj_flat = traj_points.reshape(T * N, D)  # (T*N, D)
    support_flat = support_points.unsqueeze(0)  # (1, M, D)

    # Pairwise Euclidean distances
    dists = torch.norm(traj_flat.unsqueeze(1) - support_flat, dim=2)  # (T*N, M)
    weights = torch.exp(-dists ** 2 / (2.0 * sigma ** 2))  # Gaussian kernel
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    mapped_flat = torch.matmul(weights, support_points)  # (T*N, D)

    mapped = mapped_flat.reshape(T, N, D)
    weights = weights.reshape(T, N, M)
    return mapped, weights


def soft_geodesic_loss(weights: torch.Tensor, geodesic_matrix: torch.Tensor) -> torch.Tensor:
    """Differentiable geodesic loss between consecutive time steps.

    Expected distance E[d(i,j)] where i ~ w_t, j ~ w_{t+1}.
    """
    T, N, M = weights.shape
    loss = 0.0
    for t in range(T - 1):
        w_t = weights[t]       # (N, M)
        w_tp1 = weights[t + 1] # (N, M)
        # Einsum implements \sum_{n,i,j} w_t[n,i] * d[i,j] * w_tp1[n,j]
        loss += torch.einsum("ni,ij,nj->", w_t, geodesic_matrix, w_tp1)
    return loss


def map_to_nearest_manifold_multi_space(
    traj_points: torch.Tensor,
    support_points_main: torch.Tensor,
    support_points_aux: torch.Tensor,
    k: int = 30,
    sigma_aux: float = 0.05,
    alpha_main: float = 1.0,
):
    """Vectorised k-NN mapping shared across main & auxiliary supports.

    Parameters
    ----------
    traj_points : (T, N, D) torch.Tensor
    support_points_main / aux : (M, D) torch.Tensor  
        *index correspondence assumed*: the i-th row in both tensors
        represents the same anchor point on the discrete manifold but
        embedded in different spaces.
    k : int
        Number of nearest anchors retained per trajectory point.
    sigma_aux : float
        Bandwidth for the Gaussian kernel when converting distances to
        weights.
    alpha_main : float (unused for now, kept for interface compatibility)

    Returns
    -------
    indices : (T*N, k) torch.LongTensor
        Indices of the k nearest anchors (shared for main & aux).
    weights : (T*N, k) torch.Tensor (row-stochastic)
        Soft assignment weights corresponding to those anchors.
    """
    T, N, D = traj_points.shape
    M = support_points_main.shape[0]

    traj_flat = traj_points.reshape(T * N, D)  # (T*N, D)

    # Pairwise Euclidean distances (vectorised, no Python loops)
    dists = torch.cdist(traj_flat, support_points_main)  # (T*N, M)

    # k nearest anchors in the MAIN space
    dists_k, indices = torch.topk(dists, k=k, largest=False, sorted=False)  # (T*N, k)

    # Gaussian kernel weights on auxiliary bandwidth (row-wise normalisation)
    weights = torch.exp(-dists_k ** 2 / (2.0 * sigma_aux ** 2))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    return indices, weights


# 更新后的、完全向量化的 geodesic loss（去除时间循环）

def soft_geodesic_loss(weights: torch.Tensor, geodesic_matrix: torch.Tensor) -> torch.Tensor:
    """Vectorised expected geodesic distance between consecutive steps.

    weights : (T, N, M)  –  row-stochastic over anchors.
    geodesic_matrix : (M, M) pre-computed pairwise distances on manifold.
    """
    w_t = weights[:-1]  # (T-1, N, M)
    w_tp1 = weights[1:]  # (T-1, N, M)
    # Einstein summation implements Σ_{t,n,i,j} w_t[t,n,i] * d[i,j] * w_{t+1}[t,n,j]
    return torch.einsum("tni,ij,tnj->", w_t, geodesic_matrix, w_tp1)

# =====================================================
# 1. Generate trajectory in the primary (ambient) space
# =====================================================

# Trajectory parameters
T = 20  # number of time steps
N = 1   # number of duplicated particles/paths (can be >1)
A = 0.05
f = 30

# Time axis and noisy sinusoidal y-component
_t = torch.linspace(0, 1, T).unsqueeze(1).repeat(1, N)
base_y = 0.505 * torch.ones_like(_t)
oscillation = A * torch.sin(2 * torch.pi * f * _t + torch.rand(1) * 2 * torch.pi)
point_noise = 0.03 * torch.randn(T, N)
_y = base_y + point_noise + oscillation

# (T, N, 2) trajectory, keep everything float32 for consistency
trajectory = torch.stack([_t, _y], dim=2).float()

start_point = trajectory[0].clone().detach()
end_point = trajectory[-1].clone().detach()

# =====================================================
# 2. Create support sets (main & auxiliary)
# =====================================================

M = 500  # number of support points (only used if you don't supply your own)

if "support_points_main" not in globals():
    support_points_main = torch.rand(M, 2).float()
if "support_points_aux" not in globals():
    support_points_aux = torch.rand(M, 2).float()

# 保证 dtype 一致
support_points_main = support_points_main.float()
support_points_aux = support_points_aux.float()

# Pre-compute geodesic distance matrix on auxiliary support (used for loss)
geodesic_dist = build_geodesic_distance_graph(support_points_aux, k=10)

# =====================================================
# 3. Optimisable middle control points
# =====================================================

middle_points = torch.nn.Parameter(trajectory[1:-1].clone())

optimizer = torch.optim.Adam([middle_points], lr=2e-3)

# =====================================================
# 4. Training loop
# =====================================================

epochs = 1000
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()

    # 构造完整轨迹 (T, N, 2)
    traj_full = torch.cat([start_point.unsqueeze(0), middle_points, end_point.unsqueeze(0)], dim=0)
    T_current = traj_full.shape[0]

    # ---------- 映射到 k 个离散 anchor（共享索引） ----------
    indices_A, weights_A = map_to_nearest_manifold_multi_space(
        traj_full, support_points_main, support_points_aux, k=30, sigma_aux=0.05
    )  # indices_A : (T*N, k)

    # 依据权重得到主/次空间轨迹
    mapped_primal_flat = support_points_main[indices_A]           # (T*N, k, 2)
    mapped_secondary_flat = support_points_aux[indices_A]         # (T*N, k, 2)

    primal_flat = torch.sum(mapped_primal_flat * weights_A.unsqueeze(-1), dim=2)      # (T*N, 2)
    secondary_flat = torch.sum(mapped_secondary_flat * weights_A.unsqueeze(-1), dim=2)  # (T*N, 2)

    primal_traj = primal_flat.view(T_current, N, 2)      # (T, N, 2)
    secondary_traj = secondary_flat.view(T_current, N, 2)

    # ---------- 构造完整权重矩阵以计算 geodesic loss ----------
    M = support_points_main.shape[0]
    weights_full = torch.zeros((T_current * N, M), device=traj_full.device)
    weights_full.scatter_(1, indices_A, weights_A)  # (T*N, M)
    weights_full = weights_full.view(T_current, N, M)

    # geodesic loss
    geo_loss = soft_geodesic_loss(weights_full, geodesic_dist)

    # 反向 & 更新
    geo_loss.backward()
    optimizer.step()

    loss_history.append(geo_loss.item())

    if (epoch + 1) % 200 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Geodesic loss: {geo_loss.item():.4f}")

        # 可视化主空间轨迹
        primal_np = primal_traj.detach().cpu().numpy().reshape(T_current, N, 2)
        supp_main_np = support_points_main.cpu().numpy()

        plt.figure(figsize=(7, 6))
        plt.scatter(supp_main_np[:, 0], supp_main_np[:, 1], s=8, alpha=0.3, label="Support (main)")
        for n in range(N):
            plt.plot(primal_np[:, n, 0], primal_np[:, n, 1], "-b", lw=1.5, label="Primal traj" if n == 0 else None)
        plt.scatter(primal_np[0, 0, 0], primal_np[0, 0, 1], c="red", s=50, label="Start")
        plt.scatter(primal_np[-1, 0, 0], primal_np[-1, 0, 1], c="green", s=50, label="End")
        plt.title(f"Primary trajectory @ epoch {epoch + 1}")
        plt.legend()
        plt.grid(True)
        plt.show()

# =====================================================
# 5. Plot loss curve
# =====================================================

plt.figure(figsize=(7, 5))
plt.plot(loss_history, label="Geodesic loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss curve")
plt.grid(True)
plt.legend()
plt.show() 