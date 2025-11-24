"""train_utils module"""
import numpy as np
from anndata import AnnData
import torch
from torch import nn
from torchdiffeq import odeint

def create_batch(obs, velo, times, t0_min, t0_max, min_delta_time, max_delta_time, max_points_num):
    t_max = 1
    n_points = obs.shape[0]

    index_np = np.arange(0, n_points, 1, dtype=int)
    times_np = np.linspace(0, t_max, num=n_points)
    if t0_min == t0_max:
        t0 = t0_min
    else:
        t0 = np.random.uniform(t0_min, t0_max)
    t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)

    idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])
    obs_ = obs[idx]
    ts_ = times[idx]
    vs_ = velo[idx]
    return obs_, ts_, vs_

# ---- Definition: get_lr_by_em_num ----

def get_lr_by_em_num(em_num, em_max=10, lr_start=0.1, lr_end=0.6):
    """Linearly increase lr from lr_start to lr_end as em_num goes from 0 to em_max-1."""
    if em_num < 0:
        em_num = 0
    if em_num > em_max - 1:
        em_num = em_max - 1
    lr = lr_start + (lr_end - lr_start) * em_num / (em_max - 1)
    return lr

# ---- Definition: cal_true_velocity ----

def cal_true_velocity(adata_true, t_max=25):
    n_obs = adata_true.shape[0]
    delta_t = t_max / n_obs
    delta_t = round(delta_t, 4)

    alpha_data = np.zeros((adata_true.shape[0], adata_true.shape[1]))
    for i in range(adata_true.shape[0]):
        for j in range(adata_true.shape[1]):
            alpha_data[i, j] = adata_true.X[i, j]

    cell_velo_dx = np.zeros_like(alpha_data)
    for i in range(0, alpha_data.shape[0]):
        if i == 0:
            cell_velo_dx[i] = (alpha_data[1] - alpha_data[0]) /delta_t
        else:
            cell_velo_dx[i] = (alpha_data[i] - alpha_data[i - 1]) / delta_t
    return cell_velo_dx


# ---- Definition: LinearWeightedMSE ----


### u_hat and u_x`
# def update_us(adata, trajectory_embeddings, lr_us=0.3):
#     """Update u and s in adata using the fitted circle with learning rate lr_us."""
#     adata_updated = AnnData(trajectory_embeddings)
#     adata_updated.layers['spliced'] = np.zeros_like(trajectory_embeddings)
#     adata_updated.layers['unspliced'] = np.zeros_like(trajectory_embeddings)
#     adata_updated.obs['velocity_pseudotime'] = adata.obs['velocity_pseudotime'].copy()
#     for cell in range(trajectory_embeddings.shape[0]):
#         for gene in range(trajectory_embeddings.shape[1]):
#             u_hat = adata.layers['fit_unspliced'][cell,gene]
#             s_hat = adata.layers['fit_spliced'][cell,gene]
#             u = adata.layers['unspliced'][cell,gene]
#             s = adata.layers['spliced'][cell,gene]
#             x_modified = trajectory_embeddings[cell,gene]
#             u_x = x_modified/(adata.X[cell,gene] + 1e-2) * u
#             s_x = x_modified/(adata.X[cell,gene] + 1e-2) * s
#             u = u_x + lr_us * (u_hat - u_x)
#             s = s_x + lr_us * (s_hat - s_x)

#             adata_updated.layers['spliced'][cell,gene] = s
#             adata_updated.layers['unspliced'][cell,gene] = u

    
#     return adata_updated

def update_us(adata, trajectory_embeddings,  lr_us=0.3, window_size=10, step_size=5, center_lr=0.2):
    """Update u and s in adata using the mean center"""
    adata_updated = AnnData(trajectory_embeddings)
    adata_updated.layers['spliced'] = np.zeros_like(trajectory_embeddings)
    adata_updated.layers['unspliced'] = np.zeros_like(trajectory_embeddings)
    # adata_updated.obs['velocity_pseudotime'] = adata.obs['velocity_pseudotime'].copy()
    
    # 获取spliced和unspliced数据
    s_data = adata.layers['spliced'].copy()  # spliced数据
    u_data = adata.layers['unspliced'].copy()  # unspliced数据
    
    n_cells = adata.n_obs

    # 计算滑动窗口的中心点
    window_centers = []
    window_indices = []
    
    # 按照细胞在轨迹上的顺序（需要先对细胞排序）
    # 这里假设细胞已经按照轨迹顺序排列，如果没有需要先排序
    for start_idx in range(0, n_cells - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # 计算窗口内spliced和unspliced的均值
        s_window_center = s_data[start_idx:end_idx].mean(axis=0)
        u_window_center = u_data[start_idx:end_idx].mean(axis=0)
        
        window_centers.append((s_window_center, u_window_center))
        window_indices.append((start_idx, end_idx))
    
    # 处理最后一个窗口
    if window_indices and window_indices[-1][1] < n_cells:
        start_idx = max(0, n_cells - window_size)
        s_window_center = s_data[start_idx:].mean(axis=0)
        u_window_center = u_data[start_idx:].mean(axis=0)
        window_centers.append((s_window_center, u_window_center))
        window_indices.append((start_idx, n_cells))
    
    # 对每个细胞，找到最近的两个窗口 update
    for i in range(n_cells):
        for window in range(len(window_indices)):
            start_idx, end_idx = window_indices[window]
            if start_idx <= i < end_idx:
                s_center, u_center = window_centers[window]
                # 更新spliced
                s_data[i] += center_lr * (s_center - s_data[i])
                # 更新unspliced
                u_data[i] += center_lr * (u_center - u_data[i])
        
    # 更新adata对象的layers
    adata_updated.layers['spliced'] = s_data
    adata_updated.layers['unspliced'] = u_data
    
    return adata_updated


class ODEFunc(nn.Module):
    def __init__(self, input_dim):
        super(ODEFunc, self).__init__()
        self.linear1 = nn.Linear(input_dim, 4000)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4000, 4000)
        self.linear3 = nn.Linear(4000, input_dim)
    
    def forward(self, t, x):
        y = self.linear1(x)
        y = self.relu(y)
        y = self.linear2(y) + y
        y = self.relu(y)
        return self.linear3(y)
       
options = {
    'step_size': 0.001
}

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
    
    def forward(self, y0, t):
        out = odeint(self.odefunc, y0, t, rtol=1e-7, atol=1e-9)
        return out
    
# ---- Definition: update_us ----
class LinearWeightedMSE(nn.Module):
    def __init__(self, min_weight=0.5):
        super().__init__()
        self.min_weight = min_weight  
    
    def forward(self, input, target):
        squared_diff = torch.log1p((input - target) ** 2 + 1).sum(dim=-1)
        
        batch_size = input.size(0)
        # 线性从1.0衰减到min_weight
        weights = torch.linspace(1.0, self.min_weight, steps=batch_size, device=input.device)
        
        weighted_loss = weights * squared_diff
        
        return weighted_loss.mean()