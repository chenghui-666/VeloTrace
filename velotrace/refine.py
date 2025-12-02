import numpy as np
import torch
import pandas as pd
from scipy.sparse import issparse
from anndata import AnnData
from .utils import set_seed

set_seed(4)

# refine process
def update_usxv(adata, loaded_t, window_size=7, lr=0.1):

    adata_new = AnnData(adata.X.copy())
    adata_new.var_names = adata.var_names.copy()
    adata_new.obs['velocity_pseudotime'] = loaded_t.detach().cpu().numpy()
    adata_new.obs['true_t'] = adata.obs['true_t'].copy()
    adata_new.layers['unspliced'] = adata.layers['unspliced'].copy()
    adata_new.layers['spliced'] = adata.layers['spliced'].copy()

    # 将数据按照时间重新排列（注意保证一致性）
    order = np.argsort(adata_new.obs["velocity_pseudotime"].to_numpy())
    adata_new = adata_new[order].copy()
    print(adata_new.obs["velocity_pseudotime"])

    # 设置参数
    window_size = window_size  # 滑动窗口大小（包括中心细胞）
    half_window = window_size // 2  # 每边取3个邻居（总共7个点）
    learning_rate = lr  # 学习率

    # 获取原始u和s（处理稀疏矩阵情况）
    u = adata_new.layers['unspliced'] if issparse(adata_new.layers['unspliced']) else np.array(adata.layers['unspliced'])
    s = adata_new.layers['spliced'] if issparse(adata_new.layers['spliced']) else np.array(adata.layers['spliced'])

    # 初始化更新后的u和s（直接复制原始值）
    u_updated = u.copy()
    s_updated = s.copy()

    # 遍历所有细胞，但跳过前 half_window 和最后 half_window 个点
    for i in range(half_window, len(adata_new) - half_window):
        # 确定窗口范围（i是中心点，左右各取 half_window 个邻居）
        start = i - half_window
        end = i + half_window + 1  # +1 因为 Python 切片是左闭右开
        
        # 计算窗口内细胞的平均值
        u_window_mean = np.mean(u[start:end], axis=0)
        s_window_mean = np.mean(s[start:end], axis=0)
        
        # 应用更新规则：u_new = u_old + lr * (neighborhood_mean - u_old)
        u_updated[i] = u[i] + learning_rate * (u_window_mean - u[i])
        s_updated[i] = s[i] + learning_rate * (s_window_mean - s[i])

    # 将更新后的值存回adata
    adata_new.layers['unspliced'] = u_updated
    adata_new.layers['spliced'] = s_updated
    adata_new.X = u_updated + s_updated
    
    return adata_new    

