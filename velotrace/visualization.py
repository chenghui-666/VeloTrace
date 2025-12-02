import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from sklearn.decomposition import PCA
from .model import ODEFunc, ODEBlock
from torch import nn

# ---- Definition: cal_cosine ----
def cal_cosine(adata_cell, odefunc, odeblock, y0, t, num_blocks, num_imp = 50):
    imp_per_block = num_imp//num_blocks # num_imp is the number of ode cells we sample from the ode
    block_size = adata_cell.shape[0]//num_blocks
    
    global all_ode_cell
    adata_cell_blocks = torch.split(adata_cell,block_size)
    t_blocks = torch.split(t, block_size)
    all_ode_cell = []
    all_ode_velo = []
    
    for i in range(num_blocks):
        y0 = adata_cell_blocks[i][0]
        ptimes = torch.linspace(t_blocks[i][0],t_blocks[i][-1],imp_per_block)
        ode_cell = odeblock(y0, ptimes)
        all_ode_cell.append(ode_cell)
        
        ode_velo = odefunc(0, ode_cell)
        all_ode_velo.append(ode_velo)

    all_ode_cell = torch.cat(all_ode_cell)
    all_ode_velo = torch.cat(all_ode_velo)
    print(all_ode_cell.shape[0])
    print(all_ode_velo.shape[0])
    # Extract cell types
    cell_types =  np.array(['cell' for _ in range(adata_cell.shape[0])])

    ode_types = np.array(['ode' for _ in range(all_ode_cell.shape[0])])

    cell_types = np.concatenate((ode_types, cell_types))

    all_emb = torch.concatenate((all_ode_cell, adata_cell))
    global temp_ode_cell
    global temp_ode_velo
    temp_ode_cell = torch.tensor(all_emb.cpu().detach().numpy()[:num_imp-1], dtype = torch.float32)
    temp_ode_velo = torch.tensor(all_ode_velo.cpu().detach().numpy()[:num_imp-1], dtype = torch.float32)
    cell_dirs = temp_ode_cell[1:] - temp_ode_cell[:-1]
    velo_dirs = temp_ode_velo[:-1]
    import torch.nn.functional as F
    cos_high_dim = F.cosine_similarity(velo_dirs, cell_dirs, dim = 1)
    print(cos_high_dim)
    
    global PCA_model
    PCA_model = PCA(n_components = 2)
    global PCA_embedding 
    PCA_embedding = PCA_model.fit_transform(temp_ode_cell.numpy())
    PCA_velo = torch.tensor(np.dot(temp_ode_velo, PCA_model.components_.T))
    
    # global displacement
    # displacement = temp_ode_cell[1:] - temp_ode_cell[:-1]
    # global pca_displacement
    # pca_displacement = PCA_model.transform(displacement.numpy())
    # global displacement_2d
    # displacement_2d = torch.tensor(pca_displacement, dtype = torch.float32)
    
    ode_cell_2d = torch.tensor(PCA_embedding[:num_imp-1], dtype = torch.float32)
    ode_velo_2d = torch.tensor(PCA_velo[:num_imp-1], dtype = torch.float32)
    
    cell_2d_dirs = ode_cell_2d[1:] - ode_cell_2d[:-1]
    velo_2d_dirs = ode_velo_2d[:-1]
    
    cos_low_dim = F.cosine_similarity(velo_2d_dirs, cell_2d_dirs, dim = 1)
    print(cos_low_dim)

# ---- Definition: plot_scatter_with_ode ----

def calculate_dynamic_bins(velocity):
    """
    根据速度的绝对值动态计算区间边界：
    - 0: 靠近零的值（如分位数 25% 以下）
    - 1: 靠近中位数的值（分位数 25%-75%）
    - 2: 其余较大的值（分位数 75% 以上）
    Args:
        velocity: (N, D) 张量，每行是细胞速度，每列是基因。
    Returns:
        动态计算的区间边界 (list)。
    """
    abs_velocity = torch.abs(velocity).view(-1)  # 展平成 1D
    q1 = torch.quantile(abs_velocity, 0.25)  # 25% 分位数
    median = torch.median(abs_velocity)  # 中位数
    q3 = torch.quantile(abs_velocity, 0.75)  # 75% 分位数
    return [0.0, q1.item(), q3.item(), float('inf')]  # 返回动态边界

# ---- Definition: discretize_velocity_dynamic ----

def discretize_velocity_dynamic(velocity):
    """
    对速度数据进行动态五值化处理。
    Args:
        velocity: (N, D) 张量，每行是细胞速度，每列是基因。
    Returns:
        五值化结果 (N, D)，取值为 {-2, -1, 0, 1, 2}。
    """
    bins = calculate_dynamic_bins(velocity)  # 动态计算区间
    abs_velocity = torch.abs(velocity)
    labels = torch.zeros_like(velocity, dtype=torch.long)
    
    # 按绝对值动态区间进行分类
    for i in range(len(bins) - 1):
        mask = (abs_velocity >= bins[i]) & (abs_velocity < bins[i + 1])
        labels[mask] = i  # 分类为 {0, 1, 2} 的绝对值类别
    
    # 添加正负号
    labels[velocity < 0] *= -1
    return labels

# ---- Definition: plot_scatter_with_ode ----

def plot_scatter_with_ode(adata_cell, odefunc, odeblock, y0, t, num_blocks, num_imp = 50, adata_velo = False, v_scale = 1.0, save_path = None):

    imp_per_block = num_imp//num_blocks # num_imp is the number of ode cells we sample from the ode
    block_size = adata_cell.shape[0]//num_blocks    ## 500/5 = 100
    random_indices = [random.randint(0,adata_cell.shape[0]-1) for _ in range(200)]
    adata_cell_blocks = torch.split(adata_cell,block_size)
    print("adata_cell_blocks[0] = ", adata_cell_blocks[0].shape)
    t_blocks = torch.split(t, block_size)
    all_ode_cell = []
    all_ode_velo = []
    
    for i in range(num_blocks):
        y0 = adata_cell_blocks[i][0]
        ptimes = torch.linspace(t_blocks[i][0],t_blocks[i][-1],imp_per_block)
        ode_cell = odeblock(y0, ptimes)
        all_ode_cell.append(ode_cell)
        
        ode_velo = odefunc(0, ode_cell)
        all_ode_velo.append(ode_velo)

    all_ode_cell = torch.cat(all_ode_cell)
    all_ode_velo = torch.cat(all_ode_velo)
    # Extract cell types
    cell_types =  np.array(['cell' for _ in range(adata_cell.shape[0])])

    ode_types = np.array(['ode' for _ in range(all_ode_cell.shape[0])])

    cell_types = np.concatenate((ode_types, cell_types))

    all_emb = torch.concatenate((all_ode_cell, adata_cell))


    # Compute PCA embedding
    PCA_model = PCA(n_components = 2)
    PCA_embedding = PCA_model.fit_transform(all_emb.cpu().detach().numpy())
    PCA_ode_velo = torch.tensor(np.dot(all_ode_velo.cpu().detach().numpy(),PCA_model.components_.T))
    all_cell_velo = odefunc(0, adata_cell)
    PCA_cell_velo = torch.tensor(np.dot(all_cell_velo.cpu().detach().numpy(),PCA_model.components_.T))

    # Create a DataFrame for UMAP results and cell types
    pca_df = pd.DataFrame(PCA_embedding, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cell_types

    # Plot UMAP
    plt.figure(figsize=(10, 8))
    for cell_type in pca_df['cluster'].unique():
        subset = pca_df[pca_df['cluster'] == cell_type]
        if cell_type == 'ode':
            plt.scatter(subset['PC1'], subset['PC2'], label=cell_type, alpha=0.5, s=50, color = 'k', marker= 'x')
        else:
            plt.scatter(subset['PC1'], subset['PC2'], label=cell_type, alpha=0.5, s=180, marker = '.', c = t.cpu())


    for i in range(0,all_ode_cell.shape[0]):
        plt.arrow(
            PCA_embedding[i, 0], PCA_embedding[i, 1],  # 起点
            PCA_ode_velo[i, 0] /50, PCA_ode_velo[i, 1]/50,  # 变化速率
            head_width=0.2, head_length=0.1 * v_scale, fc='black', ec='black'  # 箭头属性
        )
        
    if adata_velo:
        cell_start = all_ode_cell.shape[0]
        for i in random_indices:
            plt.arrow(
                PCA_embedding[cell_start+i, 0], PCA_embedding[cell_start+i, 1],  # 起点
                PCA_cell_velo[i, 0]/50 , PCA_cell_velo[i, 1]/50,  # 变化速率
                head_width=0.2, head_length=0.1 * v_scale, fc='red', ec='red'  # 箭头属性
            )
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.legend(fontsize = 23)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Cell Embeddings')
    plt.show()

# ---- Definition: plot_scatter ----

def plot_scatter(adata_cell, t):

    # Extract cell types
    cell_types =  np.array(['cell' for _ in range(adata_cell.shape[0])])

    all_emb = adata_cell


    # Compute PCA embedding
    PCA_model = PCA(n_components = 2)
    PCA_embedding = PCA_model.fit_transform(all_emb.cpu().detach().numpy())

    # Create a DataFrame for UMAP results and cell types
    pca_df = pd.DataFrame(PCA_embedding, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cell_types

    # Plot UMAP
    plt.figure(figsize=(10, 8))
    for cell_type in pca_df['cluster'].unique():
        subset = pca_df[pca_df['cluster'] == cell_type]
        plt.scatter(subset['PC1'], subset['PC2'], label=cell_type, alpha=0.5, s=10, c = t.cpu())
        
    plt.legend(title='Cell Type')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Cell Embeddings')
    plt.show()




def process_and_plot_ode_results(
    model_path: str,
    data_idx: int,
    theta_idx: int,
    adata,
    adata_true,
    cell_velo_dx,
    save_path=None,
    input_dim: int = 60,
    num_blocks: int = 25,
    scale: float = 1.75,
    num_genes_to_plot: int = 30,
    device: str = 'cpu'
):
    """
    加载一个训练好的ODE模型，计算轨迹和速度，并绘制结果图。

    Args:
        model_path (str): 包含模型和数据文件的目录路径。
        data_idx (int): 数据文件（如 modified_x, t, v）的索引。
        theta_idx (int): 模型参数文件（theta）的索引。
        adata: 包含观测数据的 AnnData 对象。
        adata_true: 包含真实数据的 AnnData 对象。
        cell_velo_dx (np.ndarray): 真实的细胞速度（dx/dt）数组。
        save_path (str): 保存输出图像的目录。
        input_dim (int): ODE 模型的输入维度。
        num_blocks (int): 将轨迹分割成的块数。
        scale (float): 用于绘图的缩放因子。
        num_genes_to_plot (int): 要绘制的基因数量。
        device (str): 使用的 PyTorch 设备（'cpu' 或 'cuda'）。
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 返回一个元组，包含预测的细胞状态 (pred_y) 
                                       和预测的细胞速度 (ode_velo)。
    """
  

    # 1. 加载数据和训练好的ODE模型
    adata_cell = torch.load(os.path.join(model_path, f'modified_x_{data_idx}.pth'))
    v = torch.load(os.path.join(model_path, f'v_{data_idx}.pth'))
    t_raw = torch.load(os.path.join(model_path, f't_{data_idx}.pth'))
    t = torch.tensor(t_raw)

    odefunc = ODEFunc(input_dim).to(device)
    odeblock = ODEBlock(odefunc).to(device)
    loaded_params = torch.load(os.path.join(model_path, f'theta_{theta_idx}.pth'))
    odefunc.load_state_dict(loaded_params)
    odefunc.eval()  # 设置为评估模式

    # 2. 预处理时间点，确保其严格单调递增
    eps = 1e-6
    for i in range(1, t.shape[0]):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + eps

    # 3. 将数据分割成块以便处理
    start_points = []
    num_cells = adata_cell.shape[0]  # 例如 2000
    cells_per_major_step = num_cells // 5  # 例如 400
    for j in range(0, num_cells, cells_per_major_step):
        for i in range(num_blocks // 5):
            start_points.append(j + 20 * i)

    adata_cell_blocks = []
    t_blocks = []
    for m in range(len(start_points)):
        start_point = start_points[m]
        end_point = start_points[m + 1] if m + 1 < len(start_points) else num_cells
        adata_cell_blocks.append(adata_cell[start_point:end_point])
        t_blocks.append(t[start_point:end_point])

    # 4. 计算每个块的ODE速度和细胞状态
    criterion = nn.MSELoss()
    all_ode_cell = []
    all_ode_velo = []

    with torch.no_grad():  # 在推断时关闭梯度计算
        for i in range(num_blocks):
            if len(adata_cell_blocks[i]) == 0:
                continue
            
            y0 = adata_cell_blocks[i][0].to(device)
            current_t_block = t_blocks[i].to(device)
            
            ode_cell = odeblock(y0, current_t_block)
            ode_velo = odefunc(0, adata_cell_blocks[i].squeeze().to(device))
            
            all_ode_cell.append(ode_cell)
            all_ode_velo.append(ode_velo)
            
            print(f"Block {i} MSE Loss (gene 0): {criterion(ode_cell.squeeze()[:, 0], adata_cell_blocks[i][:, 0].detach().to(device)).item()}")

    pred_y = torch.cat(all_ode_cell).detach().cpu().numpy()
    ode_velo = torch.cat(all_ode_velo).detach().cpu().numpy()

    # 5. 为指定数量的基因绘制结果图
    print(f"\nGenerating plots for {num_genes_to_plot} genes...")
    for j in range(num_genes_to_plot):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Gene {j} Analysis", fontsize=18, y=1.0)

        # 图1: 真实 vs. 噪声 vs. 预测表达量
        axes[0].scatter(adata.obs["true_t"], adata_true.X[:, j], label="True")
        axes[0].scatter(adata.obs["true_t"], adata.X[:, j] * scale, alpha=0.5, label="Noisy")
        axes[0].scatter(adata.obs["true_t"], pred_y[:, j] * scale, label="Pred_y")
        axes[0].set_title("Expression: True vs. Noisy vs. Pred", fontsize=16)
        axes[0].set_xlabel("Pseudo-time", fontsize=14)
        axes[0].set_ylabel("Gene X expression", fontsize=14)
        axes[0].legend()

        # 图2: 真实 vs. 更新后 vs. 预测表达量
        axes[1].scatter(adata.obs["true_t"], adata_true.X[:, j], label="True")
        axes[1].scatter(adata.obs["true_t"], adata_cell.detach().cpu().numpy()[:, j] * scale, label="Updated_x")
        axes[1].scatter(adata.obs["true_t"], pred_y[:, j] * scale, s=7, label="Pred_y")
        axes[1].set_title("Expression: True vs. Updated vs. Pred", fontsize=16)
        axes[1].set_xlabel("Pseudo-time", fontsize=14)
        axes[1].legend()

        # 图3: 速度比较
        axes[2].scatter(adata.obs['true_t'], cell_velo_dx[:, j], label="True")
        axes[2].scatter(adata.obs['true_t'], v[:, j].cpu().numpy(), label="scVelo")
        axes[2].scatter(adata.obs["true_t"], ode_velo[:, j], label="ODE_velo")
        axes[2].set_title("Velocity: True vs. scVelo vs. ODE", fontsize=16)
        axes[2].set_xlabel("Pseudo-time", fontsize=14)
        axes[2].set_ylabel("dx/dt", fontsize=14)
        axes[2].legend()
        

        if save_path is not None:
            fig_path = os.path.join(save_path, f"gene_{j}.png")
            fig.savefig(fig_path)



def process_and_plot_ode_results2(
    model_path: str,
    data_idx: int,
    theta_idx: int,
    adata,
    adata_true,
    cell_velo_dx,
    save_path=None,
    input_dim: int = 60,
    scale: float = 1.75,
    num_genes_to_plot: int = 30,
    num_samples: int = 200,
    device: str = 'cpu'
):
    odefunc = ODEFunc(input_dim).to(device)
    odeblock = ODEBlock(odefunc).to(device)
    loaded_params = torch.load(os.path.join(model_path, f'theta_{theta_idx}.pth'), map_location=device)
    odefunc.load_state_dict(loaded_params)
    odefunc.eval()

    adata_cell = torch.load(os.path.join(model_path, f'modified_x_{data_idx}.pth')).to(device)
    v = torch.load(os.path.join(model_path, f'v_{data_idx}.pth')).to(device)
    t_raw = torch.load(os.path.join(model_path, f't_{data_idx}.pth'))
    t = torch.tensor(t_raw, dtype=torch.float32, device=device)

    eps = 1e-6
    for i in range(1, t.shape[0]):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + eps

    with torch.no_grad():
        y0 = adata_cell[0]
        pred_y = odeblock(y0, t).detach().cpu()
        ode_velo = odefunc(0, pred_y.to(device)).detach().cpu()

    pred_y_np = pred_y.numpy()
    ode_velo_np = ode_velo.numpy()
    adata_cell_np = adata_cell.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()

    num_samples = min(num_samples, pred_y_np.shape[0])
    sample_idx = torch.linspace(0, pred_y_np.shape[0] - 1, steps=num_samples).long()
    sampled_t = adata.obs["true_t"][sample_idx]
    sampled_pred_y = pred_y_np[sample_idx]
    sampled_adata_cell = adata_cell_np[sample_idx]
    sampled_ode_velo = ode_velo_np[sample_idx]
    sampled_v = v_np[sample_idx]
    sampled_cell_velo_dx = cell_velo_dx[sample_idx]

    print(f"\nGenerating plots for {num_genes_to_plot} genes...")
    for j in range(num_genes_to_plot):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Gene {j} Analysis", fontsize=18, y=1.0)

        axes[0].scatter(adata.obs["true_t"], adata_true.X[:, j], label="True")
        axes[0].scatter(adata.obs["true_t"], adata.X[:, j] * scale, alpha=0.5, label="Noisy")
        axes[0].scatter(sampled_t, sampled_pred_y[:, j] * scale, label="Pred_y")
        axes[0].set_title("Expression: True vs. Noisy vs. Pred")
        axes[0].set_xlabel("Pseudo-time")
        axes[0].set_ylabel("Gene expression")
        axes[0].legend()

        axes[1].scatter(adata.obs["true_t"], adata_true.X[:, j], label="True")
        axes[1].scatter(adata.obs["true_t"], adata_cell_np[:, j] * scale, label="Updated_x")
        axes[1].scatter(sampled_t, sampled_pred_y[:, j] * scale, s=7, label="Pred_y")
        axes[1].set_title("Expression: True vs. Updated vs. Pred")
        axes[1].set_xlabel("Pseudo-time")
        axes[1].legend()

        axes[2].scatter(adata.obs['true_t'], cell_velo_dx[:, j], label="True")
        axes[2].scatter(sampled_t, sampled_v[:, j], label="scVelo")
        axes[2].scatter(sampled_t, sampled_ode_velo[:, j], label="ODE_velo")
        axes[2].set_title("Velocity: True vs. scVelo vs. ODE")
        axes[2].set_xlabel("Pseudo-time")
        axes[2].set_ylabel("dx/dt")
        axes[2].legend()

        if save_path is not None:
            fig_path = os.path.join(save_path, f"gene_{j}.png")
            fig.savefig(fig_path)



