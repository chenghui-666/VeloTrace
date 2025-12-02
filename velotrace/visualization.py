import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from torch import nn, optim
import scvelo as scv
import scanpy as sc
import anndata
from anndata import AnnData
import torch
import torch.utils.data as data
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize 
from sklearn.linear_model import LinearRegression
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from pygam import LinearGAM, s

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


# ------------------------------------------------------------------
# Velocity Calculation Functions
# ------------------------------------------------------------------

def cal_velocity(adata_init, t):
    delta_t = np.diff(t)
    print(delta_t)
    alpha_data = np.zeros((adata_init.shape[0], adata_init.shape[1]))
    for i in range(adata_init.shape[0]):
        for j in range(adata_init.shape[1]):
            alpha_data[i, j] = adata_init.X[i, j]
    
    cell_velo_dx = np.zeros_like(alpha_data)
    for i in range(0, alpha_data.shape[0]-1):
        if i == 0:
            cell_velo_dx[i] = (alpha_data[1] - alpha_data[0]) / delta_t[i+1]
        else:
            cell_velo_dx[i] = (alpha_data[i] - alpha_data[i - 1]) / delta_t[i]
    return cell_velo_dx

def calculate_odevelo_cell(odefunc, odeblock, t: torch.Tensor, x_modified: torch.Tensor):
    # ensure t is strict increasing
    eps = 1e-6
    for i in range(1,t.shape[0]):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + eps
            
    y0 = x_modified[0]
    ode_cell = odeblock(y0, t).cpu().detach().numpy()
    ode_velo = odefunc(0, x_modified).cpu().detach().numpy()
    
    return ode_cell, ode_velo

# ------------------------------------------------------------------
# PCA Visualization Functions
# ------------------------------------------------------------------

def plot_velocity_on_pca(adata_cell, velo, samp_indices, t, step=1.0, 
                         title=None, uniform_arrow=False, save_path=None):
    """
    Perform PCA and project velocity onto the PC space.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(adata_cell)
    
    # Project velocity: V_pca = V * W.T (Mathematically equivalent to (X + V)W - XW)
    V_pca = np.dot(velo * step, pca.components_.T)

    if uniform_arrow:
        norms = np.linalg.norm(V_pca, axis=1)
        norms[norms == 0] = 1e-9
        V_pca = (V_pca / norms[:, None]) * np.mean(norms[samp_indices])

    plt.figure(figsize=(10, 8), dpi=500)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap='YlGnBu_r',
                alpha=0.35, s=1000, marker='.', lw=0)

    plt.scatter(X_pca[samp_indices, 0], X_pca[samp_indices, 1],
                c=t[samp_indices], cmap='YlGnBu_r',
                alpha=1.0, s=1000, marker='.', edgecolor='black', linewidth=2.5)

    for i in samp_indices:
        plt.arrow(X_pca[i, 0], X_pca[i, 1], V_pca[i, 0], V_pca[i, 1],
                  linewidth=2.5, head_width=2.5, head_length=3.5, fc='black', ec='black')

    plt.axis('off')
    if title: 
        plt.title(title, fontsize=28)
    if save_path: 
        plt.savefig(save_path, dpi=500, bbox_inches='tight', format='svg')
        
    plt.show()

def plot_sctour_velocity_on_pca2(adata, samp_indices, t, step=1.0, title=None, save_path=None):
    """
    Project 5D velocity (X_VF) onto 2D PCA (X_pca2) via linear mapping from X_TNODE.
    """
    if not all(k in adata.obsm for k in ['X_pca2', 'X_TNODE', 'X_VF']):
        raise KeyError("Missing required keys in adata.obsm: X_pca2, X_TNODE, X_VF")

    X_emb = adata.obsm['X_pca2'][:, :2]
    reg = LinearRegression().fit(adata.obsm['X_TNODE'], X_emb)
    
    # Project velocity: V_2d = V_5d * W.T
    V_emb = np.dot(adata.obsm['X_VF'], reg.coef_.T)

    plt.figure(figsize=(10, 8), dpi=500)

    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=t, cmap='YlGnBu_r', 
                alpha=0.35, s=1000, marker='.', lw=0)

    colors = t[samp_indices] if len(t) == len(X_emb) else t
    plt.scatter(X_emb[samp_indices, 0], X_emb[samp_indices, 1], c=colors, cmap='YlGnBu_r',
                alpha=1, s=1000, marker='.', edgecolor='black', linewidth=2.5)

    for i in samp_indices:
        dx, dy = V_emb[i] * step
        plt.arrow(X_emb[i, 0], X_emb[i, 1], dx, dy, 
                  linewidth=2.5, head_width=0.02, head_length=0.03, fc='black', ec='black')

    plt.axis('off')
    if title: 
        plt.title(title, fontsize=28)
    if save_path: 
        plt.savefig(save_path, dpi=500, bbox_inches='tight', format='svg')
        
    plt.show()

# ------------------------------------------------------------------
# Rose Plot / Angular Similarity Functions
# ------------------------------------------------------------------

def angle_between_vectors(A, B):
    dot_product = np.sum(A * B, axis=1) 
    
    norm_A = np.linalg.norm(A, axis=1) 
    norm_B = np.linalg.norm(B, axis=1) 
    
    cosine_similarity = dot_product / (norm_A * norm_B)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    angle_radians = np.arccos(cosine_similarity)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def transform_radius(y_arr, boundary_val, boundary_pos, max_val):
    """Piecewise linear transformation mapping data to 0-1 plotting space."""
    y = np.asarray(y_arr, dtype=float)
    res = np.zeros_like(y)

    mask = y <= boundary_val
    res[mask] = (y[mask] / boundary_val) * boundary_pos
    
    res[~mask] = boundary_pos + ((y[~mask] - boundary_val) / (max_val - boundary_val)) * (1.0 - boundary_pos)
    return res

def plot_rose_plot_final(ax, angles_deg, title, cmap_name, norm, is_solid=True, 
                         custom_yticks=None, boundary_config=None): 
    counts, edges = np.histogram(angles_deg, bins=np.arange(0, 187, 6))
    centers = np.deg2rad(edges[:-1] + np.diff(edges) / 2)
    width = np.deg2rad(np.diff(edges)[0])
    
    base_cmap = plt.get_cmap(cmap_name)
    dark_cmap = mcolors.LinearSegmentedColormap.from_list("dark", base_cmap(np.linspace(0.3, 1, 256)))
    colors = dark_cmap(norm(counts))

    heights = counts
    if boundary_config:
        b_args = (boundary_config['val'], boundary_config['pos'], boundary_config['max'])
        heights = transform_radius(counts, *b_args)

    plot_kwargs = {'width': width, 'bottom': 0.0, 'linewidth': 1.5}
    if is_solid:
        ax.bar(centers, heights, color=colors, **plot_kwargs)
    else:
        ax.bar(centers, heights, facecolor='none', edgecolor=colors, **plot_kwargs)

    ax.set_thetamin(0); ax.set_thetamax(180)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)      
    ax.set_xticklabels([]); ax.set_yticklabels([])

    if custom_yticks:
        if boundary_config:
            ax.set_yticks(transform_radius(custom_yticks, *b_args))
            ax.set_ylim(0, 1.0)
            ax.grid(True, axis='y', ls='-', alpha=0.6, color='grey')
            ax.set_rlabel_position(90)
        else:
            ax.set_yticks(custom_yticks)

    ax.set_title(title, va='bottom', fontsize=20, pad=20)

# ------------------------------------------------------------------
# 2-Gene Vector Field Visualization
# ------------------------------------------------------------------


def visualization_2g_vf_cells(
    adata, 
    velo, 
    t, 
    gene1_name, 
    gene2_name, 
    sample_indices, 
    v_scale=1.0,
    title=None,
    save_path=None
):
    """
    Visualizes the 2-gene velocity vector field based on cell expression and pre-calculated velocity.

    Args:
        adata (AnnData): AnnData object containing expression data.
        velo (np.array): Full velocity matrix (n_obs, n_vars).
        t (np.array): Time/pseudotime vector used for cell coloring.
        gene1_name (str): Name of the gene for the X-axis.
        gene2_name (str): Name of the gene for the Y-axis.
        sample_indices (list/array): Indices of cells where velocity arrows will be drawn.
        v_scale (float): Scaling factor for arrow length.
        title (str, optional): Custom plot title. Defaults to 'Two-gene vector field'.
        save_path (str, optional): Path to save the figure.
    """
    
    # --- 1. Data Preparation ---
    try:
        gene_indices = [adata.var_names.get_loc(gene1_name), adata.var_names.get_loc(gene2_name)]
    except KeyError as e:
        print(f"Error: Gene name not found in adata.var_names: {e}")
        return

    X_data = adata.X.toarray() if issparse(adata.X) else adata.X
    adata_2g = X_data[:, gene_indices]

    sampled_pos_2g = adata_2g[sample_indices]
    sampled_velo_2g = velo[sample_indices][:, gene_indices]
    t_sampled = t[sample_indices] if len(t) == adata.shape[0] else t

    U = np.sign(sampled_velo_2g[:, 0]) * np.log1p(np.abs(sampled_velo_2g[:, 0])) * v_scale
    V = np.sign(sampled_velo_2g[:, 1]) * np.log1p(np.abs(sampled_velo_2g[:, 1])) * v_scale
    
    plt.figure(figsize=(10, 8))

    sc = plt.scatter(
        adata_2g[:, 0], adata_2g[:, 1], 
        c=t, cmap='YlGnBu_r', alpha=0.2, s=150, rasterized=True
    ) 

    plt.scatter(
        sampled_pos_2g[:, 0], sampled_pos_2g[:, 1], 
        c=t_sampled, cmap='YlGnBu_r', alpha=1, 
        s=150, edgecolor='black', linewidths=1, rasterized=True
    )

    plt.quiver(
        sampled_pos_2g[:, 0], sampled_pos_2g[:, 1], U, V, 
        color='black', angles='xy', scale_units='xy', scale=1.5,
        width=0.005, headwidth=3, headlength=4, alpha=0.8
    )

    plt.rcParams.update({'font.size': 20}) 
    plt.xlabel(f"{gene1_name}", fontsize=32) 
    plt.ylabel(f"{gene2_name}", fontsize=32) 
    plt.title(title if title else 'Two-gene vector field', fontsize=32) 

    ax = plt.gca() 
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    x_min, x_max = adata_2g[:, 0].min(), adata_2g[:, 0].max()
    y_min, y_max = adata_2g[:, 1].min(), adata_2g[:, 1].max()
    x_range, y_range = x_max - x_min, y_max - y_min

    pad_x, pad_y = x_range * 0.05, y_range * 0.05
    arrow_origin_x, arrow_origin_y = x_min - pad_x, y_min - pad_y

    ax.arrow(arrow_origin_x, arrow_origin_y, x_max + pad_x - arrow_origin_x, 0, 
             head_width=0.02 * y_range, head_length=0.03 * x_range, fc='black', ec='black', lw=1.5, length_includes_head=True)
    ax.arrow(arrow_origin_x, arrow_origin_y, 0, y_max + pad_y - arrow_origin_y, 
             head_width=0.02 * x_range, head_length=0.03 * y_range, fc='black', ec='black', lw=1.5, length_includes_head=True)
             
    ax.set_xlim(arrow_origin_x, x_max + pad_x)
    ax.set_ylim(arrow_origin_y, y_max + pad_y)
    
    plt.colorbar(sc, label='True Time', alpha=1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight', format='svg')
    else:
        plt.show()

# ------------------------------------------------------------------
# Phase Portrait Functions
# ------------------------------------------------------------------

def to_dense(data):
    if issparse(data):
        return data.toarray()
    return np.asarray(data)

def plot_phase_portrait_quiver(ax, adata_obj,  time, g, title, 
                               sample_indices, 
                               show_ylabel=False):

    if isinstance(g, str):
        g = adata_obj.var_names.get_loc(g)

    dsdt_raw = to_dense(adata_obj.layers["velocity"][:, g])
    dudt_raw = to_dense(adata_obj.layers["velocity_u"][:, g])
    st_raw = to_dense(adata_obj.layers["spliced"][:, g])
    ut_raw = to_dense(adata_obj.layers["unspliced"][:, g])
    time = time
    
    sc = ax.scatter(st_raw, ut_raw, c=time, 
                    cmap="YlGnBu_r",       
                    alpha=0.3,          
                    s=50,               
                    rasterized=True,    
                    zorder=2)           

    ax.scatter(st_raw[sample_indices], ut_raw[sample_indices], 
               c=time[sample_indices], 
               cmap="YlGnBu_r",       
               alpha=1,          
               s=50,  
               edgecolor='black',
               linewidth=1,             
               rasterized=True,    
               zorder=2)           
    
    st_arrows   = st_raw[sample_indices]
    ut_arrows   = ut_raw[sample_indices]
    dsdt_arrows = dsdt_raw[sample_indices]
    dudt_arrows = dudt_raw[sample_indices]
    
    
    ax.quiver(st_arrows, ut_arrows, 
              dsdt_arrows, dudt_arrows,
              angles='xy', 
              scale_units='xy', 
              scale=2,               
              width=0.005,            
              headwidth=5,            
              headlength=6,           
              color='k',
              zorder=3) 

    ax.set_xlabel("Spliced RNA", fontsize=22)
    if show_ylabel==True:
        ax.set_ylabel("Unspliced RNA", fontsize=22)
    ax.set_title(title, fontsize=24, pad=10)
    
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels([]) 
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', length=0)
    
    return sc

# ------------------------------------------------------------------
# Trajectory and Background Coloring Functions
# ------------------------------------------------------------------

def _to_numpy(data, idx=None):
    """Convert tensor/array to numpy and select index if needed."""
    arr = np.asarray(data.cpu().detach() if hasattr(data, 'cpu') else data)
    if idx is not None and arr.ndim > 1:
        return arr[:, idx]
    return arr

def _calculate_diff_velocity(t, x):
    """Calculate velocity using forward difference."""
    dt = np.diff(t)
    dx = np.diff(x)
    dt[dt == 0] = 1e-6
    v = dx / dt
    return np.append(v, v[-1]) 

def _format_output(t, x, v, original_shape_cols, gene_idx):
    """Format x and v back to original matrix shape."""
    k = len(t)
    x_out = np.zeros((k, original_shape_cols))
    v_out = np.zeros((k, original_shape_cols))
    v_log = np.sign(v) * np.log1p(np.abs(v))
    
    if original_shape_cols > 1:
        x_out[:, gene_idx] = x
        v_out[:, gene_idx] = v_log
    else:
        x_out[:] = x.reshape(-1, 1)
        v_out[:] = v_log.reshape(-1, 1)
        
    return t, x_out, v_out

def plot_velocity_bg(t, x_data, v_data, gene_idx, title, ax, scatter=True, fill=True):
    """
    Plot gene expression with background colored by velocity direction (rising/falling).
    """
    t_arr = _to_numpy(t)
    x_arr = _to_numpy(x_data, gene_idx)
    v_arr = _to_numpy(v_data, gene_idx)

    mask = np.isfinite(t_arr) & np.isfinite(x_arr) & np.isfinite(v_arr)
    t_clean, x_clean, v_clean = t_arr[mask], x_arr[mask], v_arr[mask]
    
    order = np.argsort(t_clean)
    t_ord, x_ord, v_ord = t_clean[order], x_clean[order], v_clean[order]

    if fill and len(t_ord) > 1:
        signs = np.sign(v_ord[:-1])
        signs[signs == 0] = 1 
        boundaries = np.concatenate(([0], np.where(np.diff(signs) != 0)[0] + 1, [len(signs)]))
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            seg_sign = signs[start]
            
            color = "#d62a2b" if seg_sign > 0 else "#1f77b5" # Red for rise, Blue for fall
            ax.axvspan(t_ord[start], t_ord[end], facecolor=color, alpha=0.8, edgecolor=None)

    if scatter:
        ax.scatter(t_clean, x_clean, c='#1A237E', s=10, alpha=0.8, rasterized=True)
    else:
        ax.plot(t_ord, x_ord, c='#1A237E', lw=2.5, alpha=1, rasterized=True)

    ax.set_ylabel('Expression', fontsize=22)
    ax.set_title(title, fontsize=28)
    ax.set_xticks([])
    ax.tick_params(axis='both', labelsize=16)
    if len(t_ord) > 0:
        ax.set_xlim(t_ord.min(), t_ord.max())

def get_kmeans_velocity(t, x_data, gene_idx, k=50):
    """
    Computes velocity by clustering (t, x) data using K-Means and calculating
    the finite difference (slope) between sorted cluster centers.
    The result ensures background color blocks align with the line slopes.
    """
    t_np = np.asarray(t.cpu().detach() if hasattr(t, 'cpu') else t)
    
    if x_data.shape[1] > 1:
        x_np = np.asarray(x_data[:, gene_idx].cpu().detach() if hasattr(x_data, 'cpu') else x_data[:, gene_idx])
    else:
        x_np = np.asarray(x_data.cpu().detach() if hasattr(x_data, 'cpu') else x_data)

    mask = np.isfinite(t_np) & np.isfinite(x_np)
    t_clean = t_np[mask]
    x_clean = x_np[mask]

    data = np.vstack((t_clean, x_clean)).T
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20) 
    kmeans.fit(data_scaled)
    
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    sorted_indices = np.argsort(centers[:, 0])
    centers_sorted = centers[sorted_indices]
    
    t_centers = centers_sorted[:, 0]
    x_centers = centers_sorted[:, 1]

    dt = np.diff(t_centers)
    dx = np.diff(x_centers) 

    dt[dt == 0] = 1e-6
    
    v_segments = dx / dt

    v_centers = np.append(v_segments, v_segments[-1])

    n_cols = x_data.shape[1] if x_data.ndim > 1 else 1
    k = len(v_centers)
    x_matrix_fake = np.zeros((k, n_cols))
    v_matrix_fake = np.zeros((k, n_cols))
    
    v_log_scaled = np.sign(v_centers) * np.log1p(np.abs(v_centers))
    
    if n_cols > 1:
        x_matrix_fake[:, gene_idx] = x_centers
        v_matrix_fake[:, gene_idx] = v_log_scaled
    else:
        x_matrix_fake[:] = x_centers.reshape(-1, 1)
        v_matrix_fake[:] = v_log_scaled.reshape(-1, 1)
    
    return t_centers, x_matrix_fake, v_matrix_fake

def get_spectral_velocity(t, x_data, gene_idx, k=50):
    t_np = _to_numpy(t)
    x_np = _to_numpy(x_data, gene_idx)
    n_cols = x_data.shape[1] if x_data.ndim > 1 else 1
    
    mask = np.isfinite(t_np) & np.isfinite(x_np)
    t_clean, x_clean = t_np[mask], x_np[mask]

    if len(t_clean) > 3000:
        idx = np.random.choice(len(t_clean), 3000, replace=False)
        t_in, x_in = t_clean[idx], x_clean[idx]
    else:
        t_in, x_in = t_clean, x_clean

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(np.vstack((t_in, x_in)).T)

    labels = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=15, 
                                assign_labels='kmeans', random_state=42, n_jobs=-1).fit_predict(data_scaled)
    df = pd.DataFrame(data_scaled, columns=['t', 'x'])
    df['label'] = labels
    centers = scaler.inverse_transform(df.groupby('label').mean().values)

    centers = centers[np.argsort(centers[:, 0])]
    v_centers = _calculate_diff_velocity(centers[:, 0], centers[:, 1])

    return _format_output(centers[:, 0], centers[:, 1], v_centers, n_cols, gene_idx)

def get_gam_trajectory(t, x_data, gene_idx, n_splines=20):
    t_np = _to_numpy(t)
    x_np = _to_numpy(x_data, gene_idx)
    
    mask = np.isfinite(t_np) & np.isfinite(x_np)
    t_clean, x_clean = t_np[mask], x_np[mask]

    sort_idx = np.argsort(t_clean)
    t_sorted = t_clean[sort_idx]
    
    gam = LinearGAM(s(0, n_splines=n_splines)).gridsearch(t_clean.reshape(-1, 1), x_clean)
    x_smooth = gam.predict(t_sorted)
    v_smooth = _calculate_diff_velocity(t_sorted, x_smooth)
    
    return t_sorted, x_smooth, v_smooth

# ------------------------------------------------------------------
# MSE Boxplot Functions
# ------------------------------------------------------------------

def plot_precalculated_mse_boxplot(mse_data_dict, palette, title, ax=None, n_sample_points=30):
    """Plot boxplot with white fill, colored edges, and jittered scatter points."""
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
    method_order = ["ground_truth", "scVelo", "VeloVI", "UniTVelo", "VeloTrace(x_raw)", "VeloTrace(x_refined)"]
    method_order = [m for m in method_order if m in mse_data_dict]

    plot_data = []
    for method in method_order:
        values = np.array(mse_data_dict[method]).flatten()
        for v in values[~np.isnan(values)]: plot_data.append({"Method": method, "MSE": v})
    df = pd.DataFrame(plot_data)

    if df.empty: ax.set_title(f"{title} (No Data)"); return

    sns.boxplot(data=df, x="Method", y="MSE", order=method_order, palette=palette, 
                width=0.6, linewidth=1.5, fliersize=0, ax=ax)

    for patch in ax.patches:
        rgba = patch.get_facecolor()
        patch.set_edgecolor((rgba[0], rgba[1], rgba[2], 1.0))
        patch.set_facecolor((rgba[0], rgba[1], rgba[2], 0.25))
        patch.set_linewidth(1.5)

    sampled_df = df if n_sample_points is None else df.groupby("Method", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), n_sample_points))
    )
    sns.stripplot(data=sampled_df, x="Method", y="MSE", order=method_order, palette=palette, 
                  size=4, alpha=0.5, ax=ax, edgecolor="white", linewidth=1, jitter=0.15)

    ax.set_title(title, fontsize=28, pad=15)
    ax.set_xlabel(""); ax.set_ylabel("MSE", fontsize=22)
    ax.set_xticklabels(method_order, rotation=45, ha='right', fontsize=20)
    ax.set_facecolor("#f7f7f7")
    ax.grid(True, axis="y", color="white", alpha=1, linewidth=2.5)
    ax.set_axisbelow(True); ax.tick_params(axis='y', labelsize=20)
    sns.despine(ax=ax)

def create_data_dict(gt, scv, vvi, utv, raw, refined):
    """Maps array variables to required dictionary keys, applying square root (RMSE)."""
    return {
        "ground_truth": np.power(gt, 0.5), "scVelo": np.power(scv, 0.5), 
        "VeloVI": np.power(vvi, 0.5), "UniTVelo": np.power(utv, 0.5), 
        "VeloTrace(x_raw)": np.power(raw, 0.5), "VeloTrace(x_refined)": np.power(refined, 0.5),
    }

# ------------------------------------------------------------------
# PCA Velocity Color Functions
# ------------------------------------------------------------------

def plot_pca_velocity_color(ax, adata, velo, title):
    """Visualizes binary velocity (Pos/Neg) on PCA."""
    X = adata.obsm.get('X_pca', PCA(n_components=2).fit_transform(adata.X))
    # Vectorized color assignment: Red if >= 0, else Blue
    colors = np.where(velo >= 0, '#D62728', '#1F77B4')
    
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.8, marker='.', lw=0, rasterized=True)
    ax.set_title(title, fontsize=22)
    ax.axis('off')

def plot_pca_spectral_binary(ax, adata, t, x, gene_idx, k=20, title=""):
    """Spectral clustering based velocity visualization (Binary Color)."""
    t_arr = np.asarray(t.cpu().detach() if hasattr(t, 'cpu') else t)
    x_arr = np.asarray(x[:, gene_idx].cpu().detach() if hasattr(x, 'cpu') else x[:, gene_idx])
    
    mask = np.isfinite(t_arr) & np.isfinite(x_arr)
    t_c, x_c = t_arr[mask], x_arr[mask]

    data_scaled = StandardScaler().fit_transform(np.vstack((t_c, x_c)).T)
    labels = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=15, 
                                assign_labels='kmeans', random_state=42, n_jobs=-1).fit_predict(data_scaled)

    df = pd.DataFrame({'t': t_c, 'x': x_c, 'label': labels})
    centroids = df.groupby('label').mean().sort_values('t')

    dt = np.gradient(centroids['t'].values)
    dt[dt == 0] = 1e-6
    v_cents = np.gradient(centroids['x'].values) / dt
    
    color_map = {lbl: '#D62728' if v >= 0 else '#1F77B4' for lbl, v in zip(centroids.index, v_cents)}

    final_colors = np.full(len(t_arr), '#E0E0E0', dtype=object)
    final_colors[mask] = [color_map[l] for l in labels]

    X = adata.obsm.get('X_pca', PCA(n_components=2).fit_transform(adata.X))
    ax.scatter(X[:, 0], X[:, 1], c=final_colors, s=100, alpha=0.8, marker='.', lw=0, rasterized=True)
    ax.set_title(title, fontsize=22)
    ax.axis('off')