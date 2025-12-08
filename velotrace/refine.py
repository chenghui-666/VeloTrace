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

    # Reorder the data by time while keeping aligned metadata
    order = np.argsort(adata_new.obs["velocity_pseudotime"].to_numpy())
    adata_new = adata_new[order].copy()
    print(adata_new.obs["velocity_pseudotime"])

    # Configuration
    window_size = window_size  # Sliding-window size (includes the center cell)
    half_window = window_size // 2  # Number of neighbors to each side (e.g., 3 for a size of 7)
    learning_rate = lr  # Step size for the local averaging update

    # Fetch original u and s layers, handling sparse matrices transparently
    u = adata_new.layers['unspliced'] if issparse(adata_new.layers['unspliced']) else np.array(adata.layers['unspliced'])
    s = adata_new.layers['spliced'] if issparse(adata_new.layers['spliced']) else np.array(adata.layers['spliced'])

    # Initialize updated copies of u and s
    u_updated = u.copy()
    s_updated = s.copy()

    # Iterate over all cells, skipping the boundary region that lacks full neighborhoods
    for i in range(half_window, len(adata_new) - half_window):
        # Define the window with ``i`` as the center and ``half_window`` neighbors on each side
        start = i - half_window
        end = i + half_window + 1  # +1 because Python slices are half-open
        
        # Compute neighborhood means
        u_window_mean = np.mean(u[start:end], axis=0)
        s_window_mean = np.mean(s[start:end], axis=0)
        
        # Apply the update rule: move toward the neighborhood mean at the chosen learning rate
        u_updated[i] = u[i] + learning_rate * (u_window_mean - u[i])
        s_updated[i] = s[i] + learning_rate * (s_window_mean - s[i])

    # Write the updated values back to the AnnData object
    adata_new.layers['unspliced'] = u_updated
    adata_new.layers['spliced'] = s_updated
    adata_new.X = u_updated + s_updated
    
    return adata_new    

