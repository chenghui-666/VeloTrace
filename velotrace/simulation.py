import numpy as np
from anndata import AnnData
from scvelo.core import invert, SplicingDynamics
import warnings
import torch
import pandas as pd
import random
from scipy.sparse import issparse

def get_top_gene_indices_per_cell(adata, top_genes_per_cell):
    #########################################################
    # Currently unused helper, kept for reference
    #########################################################
    """
    Return the column indices of the top ``top_genes_per_cell`` genes with the
    highest unspliced expression for every cell.

    Parameters:
        adata (AnnData): AnnData object that must provide ``adata.layers["unspliced"]``.
        top_genes_per_cell (int): Number of genes to keep per cell.

    Returns:
        list of list: Each nested list stores the column indices of the selected genes.
    """
    # Ensure unspliced data is available
    if "unspliced" not in adata.layers:
        raise ValueError("adata.layers['unspliced'] is missing; the dataset must include it.")

    # Retrieve the unspliced matrix
    unspliced_matrix = adata.layers["unspliced"]

    # Storage for the selected indices per cell
    top_gene_indices_per_cell = []

    # Iterate through each cell
    for i in range(unspliced_matrix.shape[0]):
        # Collect the unspliced expression vector of the cell
        cell_unspliced = unspliced_matrix[i, :]

        # Identify indices of the most expressed genes
        top_indices = np.argsort(cell_unspliced)[::-1][:top_genes_per_cell]

        # Record the selection
        top_gene_indices_per_cell.append(top_indices.tolist())
    
    return top_gene_indices_per_cell

# ---- Definition: reduce_matrix_by_top_genes ----

def reduce_matrix_by_top_genes(label_matrix, top_gene_indices_per_cell):
  
    num_cells = label_matrix.shape[0]
    reduced_matrix = np.zeros((num_cells, len(top_gene_indices_per_cell[0])))
    
    # Iterate over each cell
    for i, top_indices in enumerate(top_gene_indices_per_cell):
        # Extract the chosen genes for the current cell
        reduced_matrix[i, :] = label_matrix[i, top_indices]
    
    return reduced_matrix

# ---- Definition: get_vars ----

def get_vars(adata, scaled=True, key="fit"):
    """TODO."""
    alpha = (
        adata.var[f"{key}_alpha"].values if f"{key}_alpha" in adata.var.keys() else 1
    )
    beta = adata.var[f"{key}_beta"].values if f"{key}_beta" in adata.var.keys() else 1
    gamma = adata.var[f"{key}_gamma"].values
    scaling = (
        adata.var[f"{key}_scaling"].values
        if f"{key}_scaling" in adata.var.keys()
        else 1
    )
    t_ = adata.var[f"{key}_t_"].values
    return alpha, beta * scaling if scaled else beta, gamma, scaling, t_

# ---- Definition: unspliced ----

def unspliced(tau, u0, alpha, beta):
    """TODO."""
    expu = np.exp(-beta * tau)
    return u0 * expu + alpha / beta * (1 - expu)

# ---- Definition: spliced ----

def spliced(tau, s0, u0, alpha, beta, gamma):
    """TODO."""
    c = (alpha - u0 * beta) * invert(gamma - beta)
    expu, exps = np.exp(-beta * tau), np.exp(-gamma * tau)
    return s0 * exps + alpha / gamma * (1 - exps) + c * (exps - expu)

# ---- Definition: vectorize ----

def vectorize(t, t_, alpha, beta, gamma=None, alpha_=0, u0=0, s0=0, sorted=False):
    """TODO."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = np.array(t < t_, dtype=int)
    tau = t * o + (t - t_) * (1 - o)

    u0_ = unspliced(t_, u0, alpha, beta)
    s0_ = spliced(t_, s0, u0, alpha, beta, gamma if gamma is not None else beta / 2)

    # vectorize u0, s0 and alpha
    u0 = u0 * o + u0_ * (1 - o)
    s0 = s0 * o + s0_ * (1 - o)
    alpha = alpha * o + alpha_ * (1 - o)

    if sorted:
        idx = np.argsort(t)
        tau, alpha, u0, s0 = tau[idx], alpha[idx], u0[idx], s0[idx]
    return tau, alpha, u0, s0
    
# ---- Definition: compute_dynamics ----

def compute_dynamics(
    adata, basis, key="true", extrapolate=None, sort=True, t_=None, t=None):
    """TODO."""
    idx = adata.var_names.get_loc(basis) if isinstance(basis, str) else basis
    key = "fit" if f"{key}_gamma" not in adata.var_keys() else key
    alpha, beta, gamma, scaling, t_ = get_vars(adata[:, basis], key=key)

    if "fit_u0" in adata.var.keys() and key == "fit":
        u0_offset, s0_offset = adata.var["fit_u0"][idx], adata.var["fit_s0"][idx]
    else:
        u0_offset, s0_offset = 0, 0


    if t is None or isinstance(t, bool) or len(t) < adata.n_obs:
        t = (
            adata.obs[f"{key}_t"].values
            if key == "true"
            else adata.layers[f"{key}_t"][:, idx]
        )

    # if extrapolatre:
    #     u0_ = unspliced(t_, 0, alpha, beta)
    #     tmax = np.max(t) if True else tau_inv(u0_ * 1e-4, u0=u0_, alpha=0, beta=beta)
    #     t = np.concatenate(
    #         [np.linspace(0, t_, num=500), np.linspace(t_, tmax, num=500)]
    #     )

    tau, alpha, u0, s0 = vectorize(np.sort(t) if sort else t, t_, alpha, beta, gamma)

    ut, st = SplicingDynamics(
        alpha=alpha, beta=beta, gamma=gamma, initial_state=[u0, s0]
    ).get_solution(tau, stacked=False)
    ut, st = ut * scaling + u0_offset, st + s0_offset
    return alpha, ut, st

# ---- Definition: unspliced ----

def simulation(
    n_obs=300,
    n_vars=None,
    total_var=600,
    alpha=None,
    beta=None,
    gamma=None,
    alpha_=None,
    t_max=None,
    noise_model="normal",
    noise_level=1,
    switches=None,
    random_seed=0,
    umi_depth = None,
    alpha_all=[]
    ):
    """Simulation of mRNA splicing kinetics.

    Simulated mRNA metabolism with transcription, splicing and degradation.
    The parameters for each reaction are randomly sampled from a log-normal distribution
    and time events follow the Poisson law. The total time spent in a transcriptional
    state is varied between two and ten hours.

    .. image:: https://user-images.githubusercontent.com/31883718/79432471-16c0a000-7fcc-11ea-8d62-6971bcf4181a.png
       :width: 600px

    Returns
    -------
    Returns `adata` object
    """
    # np.random.seed(random_seed)

    def draw_poisson(n):
        from random import seed, uniform  # draw from poisson

        seed(random_seed)
        t = np.cumsum([-0.1 * np.log(uniform(0, 1)) for _ in range(n - 1)])
        return np.insert(t, 0, 0)  # prepend t0=0


    def simulate_dynamics(tau, alpha, beta, gamma, u0, s0, noise_model, noise_level, umi_depth):
        ut, st = SplicingDynamics(
            alpha=alpha, beta=beta, gamma=gamma, initial_state=[u0, s0]
        ).get_solution(tau, stacked=False)
        # if noise_model == "normal":  
        #     ut += np.random.normal(
        #         scale=noise_level * np.percentile(ut, 99) / 10, size=len(ut)
        #     )
        #     st += np.random.normal(
        #         scale=noise_level * np.percentile(st, 99) / 10, size=len(st)
        #     )
        return ut, st

    def simulate_gillespie(alpha, beta, gamma):
        # update rules:
        # transcription (u+1,s), splicing (u-1,s+1), degradation (u,s-1), nothing (u,s)
        update_rule = np.array([[1, 0], [-1, 1], [0, -1], [0, 0]])

        def update(props):
            if np.sum(props) > 0:
                props /= np.sum(props)
            p_cumsum = props.cumsum()
            p = np.random.rand()
            i = 0
            while p > p_cumsum[i]:
                i += 1
            return update_rule[i]

        u, s = np.zeros(len(alpha)), np.zeros(len(alpha))
        for i, alpha_i in enumerate(alpha):
            u_, s_ = (u[i - 1], s[i - 1]) if i > 0 else (0, 0)

            if (alpha_i == 0) and (u_ == 0) and (s_ == 0):
                du, ds = 0, 0
            else:
                du, ds = update(props=np.array([alpha_i, beta * u_, gamma * s_]))

            u[i], s[i] = (u_ + du, s_ + ds)
        return u, s

    alpha = 5 if alpha is None else alpha
    beta = 0.5 if beta is None else beta
    gamma = 0.3 if gamma is None else gamma
    alpha_ = 0 if alpha_ is None else alpha_

    t = np.linspace(0, t_max, n_obs)
    # if t_max is not None:
    #     t *= t_max / np.max(t)
    # t_max = np.max(t)

    def cycle(array, n_vars=None):
        if isinstance(array, (np.ndarray, list, tuple)):
            return (
                array if n_vars is None else array * int(np.ceil(n_vars / len(array)))
            )
        else:
            return [array] if n_vars is None else [array] * n_vars

    # switching time point obtained as fraction of t_max rounded down
    a, b = 0.3, 0.7  # Truncation range
    mu, sigma = 0.5, 0.05  # Mean and standard deviation
    # Convert into the parameters expected by truncnorm
    lower, upper = (a - mu) / sigma, (b - mu) / sigma
    samples_t_ = truncnorm.rvs(lower, upper, loc=mu, scale=sigma, size=n_vars)
    switches = (
        cycle(samples_t_, n_vars)
        # cycle([1], n_vars)
        if switches is None
        else cycle(switches, n_vars)
    )
    t_ = np.array([np.max(t[t < t_i * t_max]) for t_i in switches])

    noise_level = cycle(noise_level, len(switches) if n_vars is None else n_vars)

    n_vars = min(len(switches), len(noise_level)) if n_vars is None else n_vars
    n_housekeeping = n_vars * 9
    n_vars_total = n_vars + n_housekeeping
    U = np.zeros(shape=(len(t), n_vars_total))
    S = np.zeros(shape=(len(t), n_vars_total))
    U_noise = np.zeros(shape=(len(t), n_vars_total))
    S_noise = np.zeros(shape=(len(t), n_vars_total))
    P = np.zeros(shape=(len(t), n_vars_total))
    P_noise = np.zeros(shape=(len(t), n_vars_total))

    def is_list(x):
        return isinstance(x, (tuple, list, np.ndarray))

    for i in range(n_vars):
        alpha_i = alpha[i] if is_list(alpha) and len(alpha) != n_obs else alpha
        beta_i = beta[i] if is_list(beta) and len(beta) != n_obs else beta
        gamma_i = gamma[i] if is_list(gamma) and len(gamma) != n_obs else gamma
        

        np.random.seed(random_seed + i)  # Ensure reproducibility with seed offset
        t_len = len(t)
        if t_len > 1:  # Only permute if tau has multiple elements
            index = np.random.randint(0, t_len // 4)  # Random index < len(tau)/4
            end_index = np.random.randint(t_len * 7 // 10, t_len * 4 // 5)
            t_rand = np.linspace(t[index], t[end_index], n_obs)
            # t_rand = np.concatenate((t[index:end_index], t[:index], t[end_index:]))  # Rearrange tau
            # t_rand = np.concatenate((t[index:], t[:index]))


        tau, alpha_vec, u0_vec, s0_vec = vectorize(
            t_rand, t_[i], alpha_i, beta_i, gamma_i, alpha_=alpha_, u0=0, s0=0
        )
        # tau, alpha_vec, u0_vec, s0_vec = vectorize(
        #     t, t_[i], alpha_i, beta_i, gamma_i, alpha_=alpha_, u0=0, s0=0
        # )
        
        
        if noise_model == "gillespie":
            U[:, i], S[:, i] = simulate_gillespie(alpha_vec, beta, gamma)
        else:
            U[:, i], S[:, i] = simulate_dynamics(
                tau,
                alpha_vec,
                beta_i,
                gamma_i,
                u0_vec,
                s0_vec,
                noise_model,
                noise_level[i],
                umi_depth
            )
            # shuffle_len = index + t_len - end_index
            # shuffle_len = index
            # for j in range(shuffle_len):
            #     S[n_obs-shuffle_len+j, i] = S[n_obs-shuffle_len-1, i]
            #     U[n_obs-shuffle_len+j, i] = U[n_obs-shuffle_len-1, i]
    # print(S[:,:n_vars])
    mrna = U+S
    mrna_counts = np.sum(np.sum(mrna))
    # cell_prob = np.sum(mrna, axis=1)/mrna_counts
    if is_list(umi_depth) and len(umi_depth) == n_obs:
        # umi_per_cell = np.array(umi_depth) * n_obs * cell_prob
        umi_per_cell = np.array(umi_depth)
    else:
        # umi_per_cell = umi_depth * n_obs * cell_prob
        umi_per_cell = umi_depth
    for cell in range(n_obs):  # Iterate over every cell
        if is_list(umi_depth) and len(umi_depth) == n_obs:
            total_umi_cell = umi_per_cell[cell]
        else:
            total_umi_cell = umi_per_cell
        ut_cell = U[cell, :n_vars] + 1e-6  ## alpha_u
        st_cell = S[cell, :n_vars] + 1e-6  ## alpha_s
        # print(st_cell)

        alpha_us = np.concatenate([ut_cell, st_cell])
        # size_per_hk = (mrna_counts * 10 / n_obs - np.sum(alpha_us))/ n_housekeeping
        size_per_hk = mrna_counts * 9 * 9/n_obs / n_housekeeping
        alpha_c = np.ones(n_housekeeping) * size_per_hk

        # print(alpha_c)
        alpha_total = np.concatenate([alpha_us, alpha_c])
        alpha_all = alpha_all + list(alpha_us)
        p = np.random.dirichlet(alpha_total)

        umi_counts = np.random.multinomial(int(total_umi_cell), p)

        U_noise[cell,:n_vars] = umi_counts[:n_vars]
        S_noise[cell,:n_vars] = umi_counts[n_vars:2*n_vars]

        U_noise[cell,n_vars:] = umi_counts[2*n_vars:] / 2
        S_noise[cell,n_vars:] = umi_counts[2*n_vars:] / 2

        U[cell, n_vars:] = alpha_c / 2
        S[cell, n_vars:] = alpha_c / 2
    if is_list(alpha) and len(alpha) == n_obs:
        alpha = np.nan
    if is_list(beta) and len(beta) == n_obs:
        beta = np.nan
    if is_list(gamma) and len(gamma) == n_obs:
        gamma = np.nan

    obs = {"true_t": t.round(2)}
    # print(alpha.shape)
    var = {
        "true_t_": np.concatenate([np.tile(t_[:n_vars],total_var//n_vars),t_[:total_var%n_vars]]),
        "true_alpha": np.concatenate([np.tile(np.array(alpha),total_var//n_vars),np.array(alpha)[:total_var%n_vars]]),
        "true_beta": np.concatenate([np.tile(np.ones(n_vars) * beta,total_var//n_vars),np.ones(total_var%n_vars) * beta]),
        "true_gamma": np.concatenate([np.tile(np.ones(n_vars) * gamma,total_var//n_vars),np.ones(total_var%n_vars) * gamma]),
    #     "true_scaling": np.ones(n_vars),
    }
    layers_true = {"unspliced": U, "spliced": S}
    layers_obs = {"unspliced": U_noise, "spliced": S_noise}

    return AnnData(S + U, obs,var=var, layers=layers_true), AnnData(S_noise + U_noise, obs,var=var, layers=layers_obs)

# calculate true velocity from ground truth data    
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


def write_fit_layers_from_pars(adata):
    to_dense = lambda x: x.A if hasattr(x, "A") else np.asarray(x)
    if "fit_t" not in adata.layers:
        raise KeyError("adata.layers['fit_t'] is missing; run recover_dynamics first.")
    fit_t = to_dense(adata.layers["fit_t"])
    n_cells, n_genes = fit_t.shape

    fit_u = np.full((n_cells, n_genes), np.nan, dtype=np.float32)
    fit_s = np.full_like(fit_u, np.nan)

    has_scaling = "fit_scaling" in adata.var
    has_u0 = "fit_u0" in adata.var
    has_s0 = "fit_s0" in adata.var

    for g, gene in enumerate(adata.var_names):
        sub = adata[:, gene]
        alpha_i, beta_i, gamma_i, scaling_i, t_switch = get_vars(sub, key="fit")
        alpha_i = float(np.atleast_1d(alpha_i)[0])
        beta_i = float(np.atleast_1d(beta_i)[0])
        gamma_i = float(np.atleast_1d(gamma_i)[0])
        scaling_i = float(np.atleast_1d(scaling_i)[0]) if has_scaling else 1.0
        t_switch = float(np.atleast_1d(t_switch)[0])

        u0_off = float(adata.var["fit_u0"].iat[g]) if has_u0 else 0.0
        s0_off = float(adata.var["fit_s0"].iat[g]) if has_s0 else 0.0

        t_vec = fit_t[:, g]
        if np.all(~np.isfinite(t_vec)):
            continue

        tau, alpha_vec, u0_vec, s0_vec = vectorize(
            t_vec, t_switch, alpha_i, beta_i, gamma_i,
            alpha_=0.0, u0=0.0, s0=0.0, sorted=False
        )
        ut, st = SplicingDynamics(
            alpha=alpha_vec, beta=beta_i, gamma=gamma_i,
            initial_state=[u0_vec, s0_vec]
        ).get_solution(tau, stacked=False)

        fit_u[:, g] = ut * scaling_i + u0_off
        fit_s[:, g] = st + s0_off

    adata.layers["fit_u"] = fit_u
    adata.layers["fit_s"] = fit_s
    
# write_fit_layers_from_pars(adata)

def align_gene_and_cell_time(adata):
    """
    Sort cells by pseudotime, reorder each gene's ``fit_t`` accordingly,
    and keep ``fit_u``/``fit_s`` aligned with the new ordering.

    Parameters:
        adata (AnnData): Dataset containing pseudotime and the fit layers.

    Returns:
        AnnData: A copy with time-aligned ``fit_u`` and ``fit_s`` layers.
    """
    # Ensure pseudotime exists
    if 'velocity_pseudotime' not in adata.obs:
        raise ValueError("'velocity_pseudotime' is missing from adata.obs.")

    # Ensure fit_t exists
    if 'fit_t' not in adata.layers:
        raise ValueError("'fit_t' is missing from adata.layers.")

    # Convert pseudotime to a NumPy array
    pseudotime = adata.obs['velocity_pseudotime'].to_numpy()

    # Sort cells by pseudotime
    sorted_cell_indices = np.argsort(pseudotime)
    adata = adata[sorted_cell_indices].copy()

    # Retrieve fit_t, fit_u, fit_s
    fit_t = adata.layers['fit_t']
    fit_u = adata.layers['fit_u']
    fit_s = adata.layers['fit_s']

    # Convert to dense matrices if necessary
    fit_t = fit_t.A if issparse(fit_t) else np.asarray(fit_t)
    fit_u = fit_u.A if issparse(fit_u) else np.asarray(fit_u)
    fit_s = fit_s.A if issparse(fit_s) else np.asarray(fit_s)

    # Initialize aligned outputs
    aligned_fit_u = np.zeros_like(fit_u)
    aligned_fit_s = np.zeros_like(fit_s)

    # For each gene, reorder fit_u/fit_s based on fit_t
    for gene_idx in range(fit_t.shape[1]):
        # Determine the time-based ordering for the gene
        sorted_time_indices = np.argsort(fit_t[:, gene_idx])

        # Apply the ordering to fit_u and fit_s
        aligned_fit_u[:, gene_idx] = fit_u[sorted_time_indices, gene_idx]
        aligned_fit_s[:, gene_idx] = fit_s[sorted_time_indices, gene_idx]

    # Store the aligned layers back into AnnData
    adata.layers['fit_u_aligned'] = aligned_fit_u
    adata.layers['fit_s_aligned'] = aligned_fit_s

    return adata

# adata = align_gene_and_cell_time(adata)