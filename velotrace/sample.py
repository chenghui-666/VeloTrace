import numpy as np
import torch
import pandas as pd
from .utils import set_seed

set_seed(4)

# ---- Sample Definition ----
def create_batch(
    trajectory_embeddings: torch.Tensor,
    velocity_values: torch.Tensor,
    t: torch.Tensor,
    min_delta_time: float,
    sample_fraction: float = 0.4,
):
    device = trajectory_embeddings.device
    num_cells = t.shape[0]

    required_count = max(1, int(num_cells * sample_fraction))
    candidate_order = torch.randperm(num_cells, device=t.device)

    # print(f"[create_batch] num_cells={num_cells}, required_count={required_count}, "
        #   f"min_delta_time={min_delta_time}, sample_fraction={sample_fraction}")
    # print(f"[create_batch] candidate_order head: {candidate_order[:10].tolist()}")

    selected_indices = []
    selected_times = []
    scanned = 0

    for idx in candidate_order:
        scanned += 1
        time_val = t[idx].item()
        if all(abs(time_val - existing) >= min_delta_time for existing in selected_times):
            selected_indices.append(idx.item())
            selected_times.append(time_val)
            if len(selected_indices) >= required_count:
                break

    if len(selected_indices) == 0:
        fallback = torch.argmin(t).item()
        selected_indices.append(fallback)
        print(f"[create_batch] fallback to argmin(t) index={fallback}")

    print(f"[create_batch] scanned={scanned}, selected={len(selected_indices)}")

    selected_indices_tensor = torch.tensor(selected_indices, device=device, dtype=torch.long)
    selected_t, reorder = torch.sort(t.index_select(0, selected_indices_tensor))
    selected_indices_tensor = selected_indices_tensor.index_select(0, reorder)

    # Log basic time statistics for the sampled batch
    with torch.no_grad():
        tt = selected_t.detach().float().cpu()
        diffs = tt[1:] - tt[:-1]
        min_gap_str = f"{diffs.min().item():.6f}" if diffs.numel() > 0 else "NA"
        tt_np = tt.numpy()
        print(f"[create_batch] t range: [{tt_np.min():.6f}, {tt_np.max():.6f}], min_gap_in_batch={min_gap_str}")
        if tt_np.size <= 10:
            print(f"[create_batch] t all: {np.round(tt_np, 6).tolist()}")
        else:
            print(f"[create_batch] t head: {np.round(tt_np[:5], 6).tolist()} ... tail: {np.round(tt_np[-5:], 6).tolist()}")

    adata_block = trajectory_embeddings.index_select(0, selected_indices_tensor)
    t_block = t.index_select(0, selected_indices_tensor)
    v_block = velocity_values.index_select(0, selected_indices_tensor)

    # print(f"[create_batch] block shapes: x={tuple(adata_block.shape)}, t={tuple(t_block.shape)}, v={tuple(v_block.shape)}")

    return adata_block, t_block, v_block

def create_batch_by_count_range(
    trajectory_embeddings: torch.Tensor,
    velocity_values: torch.Tensor,
    t: torch.Tensor,
    min_delta_time: float,
    count_range=(16, 128),
    sort_by_time: bool = True,
    verbose: bool = True,
):
    """
    Sample a batch based on a target count drawn from ``count_range`` while enforcing
    the ``min_delta_time`` constraint; ``sample_fraction`` is not used.
    Returns ``(adata_block, t_block, v_block)`` for backward compatibility.

    Args:
    - trajectory_embeddings: [N, D] cell embeddings
    - velocity_values:       [N, D] velocities (returned even if unused for compatibility)
    - t:                     [N] time values
    - min_delta_time:        minimum time gap between any two samples inside the batch
    - count_range:           (min_count, max_count), randomly draw the target batch size
    - sort_by_time:          whether to sort sampled items by time
    - verbose:               whether to log sampling information
    """
    device = trajectory_embeddings.device
    num_cells = t.shape[0]

    # Normalize the range and sample the target count
    lo = max(1, int(count_range[0]))
    hi = max(lo, int(count_range[1]))
    # Ensure the target count never exceeds the total number of samples
    target_count = min(
        int(torch.randint(low=lo, high=hi + 1, size=(1,), device=t.device).item()),
        int(num_cells),
    )

    candidate_order = torch.randperm(num_cells, device=t.device)

    selected_indices = []
    selected_times = []
    scanned = 0

    for idx in candidate_order:
        scanned += 1
        time_val = t[idx].item()
        if all(abs(time_val - existing) >= min_delta_time for existing in selected_times):
            selected_indices.append(int(idx))
            selected_times.append(time_val)
            if len(selected_indices) >= target_count:
                break

    # Fallback: if nothing qualifies, take the cell with the smallest time value
    if len(selected_indices) == 0:
        fallback = int(torch.argmin(t).item())
        selected_indices.append(fallback)
        if verbose:
            print(f"[create_batch_by_count_range] fallback to argmin(t) index={fallback}")

    if verbose:
        print(
            f"[create_batch_by_count_range] scanned={scanned}, "
            f"selected={len(selected_indices)}, target={target_count}, "
            f"min_delta_time={min_delta_time}"
        )

    selected_indices_tensor = torch.tensor(selected_indices, device=device, dtype=torch.long)

    if sort_by_time:
        selected_t, reorder = torch.sort(t.index_select(0, selected_indices_tensor))
        selected_indices_tensor = selected_indices_tensor.index_select(0, reorder)
    else:
        selected_t = t.index_select(0, selected_indices_tensor)

    # Log time statistics for the selected batch
    if verbose:
        with torch.no_grad():
            tt = selected_t.detach().float().cpu()
            diffs = tt[1:] - tt[:-1]
            min_gap_str = f"{diffs.min().item():.6f}" if diffs.numel() > 0 else "NA"
            tt_np = tt.numpy()
            if tt_np.size <= 10:
                print(f"[create_batch_by_count_range] t all: {np.round(tt_np, 6).tolist()}")
            else:
                print(
                    f"[create_batch_by_count_range] t head: {np.round(tt_np[:5], 6).tolist()} "
                    f"... tail: {np.round(tt_np[-5:], 6).tolist()}, min_gap_in_batch={min_gap_str}"
                )

    adata_block = trajectory_embeddings.index_select(0, selected_indices_tensor)
    t_block = t.index_select(0, selected_indices_tensor)
    v_block = velocity_values.index_select(0, selected_indices_tensor)

    return adata_block, t_block, v_block