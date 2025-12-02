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

    # 打印批内时间统计信息
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
