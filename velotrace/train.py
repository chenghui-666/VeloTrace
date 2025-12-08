import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
from .utils import set_seed
from .sample import create_batch
from .visualization import plot_scatter_with_ode

set_seed(4)

# ---- Training Function Definition ----
def update_odefunc(
    odefunc,
    odeblock,
    optimizer,
    criterion,
    trajectory_embeddings,
    velocity_values,
    t,
    log_interval=None,
    max_epochs=None,
    window=10,
    patience=20,
    save_interval=10,  # Newly added checkpoint interval
    min_grad_norm=1e-3,
    lambda_reg=1e-4,
    save_dir="./checkpoints",  # Newly added checkpoint directory
):
    # Ensure the checkpoint directory exists
    import os
    import copy
    os.makedirs(save_dir, exist_ok=True)
    
    if log_interval is None:
        log_interval = max(1, window)

    train_loss, reg_loss_list, recon_loss_list, param_reg_list = [], [], [], []
    best_loss = float("inf")
    epochs_no_improve = 0
    epoch = 0

    # Track the best checkpoint across the entire run
    global_best_loss = float("inf")
    best_ckpt = None

    while True:
        epoch += 1
        optimizer.zero_grad()
        loss = 0.0
        reg_loss_total = 0.0
        recon_loss_total = 0.0
        param_reg_total = 0.0

        repeat_count = 5
        for repeat in range(repeat_count):
            adata_block, t_block, v_block = create_batch(
                trajectory_embeddings,
                velocity_values,
                t,
                min_delta_time=0.01,
                sample_fraction=0.1,
            )
            y0 = adata_block[0].unsqueeze(0)
            pred_y = odeblock(y0, t_block)
            ode_velo = odefunc(0, adata_block)

            loss_reconstruction = criterion(pred_y.squeeze(), adata_block.detach())
            reg_loss = 1 - F.cosine_similarity(ode_velo, v_block, dim=0).mean()
            param_reg = (lambda_reg / repeat_count) * sum(param.pow(2).sum() for param in odefunc.parameters())

            current_loss = loss_reconstruction + reg_loss * 10 + param_reg * 50
            current_loss.backward()

            loss += current_loss.item()
            reg_loss_total += (reg_loss * 10).item()
            recon_loss_total += loss_reconstruction.item()
            param_reg_total += param_reg.item()

        optimizer.step()
        train_loss.append(loss)
        reg_loss_list.append(reg_loss_total)
        recon_loss_list.append(recon_loss_total)
        param_reg_list.append(param_reg_total)

        # Update the global best checkpoint
        if loss < global_best_loss:
            global_best_loss = loss
            best_ckpt = {
                'epoch': epoch,
                'odefunc_state_dict': copy.deepcopy(odefunc.state_dict()),
                'odeblock_state_dict': copy.deepcopy(odeblock.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'loss': loss,
            }

        # Periodically persist checkpoints
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'odefunc_state_dict': odefunc.state_dict(),
                'odeblock_state_dict': odeblock.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        if window and epoch % window == 0:
            grad_norm = 0.0
            for param in odeblock.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()

            if loss < best_loss - 1e-4:
                best_loss = loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stop at epoch {epoch}: loss no improve {epochs_no_improve} epochs")
                break
            if grad_norm < min_grad_norm:
                print(f"Early stop at epoch {epoch}: grad_norm={grad_norm:.6f}")
                break

        if log_interval and epoch % log_interval == 0:
            print(
                f"[ODEFunc] Epoch {epoch}, Loss: {loss:.4f}, Recon: {recon_loss_total:.4f}, "
                f"Reg: {reg_loss_total:.4f}, ParamReg: {param_reg_total:.4f}"
            )
            plot_scatter_with_ode(
                trajectory_embeddings.detach(),
                odefunc,
                odeblock,
                y0,
                t.detach(),
                num_blocks=1,
                num_imp=50,
                adata_velo=False,
            )

        if max_epochs is not None and epoch >= max_epochs:
            print(f"Reach max_epochs={max_epochs}, stop training")
            break

    # Persist the best checkpoint to final.pth; fall back to the last epoch if needed
    final_path = os.path.join(save_dir, "final.pth")
    if best_ckpt is None:
        best_ckpt = {
            'epoch': epoch,
            'odefunc_state_dict': odefunc.state_dict(),
            'odeblock_state_dict': odeblock.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
    torch.save(best_ckpt, final_path)
    print(f"Final (best) checkpoint saved to {final_path}, best_loss={global_best_loss:.6f}")

    print(f"Training stopped after {epoch} epochs.")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax, title, curve in zip(
        axes,
        ["Train Loss", "Reg Loss", "Recon Loss", "Param Reg"],
        [train_loss, reg_loss_list, recon_loss_list, param_reg_list],
    ):
        ax.plot(curve)
        ax.set_title(title)
    plt.show()
    
def update_t_only(
    odefunc,
    odeblock,
    criterion,
    trajectory_embeddings,
    velocity_values,
    t,
    device,
    max_epochs=300,
    lr=1e-2,
    save_dir="./checkpoints_t_only",
    save_interval=10,
    patience=50,
    min_delta=1e-4,
):
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Freeze network weights and x; only optimize t
    x = trajectory_embeddings.clone().detach().to(device)  # No gradients required
    t_optim = t.clone().detach().requires_grad_(True).to(device)
    velocity_values = velocity_values.to(device)
    optimizer = torch.optim.Adam([t_optim], lr=lr)

    loss_history = []
    best_loss = float("inf")
    epochs_no_improve = 0

    odefunc.eval()
    odeblock.eval()
    for p in odefunc.parameters():
        p.requires_grad_(False)
    for p in odeblock.parameters():
        p.requires_grad_(False)

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        epoch_loss = 0.0

        # Accumulate gradients over several stochastic samples
        repeat = 5
        for _ in range(repeat):
            x_batch, t_batch, v_batch = create_batch(
                x,
                velocity_values,
                t_optim,
                min_delta_time=0.02,
                sample_fraction=0.4,
            )
            y0 = x_batch[0].unsqueeze(0)
            pred_y = odeblock(y0, t_batch)     # Gradients flow only through t
            ode_velo = odefunc(0, x_batch)     # Independent of t, used for the regularizer

            recon_loss = criterion(pred_y.squeeze(), x_batch)
            reg_loss = (1 - F.cosine_similarity(ode_velo, v_batch, dim=0).mean()) * 10

            loss = recon_loss + reg_loss
            loss.backward()
            epoch_loss += (recon_loss.item() + reg_loss.item())

        optimizer.step()

        loss_history.append(epoch_loss)

        # Early stopping logic
        if epoch_loss + min_delta < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save the optimized t only
        if epoch % save_interval == 0 or epoch == max_epochs:
            t_path = os.path.join(save_dir, f"t_epoch_{epoch}.pt")
            torch.save(t_optim.detach().cpu(), t_path)
            print(f"Saved t to {t_path}")

        if epochs_no_improve >= patience:
            print(f"Early stop at epoch {epoch}: no improvement for {epochs_no_improve} epochs")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

    final_path = os.path.join(save_dir, "t_final.pt")
    torch.save(t_optim.detach().cpu(), final_path)
    print(f"Final t saved to {final_path}")

    return x.detach(), t_optim.detach(), loss_history

def update_odefunc_rec(
    odefunc,
    odeblock,
    optimizer,
    criterion,
    trajectory_embeddings,
    velocity_values,
    t,
    log_interval=None,
    max_epochs=None,
    window=10,
    patience=20,
    save_interval=10,
    min_grad_norm=1e-3,
    lambda_reg=1e-4,
    save_dir="./checkpoints",
):
    import os
    import copy
    os.makedirs(save_dir, exist_ok=True)

    if log_interval is None:
        log_interval = max(1, window)

    train_loss, recon_loss_list = [], []
    best_loss = float("inf")
    epochs_no_improve = 0
    epoch = 0

    global_best_loss = float("inf")
    best_ckpt = None

    while True:
        epoch += 1
        optimizer.zero_grad()
        loss = 0.0
        recon_loss_total = 0.0

        repeat_count = 5
        for repeat in range(repeat_count):
            adata_block, t_block, v_block = create_batch(
                trajectory_embeddings,
                velocity_values,
                t,
                min_delta_time=0.1,
                sample_fraction=0.07,
            )
            y0 = adata_block[0].unsqueeze(0)
            pred_y = odeblock(y0, t_block)

            loss_reconstruction = criterion(pred_y.squeeze(), adata_block.detach())
            current_loss = loss_reconstruction
            current_loss.backward()

            loss += current_loss.item()
            recon_loss_total += loss_reconstruction.item()

        optimizer.step()
        train_loss.append(loss)
        recon_loss_list.append(recon_loss_total)

        if loss < global_best_loss:
            global_best_loss = loss
            best_ckpt = {
                'epoch': epoch,
                'odefunc_state_dict': copy.deepcopy(odefunc.state_dict()),
                'odeblock_state_dict': copy.deepcopy(odeblock.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'loss': loss,
            }

        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'odefunc_state_dict': odefunc.state_dict(),
                'odeblock_state_dict': odeblock.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        if window and epoch % window == 0:
            grad_norm = 0.0
            for param in odeblock.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()

            if loss < best_loss - 1e-4:
                best_loss = loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stop at epoch {epoch}: loss no improve {epochs_no_improve} epochs")
                break
            if grad_norm < min_grad_norm:
                print(f"Early stop at epoch {epoch}: grad_norm={grad_norm:.6f}")
                break

        if log_interval and epoch % log_interval == 0:
            print(
                f"[ODEFunc] Epoch {epoch}, Loss: {loss:.4f}, Recon: {recon_loss_total:.4f}"
            )
            plot_scatter_with_ode(
                trajectory_embeddings.detach(),
                odefunc,
                odeblock,
                y0,
                t.detach(),
                num_blocks=1,
                num_imp=50,
                adata_velo=False,
            )

        if max_epochs is not None and epoch >= max_epochs:
            print(f"Reach max_epochs={max_epochs}, stop training")
            break

    final_path = os.path.join(save_dir, "final.pth")
    if best_ckpt is None:
        best_ckpt = {
            'epoch': epoch,
            'odefunc_state_dict': odefunc.state_dict(),
            'odeblock_state_dict': odeblock.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
    torch.save(best_ckpt, final_path)
    print(f"Final (best) checkpoint saved to {final_path}, best_loss={global_best_loss:.6f}")

    print(f"Training stopped after {epoch} epochs.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, title, curve in zip(
        axes,
        ["Train Loss", "Recon Loss"],
        [train_loss, recon_loss_list],
    ):
        ax.plot(curve)
        ax.set_title(title)
    plt.show()

def update_t_only_rec(
    odefunc,
    odeblock,
    criterion,
    trajectory_embeddings,
    velocity_values,
    t,
    device,
    max_epochs=300,
    lr=1e-2,
    save_dir="./checkpoints_t_only",
    save_interval=10,
    patience=50,
    min_delta=1e-4,
):
    import os
    os.makedirs(save_dir, exist_ok=True)

    x = trajectory_embeddings.clone().detach().to(device)
    t_optim = t.clone().detach().requires_grad_(True).to(device)
    velocity_values = velocity_values.to(device)
    optimizer = torch.optim.Adam([t_optim], lr=lr)

    loss_history = []
    best_loss = float("inf")
    epochs_no_improve = 0

    odefunc.eval()
    odeblock.eval()
    for p in odefunc.parameters():
        p.requires_grad_(False)
    for p in odeblock.parameters():
        p.requires_grad_(False)

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        epoch_loss = 0.0

        repeat = 5
        for _ in range(repeat):
            x_batch, t_batch, v_batch = create_batch(
                x,
                velocity_values,
                t_optim,
                min_delta_time=0.02,
                sample_fraction=0.4,
            )
            y0 = x_batch[0].unsqueeze(0)
            pred_y = odeblock(y0, t_batch)

            recon_loss = criterion(pred_y.squeeze(), x_batch)
            loss = recon_loss
            loss.backward()
            epoch_loss += recon_loss.item()

        optimizer.step()

        loss_history.append(epoch_loss)

        if epoch_loss + min_delta < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % save_interval == 0 or epoch == max_epochs:
            t_path = os.path.join(save_dir, f"t_epoch_{epoch}.pt")
            torch.save(t_optim.detach().cpu(), t_path)
            print(f"Saved t to {t_path}")

        if epochs_no_improve >= patience:
            print(f"Early stop at epoch {epoch}: no improvement for {epochs_no_improve} epochs")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

    final_path = os.path.join(save_dir, "t_final.pt")
    torch.save(t_optim.detach().cpu(), final_path)
    print(f"Final t saved to {final_path}")

    return x.detach(), t_optim.detach(), loss_history



