import os
import torch
import wandb
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import cv2

from .utils import *
from .losses import *
from .evaluation import *


__all__ = ['train_model']


def compute_losses(predicted_heatmaps, heatmaps, landmarks, image_shape, lmk_bbox_corners,
                   sample_weights=None, htm_mse_loss_weight=1.0, lmk_dist_loss_weight=0.0,
                   lmk_dist_loss_gamma=1.0, lmk_dist_loss_radius=0.1, reduce=True,
                   force_to_compute_all_losses=False):

    # Compute the heatmap MSE loss
    if force_to_compute_all_losses or htm_mse_loss_weight > 0:
        htm_mse_loss = heatmap_mse_loss(predicted_heatmaps, heatmaps, sample_weights=sample_weights, reduce=reduce)
    else:
        htm_mse_loss = 0

    # Compute the landmark distance loss (normalised by landmark bounding box size)
    if force_to_compute_all_losses or lmk_dist_loss_weight > 0:
        lmk_dist_loss = landmark_distance_loss(predicted_heatmaps, landmarks, image_shape,
                                               get_bbox_sizes(lmk_bbox_corners)[0],
                                               sample_weights=sample_weights, gamma=lmk_dist_loss_gamma,
                                               radius=lmk_dist_loss_radius, reduce=reduce)
    else:
        lmk_dist_loss = 0

    # Compute total loss
    total_loss = 0
    total_weight = 0
    if htm_mse_loss_weight > 0:
        total_loss += htm_mse_loss * htm_mse_loss_weight
        total_weight += htm_mse_loss_weight
    if lmk_dist_loss_weight > 0:
        total_loss += lmk_dist_loss * lmk_dist_loss_weight
        total_weight += lmk_dist_loss_weight
    if total_weight > 0:
        total_loss /= total_weight

    return total_loss, htm_mse_loss, lmk_dist_loss


def fgsm_attack(model, images, heatmaps, landmarks, lmk_bbox_corners, epsilon,
                htm_mse_loss_weight=1.0, lmk_dist_loss_weight=0.0,
                lmk_dist_loss_gamma=1.0, lmk_dist_loss_radius=0.1):
    was_training = model.train
    model.eval()
    delta = torch.zeros_like(images, requires_grad=True)
    predicted_heatmaps = model(images + delta)
    total_loss, _, _ = compute_losses(predicted_heatmaps[-1], heatmaps, landmarks, images.shape,
                                      lmk_bbox_corners, None, htm_mse_loss_weight, lmk_dist_loss_weight,
                                      lmk_dist_loss_gamma, lmk_dist_loss_radius)
    if isinstance(total_loss, torch.Tensor):
        total_loss.backward()
    if was_training:
        model.train()
    return (images + delta.grad.detach().sign() * epsilon).clamp(0, 1)


def pgd_linf_attack(model, images, heatmaps, landmarks, lmk_bbox_corners, epsilon, alpha, num_steps,
                    htm_mse_loss_weight=1.0, lmk_dist_loss_weight=0.0,
                    lmk_dist_loss_gamma=1.0, lmk_dist_loss_radius=0.1):
    was_training = model.train
    model.eval()
    delta = torch.rand_like(images) * epsilon * 2 - epsilon
    for _ in range(num_steps):
        delta.requires_grad_()
        predicted_heatmaps = model((images + delta).clamp(0, 1))
        total_loss, _, _ = compute_losses(predicted_heatmaps[-1], heatmaps, landmarks, images.shape,
                                          lmk_bbox_corners, None, htm_mse_loss_weight, lmk_dist_loss_weight,
                                          lmk_dist_loss_gamma, lmk_dist_loss_radius)
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()
        delta = (delta + delta.grad.sign() * alpha).clamp(-epsilon, epsilon).detach()
    if was_training:
        model.train()
    return (images + delta).clamp(0, 1)


def plot_landmarks_on_image(image, landmarks_pred, landmarks_gt):
    """Plot predicted and ground truth landmarks on the image using OpenCV."""
    # Convert image from torch tensor to numpy and denormalize
    img_np = image.transpose(1, 2, 0)
    # Scale from [0,1] to [0,255] range and convert to BGR for OpenCV
    img_np = (img_np * 255).astype(np.uint8)
    cv2.imshow('image', img_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # For predicted landmarks, use the first prediction (original image, not flipped)
    landmarks_pred_to_plot = landmarks_pred[0] if len(landmarks_pred.shape) == 3 else landmarks_pred
    
    # Convert normalized coordinates to pixel coordinates
    pred_points = np.column_stack([
        landmarks_pred_to_plot[:, 0],
        landmarks_pred_to_plot[:, 1]
    ]).astype(np.int32)
    
    gt_points = np.column_stack([
        landmarks_gt[:, 0],
        landmarks_gt[:, 1]
    ]).astype(np.int32)
    
    # Draw landmarks
    for point in pred_points:
        cv2.circle(img_np, tuple(point), 3, (255, 0, 0), -1)  # Blue for predicted
    
    for point in gt_points:
        cv2.circle(img_np, tuple(point), 3, (0, 0, 255), -1)  # Red for ground truth
    
    # Add legend directly on the image
    cv2.putText(img_np, 'Predicted', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(img_np, 'Ground Truth', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Save the image
    cv2.imwrite('landmarks_plot.png', img_np)
    
    # Convert back to RGB for matplotlib/wandb display
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    # Create a matplotlib figure (needed for wandb logging)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    return fig


def train_model(model, optimiser, train_data_loader, val_data_loader, num_epochs, log_folder, ckpt_folder,
                exp_name, lr_scheduler=None, val_per_epoch=1, save_top_k=1, rank_by='val_unweighted_macc',
                train_weight_norm=None, htm_mse_loss_weight=1.0, lmk_dist_loss_weight=0.0,
                lmk_dist_loss_gamma=1.0, lmk_dist_loss_radius=0.1, err_th=0.07, macc_num_ticks=100,
                adv_loss_weight=0.0, pgd_epsilon=0.01, pgd_alpha=0.003, pgd_num_steps=8,
                starting_epoch=0, pbar_epochs=None, pbar_train_batches=None, pbar_val_batches=None):
    # Initialize wandb
    wandb.init(
        project="landmark-detection",
        name=exp_name,
        config={
            "epochs": num_epochs,
            "val_per_epoch": val_per_epoch,
            "save_top_k": save_top_k,
            "rank_by": rank_by,
            "htm_mse_loss_weight": htm_mse_loss_weight,
            "lmk_dist_loss_weight": lmk_dist_loss_weight,
            "lmk_dist_loss_gamma": lmk_dist_loss_gamma,
            "lmk_dist_loss_radius": lmk_dist_loss_radius,
            "err_th": err_th,
            "macc_num_ticks": macc_num_ticks,
            "adv_loss_weight": adv_loss_weight,
            "pgd_epsilon": pgd_epsilon,
            "pgd_alpha": pgd_alpha,
            "pgd_num_steps": pgd_num_steps,
        }
    )

    if pbar_epochs is not None:
        pbar_epochs.reset(total=num_epochs)
        pbar_epochs.refresh()
    if pbar_train_batches is not None:
        pbar_train_batches.reset(total=len(train_data_loader))
        pbar_train_batches.refresh()
    if pbar_val_batches is not None:
        pbar_val_batches.reset(total=len(val_data_loader))
        pbar_val_batches.refresh()
    total_num_val = int(starting_epoch * val_per_epoch + 0.5)
    total_num_batch = int(starting_epoch * len(train_data_loader) + 0.5)
    best_snapshots = []
    ckpt_path = os.path.join(ckpt_folder, f"{exp_name}.ckpt")
    device = next(model.parameters()).device
    val_at_batches = [int(idx * len(train_data_loader) / val_per_epoch + 0.5) - 1
                      for idx in range(1, val_per_epoch + 1)]
    was_training = model.train
    model.train()
    for epoch in range(num_epochs):
        # Train the model
        if pbar_train_batches is not None:
            pbar_train_batches.reset(total=len(train_data_loader))
            pbar_train_batches.refresh()
        for batch_idx, (images, heatmaps, landmarks, _, lmk_bbox_corners, metadata) in enumerate(train_data_loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            if train_weight_norm > 0:
                sample_weights = (metadata['weight'] / train_weight_norm).float().to(device)
            else:
                sample_weights = None
            landmarks = landmarks.float().to(device)
            lmk_bbox_corners = lmk_bbox_corners.float().to(device)

            # Forward pass
            predicted_heatmaps = model(images)

            # Compute losses
            total_loss, htm_mse_loss, lmk_dist_loss = compute_losses(
                predicted_heatmaps, heatmaps, landmarks, images.shape, lmk_bbox_corners, sample_weights,
                htm_mse_loss_weight, lmk_dist_loss_weight, lmk_dist_loss_gamma, lmk_dist_loss_radius,
                force_to_compute_all_losses=True)

            # Add loss on adversarial examples
            adv_loss = None
            if adv_loss_weight > 0.0:
                adv_images = pgd_linf_attack(
                    model, images, heatmaps, landmarks, lmk_bbox_corners, pgd_epsilon, pgd_alpha, pgd_num_steps,
                    htm_mse_loss_weight, lmk_dist_loss_weight, lmk_dist_loss_gamma, lmk_dist_loss_radius)
                adv_predicted_heatmaps = model(adv_images)
                adv_loss, _, _ = compute_losses(
                    adv_predicted_heatmaps, heatmaps, landmarks, images.shape, lmk_bbox_corners, sample_weights,
                    htm_mse_loss_weight, lmk_dist_loss_weight, lmk_dist_loss_gamma, lmk_dist_loss_radius)
                total_loss = (total_loss + adv_loss * adv_loss_weight) / (1 + adv_loss_weight)

            # Update parameters
            optimiser.zero_grad()
            if isinstance(total_loss, torch.Tensor):
                total_loss.backward()
            optimiser.step()

            # Log progress to wandb
            total_num_batch += 1
            train_log = {
                'train/total_loss': total_loss,
                'train/step': total_num_batch,
                'train/epoch': epoch + (batch_idx + 1) / len(train_data_loader)
            }
            
            if adv_loss is not None:
                train_log['train/adv_loss'] = adv_loss
            
            if sample_weights is not None:
                train_log.update({
                    'train/heatmap_mse_loss_weighted': htm_mse_loss,
                    'train/landmark_distance_loss_weighted': lmk_dist_loss
                })
            else:
                train_log.update({
                    'train/heatmap_mse_loss': htm_mse_loss,
                    'train/landmark_distance_loss': lmk_dist_loss
                })
            
            wandb.log(train_log)

            if pbar_train_batches is not None:
                pbar_train_batches.update(1)
                pbar_train_batches.refresh()

            # At the end of an epoch, apply the learning rate scheduler
            if batch_idx + 1 == len(train_data_loader) and lr_scheduler is not None:
                lr_scheduler.step()

            # Validation
            if batch_idx in val_at_batches:
                val_results = run_model(model, val_data_loader, pbar=pbar_val_batches)

                # Create a snapshot
                last_snapshot = dict()
                if hasattr(model, 'module'):
                    last_snapshot['model_state_dict'] = deepcopy(model.module.state_dict())
                else:
                    last_snapshot['model_state_dict'] = deepcopy(model.state_dict())
                last_snapshot['optimiser_state_dict'] = deepcopy(optimiser.state_dict())
                if lr_scheduler is not None:
                    last_snapshot['lr_scheduler_state_dict'] = deepcopy(lr_scheduler.state_dict())
                last_snapshot['epoch'] = starting_epoch + epoch + (batch_idx + 1) / len(train_data_loader)

                # Compute various metrics
                lmk_bbox_sizes = get_bbox_sizes(val_results['landmark_bbox_corners'])[0]
                lmk_errors = compute_landmark_errors(val_results['predicted_landmarks'], val_results['landmarks'])
                last_snapshot['val_weighted_auc'] = compute_auc(lmk_errors, err_th, lmk_bbox_sizes,
                                                                val_results['sample_weights'])[0]
                last_snapshot['val_unweighted_auc'] = compute_auc(lmk_errors, err_th, lmk_bbox_sizes)[0]
                acc_ths = (np.arange(1, macc_num_ticks + 1)) * err_th / macc_num_ticks
                last_snapshot['val_weighted_macc'] = np.mean([compute_accuracy(lmk_errors, th, lmk_bbox_sizes,
                                                                               val_results['sample_weights'])
                                                              for th in acc_ths])
                last_snapshot['val_unweighted_macc'] = np.mean([compute_accuracy(lmk_errors, th, lmk_bbox_sizes)
                                                                for th in acc_ths])
                last_snapshot['val_weighted_htm_mse_loss'] = ((val_results['heatmap_mse_losses'].mean(axis=-1) *
                                                               val_results['sample_weights']).mean() /
                                                              val_results['sample_weights'].mean())
                last_snapshot['val_unweighted_htm_mse_loss'] = val_results['heatmap_mse_losses'].mean()
                normalised_lmk_loss = (val_results['landmark_distance_losses'].mean(axis=-1) /
                                       np.clip(lmk_bbox_sizes, np.finfo(lmk_bbox_sizes.dtype).eps, None))
                last_snapshot['val_weighted_lmk_dist_loss'] = ((normalised_lmk_loss *
                                                                val_results['sample_weights']).mean() /
                                                               val_results['sample_weights'].mean())
                last_snapshot['val_unweighted_lmk_dist_loss'] = normalised_lmk_loss.mean()

                # Log validation metrics to wandb
                total_num_val += 1
                val_log = {
                    f"val/auc_{err_th}_weighted": last_snapshot['val_weighted_auc'],
                    f"val/auc_{err_th}_unweighted": last_snapshot['val_unweighted_auc'],
                    f"val/macc_{err_th}_weighted": last_snapshot['val_weighted_macc'],
                    f"val/macc_{err_th}_unweighted": last_snapshot['val_unweighted_macc'],
                    "val/heatmap_mse_loss_weighted": last_snapshot['val_weighted_htm_mse_loss'],
                    "val/heatmap_mse_loss_unweighted": last_snapshot['val_unweighted_htm_mse_loss'],
                    "val/landmark_distance_loss_weighted": last_snapshot['val_weighted_lmk_dist_loss'],
                    "val/landmark_distance_loss_unweighted": last_snapshot['val_unweighted_lmk_dist_loss'],
                    "val/step": total_num_val,
                    "val/epoch": epoch + (batch_idx + 1) / len(train_data_loader)
                }

                # Visualize predictions on a few validation images
                num_viz_samples = min(5, len(val_results['images']))  # Visualize up to 5 samples
                for i in range(num_viz_samples):
                    fig = plot_landmarks_on_image(
                        val_results['images'][i],
                        val_results['predicted_landmarks'][i],
                        val_results['landmarks'][i]
                    )
                    val_log[f"val/sample_{i}"] = wandb.Image(fig)
                    plt.close(fig)

                wandb.log(val_log)

                # Model selection
                best_snapshots_updated = False
                if len(best_snapshots) < save_top_k:
                    best_snapshots.append(last_snapshot)
                    best_snapshots_updated = True
                elif ('loss' in rank_by and best_snapshots[-1][rank_by] > last_snapshot[rank_by] or
                      'loss' not in rank_by and best_snapshots[-1][rank_by] < last_snapshot[rank_by]):
                    best_snapshots[-1] = last_snapshot
                    best_snapshots_updated = True
                if best_snapshots_updated:
                    acc_order = np.argsort([ss[rank_by] for ss in best_snapshots])
                    if 'loss' not in rank_by:
                        acc_order = acc_order[::-1]
                    best_snapshots = [best_snapshots[idx] for idx in acc_order]

                # Update saved checkpoint
                os.makedirs(ckpt_folder, exist_ok=True)
                torch.save({'last_snapshot': last_snapshot, 'best_snapshots': best_snapshots}, ckpt_path)

                # Log best model to wandb
                if best_snapshots_updated:
                    wandb.run.summary[f"best_{rank_by}"] = best_snapshots[0][rank_by]
                    wandb.save(ckpt_path)

        if pbar_epochs is not None:
            pbar_epochs.update(1)
            pbar_epochs.refresh()

    if not was_training:
        model.eval()
    
    # Close wandb run
    wandb.finish()
    
    return ckpt_path
