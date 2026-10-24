"""Train frozen-backbone nnInteractive channel weights on OASIS."""
import argparse
import math
from config import (
    DEFAULT_DATA_ROOT, DEFAULT_BACKBONE_DIR, DEFAULT_TRAIN_OUTPUT, resolve_path,
)
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.nnInteractive_metric import nnInteractiveMetric
from data.oasis import OASISRaw3DDataset
from utils.deformation_transform import random_deform, apply_rigid_transform
from utils.loss import soft_margin_triplet_loss

def train_and_validate(args):
    DEVICE = args.device
    DATA_ROOT = resolve_path(args.data_root)
    SAVE_DIR = resolve_path(args.output_dir)
    LOCAL_WEIGHTS_DIR = str(resolve_path(args.backbone_dir))
    os.makedirs(SAVE_DIR, exist_ok=True)
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    LR_MILESTONES = [200, 275, 350, 425]
    EARLY_STOP_PATIENCE = 50
    POS_MAG_VAL = 2.0
    NEG_MAG_VAL = 4.0
    train_dataset = OASISRaw3DDataset(DATA_ROOT, mode='train')
    val_dataset = OASISRaw3DDataset(DATA_ROOT, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print('Training nnInteractive channel weights with soft-margin triplet loss')
    model = nnInteractiveMetric(weights_dir=LOCAL_WEIGHTS_DIR, requires_grad=True, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=0.1)
    active_layers = list(range(model.num_total_stages))
    best_train_loss = float('inf')
    early_stop_counter = 0
    best_epoch = -1
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, anchor in enumerate(train_loader):
            anchor = anchor.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.no_grad():
                if torch.rand(1).item() < 0.5:
                    pos_mag = torch.rand(1).item() * 2.0
                    pos = random_deform(anchor, geo_mag=pos_mag)
                    neg_mag = torch.rand(1).item() * 6.0 + 2.0
                    neg = random_deform(anchor, geo_mag=neg_mag)
                else:
                    rigid_type = 'translation' if torch.rand(1).item() < 0.5 else 'rotation'
                    if rigid_type == 'translation':
                        pos_trans = torch.rand(1).item() * 1.0
                        pos = apply_rigid_transform(anchor, 'translation', translate_voxels=pos_trans)
                        neg_trans = torch.rand(1).item() * 9.0 + 1.0
                        neg = apply_rigid_transform(anchor, 'translation', translate_voxels=neg_trans)
                    else:
                        pos_sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
                        pos_rot = torch.rand(1).item() * 3.0 * pos_sign
                        pos = apply_rigid_transform(anchor, 'rotation', rotate_angle=pos_rot)
                        neg_sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
                        neg_rot = (torch.rand(1).item() * 7.0 + 3.0) * neg_sign
                        neg = apply_rigid_transform(anchor, 'rotation', rotate_angle=neg_rot)
            feat_a = model(anchor)
            feat_p = model(pos)
            feat_n = model(neg)
            loss = soft_margin_triplet_loss(model, feat_a, feat_p, feat_n, active_layers)
            if not torch.isfinite(loss):
                print(f'Non-finite loss! Epoch {epoch} Batch {i}')
                sys.exit(1)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for layer in model.lin_layers.values():
                    layer.weight.clamp_(min=1e-05)
            total_loss += loss.item()
            if i % 10 == 0:
                print(f'Epoch[{epoch}] Batch[{i}/{len(train_loader)}] Loss: {loss.item():.4f}')
        avg_loss = total_loss / len(train_loader)
        val_acc, val_gap = validate(model, val_loader, POS_MAG_VAL, NEG_MAG_VAL, active_layers, DEVICE)
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {avg_loss:.4f}')
        print(f'  Val Acc:    {val_acc * 100:.2f}% (Pos={POS_MAG_VAL}, Neg={NEG_MAG_VAL})')
        print(f'  Val Gap:    {val_gap:.5f}')
        current_save_path = f'{SAVE_DIR}/epoch_{epoch}_acc_{val_acc:.2f}.pth'
        torch.save(model.lin_layers.state_dict(), current_save_path)
        print(f'  Model channel weights saved to {current_save_path}')
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            early_stop_counter = 0
            best_epoch = epoch
            best_path = os.path.join(SAVE_DIR, 'best_train_loss_model.pth')
            torch.save(model.lin_layers.state_dict(), best_path)
            print(f'  [New Best Train Loss!] Saved to {best_path}')
        else:
            early_stop_counter += 1
            print(f'  [Early Stop Counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}]')
        scheduler.step()
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f'\n>>> Early stopping triggered. Best Train Loss was {best_train_loss:.4f} at epoch {best_epoch}.')
            break

def validate(model, loader, pos_mag, neg_mag, target_layers, device):
    model.eval()
    correct = 0
    total = 0
    gap_sum = 0
    with torch.no_grad():
        for anchor in loader:
            anchor = anchor.to(device, non_blocking=True)
            pos = random_deform(anchor, geo_mag=pos_mag)
            neg = random_deform(anchor, geo_mag=neg_mag)
            feat_a = model(anchor)
            feat_p = model(pos)
            feat_n = model(neg)
            dist_p_total = 0
            dist_n_total = 0
            for i, layer_id in enumerate(target_layers):
                dist_p_total += model.calc_distance(feat_a[i], feat_p[i], layer_id).item()
                dist_n_total += model.calc_distance(feat_a[i], feat_n[i], layer_id).item()
            dist_p_total /= len(target_layers)
            dist_n_total /= len(target_layers)
            if dist_p_total < dist_n_total:
                correct += 1
            gap_sum += dist_n_total - dist_p_total
            total += 1
    if total == 0:
        return (0.0, 0.0)
    return (correct / total, gap_sum / total)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--backbone-dir", default=DEFAULT_BACKBONE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0 or args.num_workers < 0:
        parser.error("epochs, batch-size and lr must be positive; num-workers must be nonnegative")
    return args


if __name__ == "__main__":
    train_and_validate(parse_args())
