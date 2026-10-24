"""Soft-margin triplet objective used by the OASIS metric training."""

import torch.nn.functional as F


def soft_margin_triplet_loss(model, anchor_feats, pos_feats, neg_feats, target_layers):
    if not target_layers or not (
        len(target_layers) == len(anchor_feats) == len(pos_feats) == len(neg_feats)
    ):
        raise ValueError("Feature lists must match the nonempty target layer list.")
    loss = 0.0
    count = 0
    for layer_id, anchor, positive, negative in zip(
        target_layers, anchor_feats, pos_feats, neg_feats
    ):
        positive_distance = model.calc_distance(anchor, positive, layer_id)
        negative_distance = model.calc_distance(anchor, negative, layer_id)
        loss += F.softplus(positive_distance - negative_distance)
        count += 1
    return loss / (count + 1e-8)
