import torch
import torch.nn.functional as F


def _iter_layer_features(model, target_layers, anchor_feats, pos_feats, neg_feats):
    feature_sets = (anchor_feats, pos_feats, neg_feats)
    if target_layers is None:
        target_layers = getattr(model, "target_layers", range(len(anchor_feats)))
    target_layers = tuple(target_layers)
    if not target_layers:
        raise ValueError("At least one target layer is required")
    for layer_id in target_layers:
        if not isinstance(layer_id, int):
            raise TypeError(f"Layer indices must be integers, got {type(layer_id).__name__}")
        if layer_id < 0 or any(layer_id >= len(features) for features in feature_sets):
            lengths = tuple(len(features) for features in feature_sets)
            raise IndexError(f"Layer {layer_id} is unavailable for feature lengths {lengths}")
        yield layer_id, anchor_feats[layer_id], pos_feats[layer_id], neg_feats[layer_id]



# ---------------- 1. Triplet Loss (Hard Margin) ----------------
def triplet_loss(model, anchor_feats, pos_feats, neg_feats, margin=1.0, target_layers=None):
    loss = 0.0
    count = 0
    for layer_id, fa, fp, fn in _iter_layer_features(model, target_layers, anchor_feats, pos_feats, neg_feats):
        # Compute positive and negative distances with the stage-specific weights.
        dist_ap = model.calc_distance(fa, fp, layer_id)
        dist_an = model.calc_distance(fa, fn, layer_id)

        # Loss = max(dp - dn + margin, 0)
        loss += torch.relu(dist_ap - dist_an + margin)
        count += 1

    return loss / (count + 1e-8)


# ---------------- 2. Soft Margin Triplet Loss (Recommended) ----------------
def soft_margin_triplet_loss(model, anchor_feats, pos_feats, neg_feats, margin=0.0, target_layers=None):
    loss = 0.0
    count = 0
    for layer_id, fa, fp, fn in _iter_layer_features(model, target_layers, anchor_feats, pos_feats, neg_feats):
        dist_ap = model.calc_distance(fa, fp, layer_id)
        dist_an = model.calc_distance(fa, fn, layer_id)

        # Loss = ln(1 + exp(dp - dn))
        loss += F.softplus(dist_ap - dist_an + margin)
        count += 1

    return loss / (count + 1e-8)


# ---------------- 3. Circle Loss ----------------
def circle_loss(model, anchor_feats, pos_feats, neg_feats, margin=0.25, gamma=256, target_layers=None):
    loss = 0.0
    count = 0
    for layer_id, fa, fp, fn in _iter_layer_features(model, target_layers, anchor_feats, pos_feats, neg_feats):
        dist_ap = model.calc_distance(fa, fp, layer_id)
        dist_an = model.calc_distance(fa, fn, layer_id)

        sp = -dist_ap
        sn = -dist_an

        alpha_p = torch.relu(-sp + 1 + margin)
        alpha_n = torch.relu(sn + margin)

        delta_p = 1 - margin
        delta_n = margin

        logit_p = - alpha_p * (sp - delta_p) * gamma
        logit_n = alpha_n * (sn - delta_n) * gamma

        loss += F.softplus(logit_p + logit_n)
        count += 1

    return loss / (count + 1e-8)


# ---------------- Unified loss API ----------------
def get_loss(loss_type, model, anchor_feats, pos_feats, neg_feats, target_layers=None, **kwargs):
    if loss_type == "triplet":
        return triplet_loss(model, anchor_feats, pos_feats, neg_feats,
                            margin=kwargs.get("margin", 1.0),
                            target_layers=target_layers)
    elif loss_type == "soft_margin_triplet":
        return soft_margin_triplet_loss(model, anchor_feats, pos_feats, neg_feats,
                                        margin=kwargs.get("margin", 0.0),
                                        target_layers=target_layers)
    elif loss_type == "circle":
        return circle_loss(model, anchor_feats, pos_feats, neg_feats,
                           margin=kwargs.get("margin", 0.25),
                           gamma=kwargs.get("gamma", 256),
                           target_layers=target_layers)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
