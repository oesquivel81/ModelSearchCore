import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


def classification_metrics_from_logits(logits, targets, num_classes):
    preds = logits.argmax(dim=1)
    acc = (preds == targets).float().mean().item()

    prec_per_class = []
    rec_per_class = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        prec = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        prec_per_class.append(prec)
        rec_per_class.append(rec)

    precision_macro = float(np.mean(prec_per_class))
    recall_macro = float(np.mean(rec_per_class))
    f1_macro = (2 * precision_macro * recall_macro) / (precision_macro + recall_macro + 1e-8)

    return {
        "acc": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": float(f1_macro),
    }


def dice_from_probs(preds, targets, eps=1e-8):
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()


def iou_from_probs(preds, targets, eps=1e-8):
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


def precision_from_probs(preds, targets, eps=1e-8):
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    return ((tp + eps) / (tp + fp + eps)).mean().item()


def recall_from_probs(preds, targets, eps=1e-8):
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    return ((tp + eps) / (tp + fn + eps)).mean().item()


def f1_from_precision_recall(precision, recall, eps=1e-8):
    return (2 * precision * recall) / (precision + recall + eps)


def hausdorff_distance_binary(pred, target):
    pred_pts = np.argwhere(pred > 0)
    tgt_pts = np.argwhere(target > 0)
    if len(pred_pts) == 0 or len(tgt_pts) == 0:
        return np.nan
    d1 = directed_hausdorff(pred_pts, tgt_pts)[0]
    d2 = directed_hausdorff(tgt_pts, pred_pts)[0]
    return max(d1, d2)
