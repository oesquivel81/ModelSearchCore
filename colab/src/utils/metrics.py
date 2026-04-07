import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


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
