from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def evaluate_image_rocauc(total_roc_auc, gt_list, scores, print=True):
    """
    Compute the ROC AUC for a class and append it to total_roc_auc.
    Args:
        total_roc_auc (list): List of ROC AUCs for all classes.
        gt_list (list): List of ground truth images.
        scores (list): List of predicted scores."""
    fpr, tpr, _ = roc_curve(gt_list, scores)
    roc_auc = roc_auc_score(gt_list, scores)
    total_roc_auc.append(roc_auc)
    return roc_auc, fpr, tpr

def evaluate_pixel_rocauc(total_pixel_roc_auc, flatten_gt_mask_list, flatten_score_map_list, print=True):
    """
    Compute the ROC AUC for a class and append it to total_roc_auc.
    Args:
        total_pixel_roc_auc (list): List of pixel ROC AUCs for all classes.
        flatten_gt_mask_list (list): List of ground truth images.
        flatten_score_map_list (list): List of predicted scores for each pixel."""
    fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
    per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
    total_pixel_roc_auc.append(per_pixel_rocauc)
    return per_pixel_rocauc, fpr, tpr