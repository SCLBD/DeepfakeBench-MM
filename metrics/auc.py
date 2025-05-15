import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, auc


def binary_AUC(predictions, labels, save_path=None, **kwargs):
    """
    In binary classification case. calculate AUC and save ROC curve if save_path provided.
    Args:
        predictions: [np.ndarray] predicted probability or logit
        labels: [np.ndarray] ground truth label, with values 0s or 1s
        save_path: [str, default=None] ROC curve saving root if provided
    Returns:
        auc_score: [float] AUC value
    """
    fpr, tpr, _ = roc_curve(labels, predictions)
    auc_score = auc(fpr, tpr)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, 'ROC_curve_AUC.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return auc_score

if __name__ == '__main__':
    test_predictions = np.array([0.1, 0.4, 0.35, 0.8])
    test_labels = np.array([0, 0, 1, 1])
    test_auc = binary_AUC(test_predictions, test_labels, './temp')
    print(f'AUC = {test_auc:.6f}')