import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve


def binary_EER(predictions, labels, save_path=None, **kwargs):
    """
    In binary classification case. calculate EER and save ROC curve if save_path provided.
    Args:
        predictions: [np.ndarray] predicted probability or logit
        labels: [np.ndarray] ground truth label, with values 0s or 1s
        save_path: [str, default=None] ROC curve saving root if provided
    Returns:
        eer: [float] EER value
    """
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(eer, 1 - eer, 'ro', label=f'EER = {eer:.2f}\n(Threshold = {eer_threshold:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve with Equal Error Rate (EER)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, 'ROC_curve_EER.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return eer

if __name__ == '__main__':
    test_predictions = np.array([0.1, 0.4, 0.35, 0.8])
    test_labels = np.array([0, 0, 1, 1])
    test_eer = binary_EER(test_predictions, test_labels, './temp')
    print(f'EER = {test_eer:.6f}')