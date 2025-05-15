import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import precision_recall_curve, average_precision_score


def binary_AP(predictions, labels, save_path=None, **kwargs):
    """
    In binary classification case. calculate AP and save P-R curve if save_path provided.
    Args:
        predictions: [np.ndarray] predicted probability or logit
        labels: [np.ndarray] ground truth label, with values 0s or 1s
        save_path: [str, default=None] ROC curve saving root if provided
    Returns:
        ap_score: [float] AP value
    """
    precision, recall, _ = precision_recall_curve(labels, predictions)
    ap_score = average_precision_score(labels, predictions)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap_score:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(save_path, 'PR_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return ap_score

if __name__ == '__main__':
    test_predictions = np.array([0.1, 0.4, 0.35, 0.8])
    test_labels = np.array([0, 0, 1, 1])
    test_eer = binary_AP(test_predictions, test_labels, './temp')
    print(f'Average Precision = {test_eer:.6f}')