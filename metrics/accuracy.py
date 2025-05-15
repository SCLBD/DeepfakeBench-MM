import numpy as np

from sklearn.metrics import accuracy_score

def binary_accuracy(predictions, labels, threshold=None, **kwargs):
    """
    In binary classification case. calculate accuracy given threshold value, if given. Otherwise, find out the optimal
        threshold value and calculate accuracy.
    Args:
        predictions: [np.ndarray] predicted probability or logit
        labels: [np.ndarray] ground truth label, with values 0s or 1s
        threshold: [float, default=None] threshold used for accuracy calculation
    Returns:
        accuracy: [float] accuracy value
    """
    assert isinstance(predictions, np.ndarray)

    # Given threshold, directly calculate accuracy
    if threshold is not None:
        pred_labels = (predictions >= threshold).astype(int)
        accuracy = accuracy_score(labels, pred_labels)
        return accuracy
    # Otherwise, find out optimal threshold and accuracy
    else:
        if np.min(predictions) >= 0 and np.max(predictions) <= 1:
            mid = 0.5
            threshold_candidates = np.linspace(0, 1, 100)
        else:
            mid = (np.min(predictions) + np.max(predictions)) / 2
            threshold_candidates = np.sort(np.unique(predictions))
        best_threshold, best_accuracy = 0, 0
        for threshold_candidate in threshold_candidates:
            pred_labels = (predictions >= threshold_candidate).astype(int)
            accuracy = accuracy_score(labels, pred_labels)
            if accuracy > best_accuracy:
                best_threshold = threshold_candidate
                best_accuracy = accuracy
            elif accuracy == best_accuracy: # choose threshold closed to (min + max) / 2, i.e., 0.5 for [0, 1] case
                if abs(threshold_candidate - mid) < abs(best_threshold - mid):
                    best_threshold = threshold_candidate
        print(f'Optimal accuracy threshold: {best_threshold:.6f}')
        return best_accuracy


if __name__ == '__main__':
    test_predictions = np.array([0.1, 0.4, 0.35, 0.8])
    test_labels = np.array([0, 0, 1, 1])
    test_accuracy = binary_accuracy(test_predictions, test_labels)
    print(f'Without threshold provided, accuracy = {test_accuracy:.6f}')
    test_accuracy = binary_accuracy(test_predictions, test_labels, 0.5)
    print(f'With threshold = 0.5, accuracy = {test_accuracy:.6f}')