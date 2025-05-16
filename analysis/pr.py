import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 示例数据 - 用你的实际数据替换这些
predictions_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.6, 0.4, 0.9, 0.3, 0.45])
labels = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])

precision, recall, _ = precision_recall_curve(labels, predictions_prob)
ap = average_precision_score(labels, predictions_prob)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()