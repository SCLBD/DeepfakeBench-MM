import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

predictions_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.6, 0.4, 0.9, 0.3, 0.45])
labels = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])

fpr, tpr, _ = roc_curve(labels, predictions_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()