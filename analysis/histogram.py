import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 示例数据 - 用你的实际数据替换这些
predictions_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.6, 0.4, 0.9, 0.3, 0.45])
labels = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])

hist_pos, bins = np.histogram(predictions_prob[labels == 1], bins=100, density=True)
hist_neg, _ = np.histogram(predictions_prob[labels == 0], bins=bins, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.bar(bin_centers, hist_pos, width=bins[1]-bins[0], alpha=0.5, color='red', label='Positive class')
plt.bar(bin_centers, hist_neg, width=bins[1]-bins[0], alpha=0.5, color='blue', label='Negative class')

kde_pos = gaussian_kde(predictions_prob[labels == 1])
kde_neg = gaussian_kde(predictions_prob[labels == 0])
x_grid = np.linspace(0, 1, 100)
plt.plot(x_grid, kde_pos(x_grid), color='red', lw=2, linestyle='--', label='Pos KDE')
plt.plot(x_grid, kde_neg(x_grid), color='blue', lw=2, linestyle='--', label='Neg KDE')

plt.xlabel('Predicted Probability')
plt.ylabel('Density/Proportion')
plt.title('Prediction Distribution by Class (Density)')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()