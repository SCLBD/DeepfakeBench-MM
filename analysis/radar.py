import numpy as np
import matplotlib.pyplot as plt

def plot_radar_chart(data_groups, labels, categories=["FakeAVCeleb", "IDForge", "LAV-DF", "AVDeepfake1M", "Mega-MMDF"], output_path="radar_chart.png"):
    """
    Plot radar plot
    Args:
        data_groups: [list] data list, e.g., [[0.2, 0.8, 0.6, 0.4, 0.9], [0.5, 0.3, 0.7, 0.1, 0.6]]
        labels: [list] data label list, e.g., ["Group A", "Group B"]
    """
    # 检查输入数据合法性
    assert all(len(data) == 5 for data in data_groups), "每组数据必须包含5个值！"
    assert all(0 <= num <= 1 for data in data_groups for num in data), "数值必须在[0, 1]范围内！"

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

    for i, (data, label) in enumerate(zip(data_groups, labels)):
        data += data[:1]
        ax.plot(angles, data, linewidth=2, label=label, marker='o')
        ax.fill(angles, data, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)
    plt.xticks(angles[:-1], categories)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"])
    plt.ylim(0, 1)
    plt.title("Radar Chart Comparison", pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

#
if __name__ == "__main__":
    data_groups = [
        [0.745, 0.621, 0.523, 0.596, 0.950],
        [0.681, 0.626, 0.514, 0.538, 0.859],
        [0.889, 0.635, 0.852, 0.765, 0.997],
        [0.680, 0.531, 0.603, 0.574, 0.996]
    ]
    labels = ["MRDF", "AVTS", "AVFF", "Baseline"]

    plot_radar_chart(data_groups, labels, )