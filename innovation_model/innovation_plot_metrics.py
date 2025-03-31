import json
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics():
    # 设置图像分辨率和 PDF 字体内嵌（适用于论文投稿）
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Times New Roman'

    # 设置 seaborn 样式及上下文，确保文字大小适合论文排版
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid", {
        'axes.edgecolor': 'black',
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
    })

    # 定义颜色：
    # 损失曲线使用蓝色和橙色
    color_loss_train = '#1f77b4'  # 深蓝色
    color_loss_val = '#ff7f0e'  # 橙色

    # 准确率曲线使用绿色和红色
    color_acc_train = '#2ca02c'  # 绿色
    color_acc_val = '#d62728'  # 红色

    # 加载训练过程中记录的指标
    with open('innovation_metrics.json', 'r') as f:
        metrics = json.load(f)

    epochs = range(1, len(metrics["train_loss"]) + 1)

    # 1) 损失函数曲线：训练损失 vs 验证损失
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, metrics["train_loss"], color=color_loss_train, linestyle='-', marker='o',
             markersize=4, linewidth=1.5, label='Training Loss')
    plt.plot(epochs, metrics["val_loss"], color=color_loss_val, linestyle='-', marker='s',
             markersize=4, linewidth=1.5, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig('../innovation_results/loss_curve.png', bbox_inches='tight')
    plt.show()

    # 2) 准确率曲线：训练准确率 vs 验证准确率
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, metrics["train_acc"], color=color_acc_train, linestyle='-', marker='o',
             markersize=4, linewidth=1.5, label='Training Accuracy')
    plt.plot(epochs, metrics["val_acc"], color=color_acc_val, linestyle='-', marker='s',
             markersize=4, linewidth=1.5, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig('../innovation_results/accuracy_curve.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_metrics()
