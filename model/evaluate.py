import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, average_precision_score
import seaborn as sns
from model_v3 import mobilenet_v3_large

def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    # 数据预处理（验证集使用）
    data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    image_path = os.path.join(data_root, "data_set")
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform)
    val_loader = torch.utils.data.DataLoader(validate_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)
    class_names = validate_dataset.classes
    num_classes = len(class_names)

    # 加载模型与最佳权重
    model = mobilenet_v3_large(num_classes=num_classes)
    checkpoint_path = './best_mobilenet_v3_large.pth'
    assert os.path.exists(checkpoint_path), "Checkpoint does not exist."
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 3) 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("../results/confusion_matrix.png")
    plt.show()

    # 4) F1分数分布图（各类别F1分数）
    f1_scores = f1_score(all_labels, all_preds, average=None)
    plt.figure()
    bars = plt.bar(class_names, f1_scores, color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.title("F1 Score per Class")
    plt.ylim(0, 1)

    # 在每个柱状图上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}',
                 ha='center', va='bottom')

    plt.savefig("../results/f1_scores.png")
    plt.show()

    # 5) 精确率-召回率（PR）曲线：针对每个类别
    plt.figure()
    for i in range(num_classes):
        # 将标签二值化（当前类别为正类，其余为负类）
        true_binary = (all_labels == i).astype(int)
        prob_scores = all_probs[:, i]
        precision, recall, _ = precision_recall_curve(true_binary, prob_scores)
        ap = average_precision_score(true_binary, prob_scores)
        plt.plot(recall, precision, label=f'{class_names[i]} (AP={ap:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("../results/pr_curve.png")
    plt.show()

if __name__ == '__main__':
    evaluate()
