import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model_v3 import mobilenet_v3_large
import logging

def setup_logging(log_file="../logs/experiment_log.txt"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    setup_logging()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using {} device.".format(device))

    batch_size = 32
    epochs = 300

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # 数据路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    logging.info("Data root: {}".format(data_root))
    image_path = os.path.join(data_root, "data_set")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # 训练集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    cla_dict = {val: key for key, val in flower_list.items()}
    with open('class_indices.json', 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    logging.info("Using {} dataloader workers per process".format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    # 验证集
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    logging.info("Using {} images for training, {} images for validation.".format(train_num, val_num))

    # 创建模型
    net = mobilenet_v3_large(num_classes=5)

    # 加载预训练权重（需确保权重文件存在）
    # model_weight_path = "./mobilenet_v3_large.pth"
    # assert os.path.exists(model_weight_path), "File {} does not exist.".format(model_weight_path)
    # pre_weights = torch.load(model_weight_path, map_location='cpu')
    # pre_dict = {k: v for k, v in pre_weights.items() if
    #             k in net.state_dict() and net.state_dict()[k].numel() == v.numel()}
    # missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    # logging.info("Pre-trained weights loaded. Missing keys: {}. Unexpected keys: {}."
    #              .format(missing_keys, unexpected_keys))

    # freeze features weights
    # for param in net.features.parameters():
    #     param.requires_grad = False

    net.to(device)

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = './best_mobilenet_v3_large.pth'

    # 用于记录每个epoch的指标
    metrics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    logging.info("Start training for {} epochs.".format(epochs))
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, (images, labels) in enumerate(train_bar):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_bar.desc = "Train Epoch [{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss.item())

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        net.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for (val_images, val_labels) in val_bar:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                outputs = net(val_images)
                loss = loss_function(outputs, val_labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += val_labels.size(0)
                correct_val += (predicted == val_labels).sum().item()
                val_bar.desc = "Valid Epoch [{}/{}]".format(epoch + 1, epochs)
        val_loss = val_loss / len(validate_loader)
        val_acc = correct_val / total_val

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        epoch_log = "[Epoch {}] train_loss: {:.3f}, train_acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}" \
            .format(epoch + 1, train_loss, train_acc, val_loss, val_acc)
        logging.info(epoch_log)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
            logging.info("New best model saved with val_acc: {:.3f}".format(best_acc))

    # 保存指标记录
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    logging.info("Finished Training. Best val_acc: {:.3f}".format(best_acc))
    logging.info("Metrics saved to metrics.json and best model saved to {}".format(save_path))


if __name__ == '__main__':
    main()
