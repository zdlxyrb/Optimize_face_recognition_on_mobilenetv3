import os
import time
import torch
from innovation_model_v3 import mobilenet_v3_large
from thop import profile

def model_efficiency():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 创建模型并加载权重（请确保权重文件存在）
    model = mobilenet_v3_large(num_classes=5)
    model_weight_path = "./best_innovation_mobilenet_v3_large.pth"
    assert os.path.exists(model_weight_path), "File {} does not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')
    pre_dict = {k: v for k, v in pre_weights.items() if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(pre_dict, strict=False)
    model.to(device)
    model.eval()

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Params: {:.2f}M".format(total_params / 1e6))

    # 使用thop计算FLOPs
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    print("FLOPs: {:.2f}G".format(flops / 1e9))

    # 计算推理时间（ms/帧）
    iterations = 100
    with torch.no_grad():
        start_time = time.time()
        for _ in range(iterations):
            _ = model(dummy_input)
        end_time = time.time()
    avg_inference_time = (end_time - start_time) * 1000 / iterations
    print("Average inference time: {:.2f} ms/frame".format(avg_inference_time))

    # 内存占用（仅限GPU，单位MB）
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(dummy_input)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print("Peak Memory Usage: {:.2f} MB".format(peak_memory))
    else:
        print("Peak Memory Usage: N/A (CPU)")

if __name__ == '__main__':
    model_efficiency()
