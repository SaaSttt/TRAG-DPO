import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import configs as configs
logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}

# 这里假设你已经定义并导入了 CONFIGS 字典和 VisionTransformer 类



def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 你训练时如果做了归一化，需要在这里添加normalize
        # transforms.Normalize(mean=[...], std=[...])
    ])
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # 增加batch维度
    return image, img_tensor
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
def visualize_focused_patches(
    orig_image,
    part_inx,
    config_name='ViT-B_16',        # 配置名称
    save_path='/data/datasets/zhuhaoran/CDDMBench/LLaVA-1.1.3/clipmain/models/ksh.jpg',
    treat_index_as='token',        # 'token' 表示含 CLS 偏移，需要减 1；'patch' 表示已是 0..N-1
    topk=None,                     # 例如 4：只画前 4 个索引；None：全画
    dedup=True,                    # 是否去重
):
    # 1) 获取 config
    config = CONFIGS[config_name]
    patch_size = config.patches.size[0]  # (H, W)，取H
    slide_step = config.slide_step

    img_w, img_h = orig_image.size
    patches_per_row = (img_w - patch_size +slide_step) // slide_step
    patches_per_col = (img_h - patch_size+slide_step) // slide_step
    num_patches = patches_per_row * patches_per_col
    print(part_inx[0])
    print(num_patches)

    idxs = part_inx[0].detach().cpu().numpy().astype(int)

    # 3) 若是 token 索引（包含 CLS=0），映射到 patch 索引
    if treat_index_as == 'token':
        idxs = idxs - 1  # CLS 对齐 -> patch 索引

    # 限定在合法范围
    idxs = idxs[(idxs >= 0) & (idxs < num_patches)]

    # 4) 去重（可选）
    if dedup:
        idxs = np.unique(idxs)

    # 5) 只取前 topk（可选）
    if topk is not None and topk > 0:
        idxs = idxs[:topk]

    # 6) 绘图
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(orig_image)
    ax.axis('off')

    for idx in idxs:
        row = idx // patches_per_row
        col = idx % patches_per_row
        x = col * slide_step  
        y = row * slide_step  
        rect = patches.Rectangle(
            (x, y), patch_size, patch_size,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    # 7) 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close(fig)
def coord_to_patch_index(x, y, img_w, img_h, config):
    """
    计算给定像素坐标 (x,y) 对应的 patch 索引。

    参数:
        x, y: 像素坐标（整数）
        img_w, img_h: 图片宽度和高度（像素）
        config: 配置对象，含有patch_size和slide_step等属性（比如 CONFIGS['ViT-B_16']）

    返回:
        patch索引 (int)
    """
    patch_size = config.patches.size[0]  # 比如16
    slide_step = getattr(config, 'slide_step', patch_size)  # 如果没有slide_step，默认不重叠，步长=patch_size

    patches_per_row = (img_w - patch_size) // slide_step + 1
    patches_per_col = (img_h - patch_size) // slide_step + 1

    # 计算像素点在哪个patch的行列
    col = x // slide_step
    row = y // slide_step

    # 防止越界
    col = min(max(col, 0), patches_per_row - 1)
    row = min(max(row, 0), patches_per_col - 1)

    patch_index = row * patches_per_row + col
    return patch_index

def main(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    orig_image, img_tensor = preprocess_image(image_path)
    img_w, img_h = orig_image.size
    img_tensor = img_tensor.to(device)
    config = CONFIGS["ViT-B_16"]

    # 假设你有一组像素坐标：
    coords = [(172, 45.3), (5.86, 3.77), (6.70, 3.07)]  # 这里改成你的坐标列表

    # 针对每个坐标计算patch索引
    patch_indices = []
    for (x, y) in coords:
        idx = coord_to_patch_index(x, y, img_w, img_h, config)
        patch_indices.append(idx)

    print("坐标对应的patch索引:", patch_indices)

if __name__ == "__main__":
    image_path = "/data/datasets/zhuhaoran/image/tomato_leaf_mold/0bae0514-799a-479f-a407-ad763f458916___Crnl_L.Mold 8989.JPG"
    main( image_path)
