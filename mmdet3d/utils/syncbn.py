import copy
import torch
from collections import deque


__all__ = ["convert_sync_batchnorm"]


def convert_sync_batchnorm(input_model, exclude=[]):
    """
     将模型中的普通BatchNorm层转换为SyncBatchNorm层（适用于分布式训练）

     参数：
     input_model (nn.Module)：输入的模型
     exclude (list)：排除不需要转换的层的名称列表

     返回：
     input_model (nn.Module)：转换后的模型
     """
    # 遍历模型的每个子模块
    for name, module in input_model._modules.items():
        # 检查是否需要跳过当前子模块
        skip = sum([ex in name for ex in exclude])
        if skip:
            continue
        # 转换当前子模块中的普通BatchNorm层为SyncBatchNorm层
        input_model._modules[name] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
    # 返回转换后的模型
    return input_model
    