import mmcv
import torch


def single_gpu_test(model, data_loader):
    model.eval() # 设置模型为评估模式
    results = [] # 存储测试结果的列表
    dataset = data_loader.dataset # 获取数据集对象
    prog_bar = mmcv.ProgressBar(len(dataset))  # 创建一个进度条，用于显示测试进度
    for data in data_loader:  # 遍历数据加载器中的每个数据
        with torch.no_grad(): # 在这个上下文中，禁用梯度计算，因为这是测试阶段
            result = model(return_loss=False, rescale=True, **data) # 使用模型进行前向推理，返回预测结果
        results.extend(result)  # 将预测结果添加到结果列表中

        batch_size = len(result) # 获取批次大小（预测结果数量）
        for _ in range(batch_size): # 更新进度条，表示处理了一个批次中的样本
            prog_bar.update()
    return results # 返回所有测试结果的列表
