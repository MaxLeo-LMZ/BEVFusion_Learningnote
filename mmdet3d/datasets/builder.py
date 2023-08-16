import platform
from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset
# 如果不是Windows操作系统，调整文件句柄数量的软限制
if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    # 导入resource模块，用于管理系统资源
    import resource

    # 获取系统文件句柄的软限制和硬限制（RLIMIT_NOFILE）
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    # 设置文件句柄数量的软限制为计算得到的soft_limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))
# 创建一个名为OBJECTSAMPLERS的注册表，用于存储对象采样器
OBJECTSAMPLERS = Registry("Object sampler")
# 定义一个函数build_dataset，用于构建数据集对象
def build_dataset(cfg, default_args=None):
    # 导入所需的数据集包装类
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import ClassBalancedDataset, ConcatDataset, RepeatDataset
    # 如果cfg是列表或元组类型，递归构建多个数据集并拼接为ConcatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    # 如果cfg的type为"ConcatDataset"，构建ConcatDataset
    elif cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg["datasets"]],
            cfg.get("separate_eval", True),
        )
    # 如果cfg的type为"RepeatDataset"，构建RepeatDataset
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(build_dataset(cfg["dataset"], default_args), cfg["times"])
    # 如果cfg的type为"ClassBalancedDataset"，构建ClassBalancedDataset
    elif cfg["type"] == "ClassBalancedDataset":
        dataset = ClassBalancedDataset(
            build_dataset(cfg["dataset"], default_args), cfg["oversample_thr"]
        )
    # 如果cfg的type为"CBGSDataset"，构建CBGSDataset
    elif cfg["type"] == "CBGSDataset":
        dataset = CBGSDataset(build_dataset(cfg["dataset"], default_args))
    # 如果ann_file是列表或元组类型，调用_concat_dataset将多个数据集合并
    elif isinstance(cfg.get("ann_file"), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    # 否则，根据配置信息从DATASETS注册表中构建数据集对象
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    # 返回构建好的数据集对象
    return dataset
