import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
# 以上导入了许多第三方模块，包括参数解析、文件操作、随机数生成、时间操作、NumPy、PyTorch、MMDetection3D的相关模块等
# 定义主函数
def main():
    # dist.init()
    torch.cuda.empty_cache()
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()
    # 加载配置文件
    # 使用mmcv.Config加载配置文件，并通过recursive=True参数递归地加载嵌套的配置。
    # 然后使用opts更新配置，opts存储的是命令行中未被解析的参数。
    configs.load(args.config, recursive=True)
    configs.update(opts)
    # 创建配置对象
    # 创建一个配置对象cfg，其中配置信息是通过recursive_eval对嵌套配置进行了求值得到的，filename指定了配置文件的路径。
    cfg = Config(recursive_eval(configs), filename=args.config)
    # 设置GPU加速相关参数
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(torch.cuda.current_device())
    # 设置运行目录
    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config 将配置保存到文件
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model,)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()

