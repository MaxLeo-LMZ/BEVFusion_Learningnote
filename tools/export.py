import argparse
import os
import time
import warnings

import mmcv
import onnx
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from onnxsim import simplify
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    # 启用CuDNN的自动优化，以加速训练过程
    torch.backends.cudnn.benchmark = True
    # 不使用预训练权重
    cfg.model.pretrained = None
    # 测试模式下加载数据集
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # data_loader: 使用build_dataloader()构建的数据加载器，其中包含以下参数：
        # dataset: 要加载的数据集。
        # samples_per_gpu: 每个GPU上的样本数。
        # workers_per_gpu: 每个GPU上的数据加载器工作线程数。
        # dist: 是否使用分布式训练。
        # shuffle: 是否在每个epoch中打乱数据。
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility 旧版本没有在检查点中保存类信息，此遍历是为了向后兼容
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    # 设置模型为评估模式
    model.eval()

    with torch.no_grad():
        for data in data_loader:
            img = [torch.cat([data["img"][0].data[0]] * 6, dim=0)]
            metas = data["metas"][0].data

            from functools import partial

            model.forward = partial(
                model.forward_test,
                metas=metas,
                rescale=True,
            )

            torch.onnx.export(
                model,
                img,
                "model.onnx",
                input_names=["input"],
                opset_version=13,
                do_constant_folding=True,
            )
            model = onnx.load("model.onnx")
            model, _ = simplify(model)
            onnx.save(model, "model.onnx")
            return


if __name__ == "__main__":
    main()
