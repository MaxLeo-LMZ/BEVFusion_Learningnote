# 这段代码完成了训练模型的主要逻辑，包括数据加载、模型构建、优化器设置、训练循环和评估等步骤，使得整个训练过程得以顺利进行。
import torch
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    GradientCumulativeFp16OptimizerHook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmdet3d.runner import CustomEpochBasedRunner

from mmdet3d.utils import get_root_logger
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor

# 定义 train_model 函数
# 这是一个主要用于模型训练的函数，它接收以下参数：
#   model: 要训练的模型。
#   dataset: 训练使用的数据集。
#   cfg: 配置对象。
#   distributed: 是否使用分布式训练。
#   validate: 是否进行验证。
#   timestamp: 时间戳。
def train_model(
        model,
        dataset,
        cfg,
        distributed=False,
        validate=False,
        timestamp=None,
):
    logger = get_root_logger()

    # prepare data loaders 准备数据加载器
    # 这部分代码将数据集转换为列表形式，然后为每个数据集创建一个数据加载器（使用build_dataloader函数），
    # 设置每个GPU上的样本数和工作进程数，并根据是否分布式训练来设置随机种子。
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]

    # put model on gpus
    find_unused_parameters = cfg.get("find_unused_parameters", False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    # 如果使用分布式训练，将模型放置在MMDistributedDataParallel中；
    # 否则，放置在MMDataParallel中。这将模型移到GPU上，并根据需要配置分布式训练的参数。
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(
            model.cuda(),
            device_ids=[0],
        )
    # build runner
    # 使用build_optimizer函数构建优化器，然后使用build_runner函数构建Runner。
    # Runner的配置信息包括模型、优化器、工作目录、日志记录器等。
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.run_dir,
            logger=logger,
            meta={},
        ),
    )
    # 如果Runner具有set_dataset方法，将数据集设置给Runner。
    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    # 这部分代码根据配置文件中是否启用了fp16训练（fp16_cfg），来选择不同的优化器设置。
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if "cumulative_iters" in cfg.optimizer_config:
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    # 这里注册了一系列训练时所需要的配置，包括学习率调整、优化器配置、保存检查点、日志记录等。
    # 还可以根据配置中是否定义了自定义的hooks进行注册。
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
        custom_hooks_config=cfg.get('custom_hooks', None)
    )
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    # 如果设置了需要验证，这部分代码会注册一个评估hook，用于在训练过程中定期进行模型评估。
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook if distributed else EvalHook
        ###主要是这一步
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
    # 如果有恢复训练的路径，恢复训练。如果有加载检查点的路径，加载检查点。
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    # 使用Runner的run方法开始训练，传递数据加载器和训练周期信息。
    runner.run(data_loaders, [("train", 1)])

