# 这段代码的主要任务是根据命令行参数准备nuScenes数据集相关的数据。
# 具体操作包括创建基本信息、2D标注以及地面真值数据库等文件，以便后续在训练过程中使用。
import argparse

from data_converter import nuscenes_converter as nuscenes_converter
from data_converter.create_gt_database import create_groundtruth_database

# 这个函数用于准备与nuScenes数据集相关的数据。
#   root_path: 数据集根目录。
#   info_prefix: info文件名前缀。
#   version: 数据集版本。
#   dataset_name: 数据集名称。
#   out_dir: 输出地面真值数据库信息的目录。
#   max_sweeps: 每个样本的连续帧数，默认为10。
#   load_augmented: 是否加载增强数据，通常是一些虚拟的传感器数据（可选）。
def nuscenes_data_prep(
    root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    load_augmented=None,
):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    if load_augmented is None:
        # otherwise, infos must have been created, we just skip.

        # 给定原始数据，以pkl格式生成其相关信息文件。
        # 参数:
        #   root_path (str):数据根路径。
        #   info_prefix (str):要生成的info文件前缀。
        #   version (str):数据的版本。 默认值:“v1.0-trainval”
        # max_scans (int):最大扫描次数。 默认值:10
        nuscenes_converter.create_nuscenes_infos(
            root_path, info_prefix, version=version, max_sweeps=max_sweeps
        )

        # if version == "v1.0-test":
        #     info_test_path = osp.join(root_path, f"{info_prefix}_infos_test.pkl")
        #     nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
        #     return

        # info_train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
        # info_val_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
        # nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        # nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)

    # 给定原始数据，生成地面真相数据库。
    # 参数:
    #   dataset_class_name (str):输入数据集的名称。
    #   data_path (str):数据的路径。
    #   info_prefix (str): info文件的前缀。
    #   info_path (str): info文件路径。 默认值:没有。
    #   mask_anno_path (str): mask_annono的路径。 默认值:没有。
    #   used_classes (list[str]):类已被使用。 默认值:没有。
    #   database_save_path (str):保存数据库的路径。 默认值:没有。
    #   db_info_save_path (str): db_info的保存路径。 默认值:没有。
    #   relative_path (bool):是否使用相对路径。 默认值:真的。
    #   with_mask (bool):是否使用掩码。 默认值:False。
    create_groundtruth_database(
        dataset_name,
        root_path,
        info_prefix,
        f"{out_dir}/{info_prefix}_infos_train.pkl",
        load_augmented=load_augmented,
    )


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kitti", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="./data/kitti",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--version",
    type=str,
    default="v1.0",
    required=False,
    help="specify the dataset version, no need for kitti",
)
parser.add_argument(
    "--max-sweeps",
    type=int,
    default=10,
    required=False,
    help="specify sweeps of lidar per example",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="./data/kitti",
    required=False,
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="kitti")
parser.add_argument("--painted", default=False, action="store_true")
parser.add_argument("--virtual", default=False, action="store_true")
parser.add_argument(
    "--workers", type=int, default=4, help="number of threads to be used"
)
args = parser.parse_args()

if __name__ == "__main__":
    load_augmented = None
    if args.virtual:
        if args.painted:
            load_augmented = "mvp"
        else:
            load_augmented = "pointpainting"

    if args.dataset == "nuscenes" and args.version != "v1.0-mini":
        train_version = f"{args.version}-trainval"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
        test_version = f"{args.version}-test"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
    elif args.dataset == "nuscenes" and args.version == "v1.0-mini":
        train_version = f"{args.version}"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
