import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

nus_categories = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

nus_attributes = (
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
    "None",
)

# 这个函数主要完成了以下任务：根据数据版本和预定义的训练/验证场景，生成训练和验证样本的信息数据，
# 并将这些信息数据保存到 '.pkl' 文件中，以便后续在训练和评估中使用。
def create_nuscenes_infos(
    root_path, info_prefix, version="v1.0-trainval", max_sweeps=10
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    # 定义了支持的nuScenes数据集版本
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    # 这部分代码根据指定的版本，确定了在训练集和验证集中要处理的场景列表。
    # 这些列表来自于 nuscenes.utils.splits 中的预定义场景列表。
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    # filter existing scenes.
    # 这部分代码使用 get_available_scenes 函数从nuScenes数据集中获取可用的场景信息。
    # 然后，根据预定义的训练和验证场景，过滤出现有数据中真正存在的场景，这是为了确保只处理存在的场景。
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    # 这段代码的目的是将训练场景列表和验证场景列表中的场景名称转换为对应的token，
    # 并将这些token存储在集合中，以便后续在数据处理中使用。这样做可以更高效地在数据集中查找和筛选出具体的场景。
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )

    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print(
            "train scene: {}, val scene: {}".format(len(train_scenes), len(val_scenes))
        )
    # 这一部分调用 _fill_trainval_infos 函数，
    # 该函数接收数据集、训练场景、验证场景、是否为测试集和最大连续帧数等参数，生成并填充训练样本和验证样本的信息。
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps
    )
    # 这一部分根据是否是测试集，生成相应的元数据和信息数据，然后使用 mmcv.dump 将这些数据保存为 '.pkl' 文件。
    metadata = dict(version=version)
    if test:
        print("test sample: {}".format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, "{}_infos_test.pkl".format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print(
            "train sample: {}, val sample: {}".format(
                len(train_nusc_infos), len(val_nusc_infos)
            )
        )
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, "{}_infos_train.pkl".format(info_prefix))
        mmcv.dump(data, info_path)
        data["infos"] = val_nusc_infos
        info_val_path = osp.join(root_path, "{}_infos_val.pkl".format(info_prefix))
        mmcv.dump(data, info_val_path)

# 这个函数 get_available_scenes(nusc) 的作用是从nuScenes数据集中获取可用的场景信息。
# 它会遍历所有场景，并检查每个场景是否存在有效的LIDAR_TOP数据（激光雷达扫描数据），然后将符合条件的场景信息收集起来。
def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    # 初始化变量
    available_scenes = []
    print("total scene num: {}".format(len(nusc.scene)))
    # 遍历所有场景
    for scene in nusc.scene:
        # 对于每个场景，首先获取场景的token，然后使用token获取场景、样本和样本数据的记录（record）。
        # 其中，LIDAR_TOP 是激光雷达的顶部扫描数据。
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        # 检查数据路径
        # 这一部分代码检查LIDAR_TOP数据的路径。它循环获取样本数据中的LIDAR_TOP数据路径，然后判断路径是否有效。
        # 如果路径无效，将 scene_not_exist 设置为 True，表示该场景不存在。
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        # 如果场景存在并且LIDAR_TOP数据路径有效，则将该场景信息添加到 available_scenes 列表中。
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    # 函数返回收集到的所有可用场景的信息列表
    return available_scenes


def _fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    # 初始化信息列表
    train_nusc_infos = []
    val_nusc_infos = []
    # 遍历样本数据
    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        location = nusc.get(
            "log", nusc.get("scene", sample["scene_token"])["log_token"]
        )["location"]
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        # 检查文件是否存在
        mmcv.check_file_exist(lidar_path)
        # 生成基本信息
        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
            "location": location,
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        # 针对每个图像相机，获取相机的token，
        # 并使用 obtain_sensor2top 函数获取相机的转换信息和内参，然后将这些信息存储在 info 字典的 cams 字段中。
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for cam in camera_types:
            cam_token = sample["data"][cam]
            cam_path, _, camera_intrinsics = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
            )
            cam_info.update(camera_intrinsics=camera_intrinsics)
            info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        # 获取连续帧的扫描信息，通过循环获取前一帧扫描数据的token，
        # 并使用 obtain_sensor2top 函数获取其转换信息，然后将这些连续帧的信息存储在 info 字典的 sweeps 字段中。
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(
                    nusc, sd_rec["prev"], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar"
                )
                sweeps.append(sweep)
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps
        # obtain annotation
        # 如果不是在测试模式下，这部分代码获取样本的注释信息，包括物体的位置、尺寸、朝向、速度等。
        if not test:
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
                -1, 1
            )
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample["anns"]]
            )
            valid_flag = np.array(
                [
                    (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                    for anno in annotations
                ],
                dtype=bool,
            ).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                # 速度转换
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]
            # 名称印射
            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            # 物体边界框信息的处理
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations
            ), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
            info["valid_flag"] = valid_flag
        # 根据样本所属的场景判断是训练集还是验证集，并将信息添加到相应的列表中
        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)
    # 函数返回生成的训练集和验证集的信息列表。
    return train_nusc_infos, val_nusc_infos

# 函数的目的是将特定传感器坐标系下的数据转换到激光雷达坐标系下，以便进行数据的统一处理。
# 函数最终返回了经过坐标系变换后的信息，该信息包括传感器数据路径、类型、令牌、平移和旋转信息等。
# 这些信息将用于数据处理和训练过程
def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.


    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

        nusc: nuScenes数据集的类实例。
        sensor_token: 指定传感器类型的样本数据令牌。
        l2e_t、l2e_r_mat: 从激光雷达到车辆坐标系的平移和旋转矩阵。
        e2g_t、e2g_r_mat: 从车辆坐标系到全局坐标系的平移和旋转矩阵。
        sensor_type: 传感器类型，默认为 'lidar'
    Returns:
        sweep (dict): Sweep information after transformation.
    """
    # 获取传感器数据、传感器校准信息和车辆位姿信息等。
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    # 将传感器数据的路径转换为相对路径
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    # 构建转换后的信息
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    # 计算从传感器坐标系到激光雷达坐标系的旋转矩阵 R 和平移向量 T。最后，将变换信息存储在 sweep 字典中。
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.

        root_path: 原始数据的根路径。
        info_path: 信息文件的路径。
        version: 数据集版本。
        mono3d: 是否导出mono3d标注，默认为 True。
    """
    # get bbox annotations for camera
    # 获取相机类型列表
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    # 加载信息文件和NuScenes数据集类实例
    nusc_infos = mmcv.load(info_path)["infos"]
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    # 定义COCO格式中的类别映射
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    # 初始化COCO格式的标注信息
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    # 遍历每个信息并导出2D标注
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info["cams"][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info["sample_data_token"],
                visibilities=["", "1", "2", "3", "4"],
                mono3d=mono3d,
            )
            (height, width, _) = mmcv.imread(cam_info["data_path"]).shape
            coco_2d_dict["images"].append(
                dict(
                    file_name=cam_info["data_path"].split("data/nuscenes/")[-1],
                    id=cam_info["sample_data_token"],
                    token=info["token"],
                    cam2ego_rotation=cam_info["sensor2ego_rotation"],
                    cam2ego_translation=cam_info["sensor2ego_translation"],
                    ego2global_rotation=info["ego2global_rotation"],
                    ego2global_translation=info["ego2global_translation"],
                    camera_intrinsics=cam_info["camera_intrinsics"],
                    width=width,
                    height=height,
                )
            )
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info["segmentation"] = []
                coco_info["id"] = coco_ann_id
                coco_2d_dict["annotations"].append(coco_info)
                coco_ann_id += 1
    # 根据 mono3d 参数设置导出文件的前缀
    if mono3d:
        json_prefix = f"{info_path[:-4]}_mono3d"
    else:
        json_prefix = f"{info_path[:-4]}"
    # 将COCO格式的标注信息保存到JSON文件中
    mmcv.dump(coco_2d_dict, f"{json_prefix}.coco.json")

# 该函数的目的是从给定的 sample_data_token 获取满足条件的标注记录，并生成对应的 2D 投影坐标。
def get_2d_boxes(nusc, sample_data_token: str, visibilities: List[str], mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

        nusc: NuScenes 数据集类实例。
        sample_data_token: 一个属于相机关键帧的样本数据标识。
        visibilities: 可见性筛选列表。
        mono3d: 是否获取具有 mono3d 标注的框，默认为 True。
    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    # 获取 sample_data 和与之相关的样本信息 sample
    sd_rec = nusc.get("sample_data", sample_data_token)

    assert sd_rec["sensor_modality"] == "camera", (
        "Error: get_2d_boxes only works" " for camera sample_data!"
    )
    if not sd_rec["is_key_frame"]:
        raise ValueError("The 2D re-projections are available only for keyframes.")
    # 获取 sample_data 和与之相关的样本信息 sample
    s_rec = nusc.get("sample", sd_rec["sample_token"])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    # 获取校准的传感器和车辆位置信息，用于获取转换矩阵
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    camera_intrinsic = np.array(cs_rec["camera_intrinsic"])

    # Get all the annotation with the specified visibilties.
    # 获取满足可见性条件的标注记录
    ann_recs = [nusc.get("sample_annotation", token) for token in s_rec["anns"]]
    ann_recs = [
        ann_rec for ann_rec in ann_recs if (ann_rec["visibility_token"] in visibilities)
    ]

    repro_recs = []
    # 对满足条件的标注记录进行处理，生成 2D 投影坐标
    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec["sample_annotation_token"] = ann_rec["token"]
        ann_rec["sample_data_token"] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec["token"])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec["translation"]))
        box.rotate(Quaternion(pose_rec["rotation"]).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec["translation"]))
        box.rotate(Quaternion(cs_rec["rotation"]).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = (
            view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
        )

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(
            ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec["filename"]
        )

        # If mono3d=True, add 3D annotations in camera coordinates
        #　若 mono3d 为 True，添加 3D 注释信息
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec["rotation"]).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec["rotation"]).rotation_matrix
            cam_velo3d = (
                global_velo3d @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            )
            velo = cam_velo3d[0::2].tolist()

            repro_rec["bbox_cam3d"] = loc + dim + rot
            repro_rec["velo_cam3d"] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(center3d, camera_intrinsic, with_depth=True)
            repro_rec["center2d"] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec["center2d"][2] <= 0:
                continue

            ann_token = nusc.get("sample_annotation", box.token)["attribute_tokens"]
            if len(ann_token) == 0:
                attr_name = "None"
            else:
                attr_name = nusc.get("attribute", ann_token[0])["name"]
            attr_id = nus_attributes.index(attr_name)
            repro_rec["attribute_name"] = attr_name
            repro_rec["attribute_id"] = attr_id

        repro_recs.append(repro_rec)
    # 返回最终的处理结果
    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

        corner_coords：投影后的边界框角点坐标列表。
        imsize：图像画布的大小，作为一个元组 (width, height)

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    # 通过投影后的边界框角点坐标列表生成多边形的凸包
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    # 创建图像画布的边界框
    img_canvas = box(0, 0, imsize[0], imsize[1])
    # 检查凸包是否与图像画布相交
    if polygon_from_2d_box.intersects(img_canvas):
        # 计算交集的矩形边界框坐标
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords]
        )

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])
        # 返回交集的矩形边界框坐标或 None
        return min_x, min_y, max_x, max_y
    else:
        return None

# 根据给定的2D边界框坐标和其他信息，生成一个符合COCO格式的标注记录，并将其用于导出2D标注
def generate_record(
    ann_rec: dict,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    sample_data_token: str,
    filename: str,
) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    # 创建一个有序字典 repro_rec 用于存储生成的2D标注记录
    repro_rec = OrderedDict()
    repro_rec["sample_data_token"] = sample_data_token
    # 创建一个字典 coco_rec 用于存储COCO格式的标注记录。
    coco_rec = dict()
    # 从原始的3D注释记录中提取与标注相关的键的值，并将其存储到 repro_rec 中。
    relevant_keys = [
        "attribute_tokens",
        "category_name",
        "instance_token",
        "next",
        "num_lidar_pts",
        "num_radar_pts",
        "prev",
        "sample_annotation_token",
        "sample_data_token",
        "visibility_token",
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value
    # 存储边界框的角点坐标、图像文件名等信息到 repro_rec 中
    repro_rec["bbox_corners"] = [x1, y1, x2, y2]
    repro_rec["filename"] = filename
    # 基于2D边界框的坐标计算COCO格式的记录中的相关信息
    coco_rec["file_name"] = filename
    coco_rec["image_id"] = sample_data_token
    coco_rec["area"] = (y2 - y1) * (x2 - x1)
    # 如果标注的类别名不在 NuScenesDataset.NameMapping 中，则返回 None
    if repro_rec["category_name"] not in NuScenesDataset.NameMapping:
        return None
    # 如果存在类别名映射，将类别名映射到 nus_categories 中的索引，并存储到 coco_rec 中
    cat_name = NuScenesDataset.NameMapping[repro_rec["category_name"]]
    coco_rec["category_name"] = cat_name
    coco_rec["category_id"] = nus_categories.index(cat_name)
    coco_rec["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec["iscrowd"] = 0

    return coco_rec
