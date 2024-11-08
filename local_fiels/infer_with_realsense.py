import os

import shutil

os.chdir("../")

import cv2
from tqdm import tqdm
import numpy as np

from estimater import *
from datareader import *


def infer_with_realsense():
    # root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855"
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009"
    root = "/home/pxn-lyj/Egolee/data/test/pose_shi"
    img_root = os.path.join(root, "colors")
    depth_root = os.path.join(root, "depths")

    vis_img_root = os.path.join(root, "colors_vis")
    pose_root = os.path.join(root, "poses")
    # for _path in [vis_img_root, pose_root]:
    #     if os.path.exists(_path):
    #         shutil.rmtree(_path)
    #     os.mkdir(_path)

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", ".png"]]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    cam_k_path = os.path.join(root, "cam_k.txt")
    cam_k = np.loadtxt(cam_k_path)

    # mask_root = os.path.join(root, "masks")
    mask_root = os.path.join(root, "masks_num")

    init_pose_model = False

    # mesh_file = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample/untitled.obj"
    # mesh_file = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample/untitled.obj"
    # mesh_file = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490/taizi1kgxiyiye.obj"
    mesh_file = "/home/pxn-lyj/Egolee/data/test/pose_shi/obj_000016_1.obj"

    # blender到处的时候需要选择triangulate faces
    mesh = trimesh.load(mesh_file, force='mesh')  # 从blender导出的模型需要设置force='mesh', 要不然没有vertices和vertex_normals这些属性
    # mesh.apply_scale(0.001)   # mesh模型的单位是毫米，需要将其设置为米的单位 深度图输入的单位也是米

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    debug_dir = os.path.join(root, "debug")

    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                         refiner=refiner, debug_dir=debug_dir, debug=0, glctx=glctx)
    iteration = 4
    for img_name in img_names:
        num = int(img_name.split(".")[0].split("_")[0])
        # if num > 319:
        #     continue

        img_path = os.path.join(img_root, img_name)
        depth_path = os.path.join(depth_root, img_name.replace("_color.jpg", "_depth.npy"))

        color = cv2.imread(img_path)
        depth = np.load(depth_path)

        color = color[:, :, ::-1].copy()
        if not init_pose_model or 1:   # 全程采用mask
        # if not init_pose_model:
            mask_path = os.path.join(mask_root, img_name.replace("_color.jpg", "_mask.npy"))
            mask = np.load(mask_path)
            pose = est.register(K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=iteration)
            init_pose_model = True
        else:
            pose = est.track_one(rgb=color, depth=depth, K=cam_k, iteration=iteration)

        # center_pose = pose@np.linalg.inv(to_origin)
        center_pose = pose

        if ~np.all(pose == np.eye(4)):
            vis = draw_posed_3d_box(cam_k, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_k, thickness=3, transparency=0, is_input_rgb=True)
        else:
            vis = color

        cv2.imshow('1', vis[...,::-1])

        cv2.imwrite(os.path.join(vis_img_root, img_name), vis[:, :, ::-1])
        np.save(os.path.join(pose_root, img_name.replace("_color.jpg", "_pose.npy")), center_pose)

        # cv2.imshow('1', vis)
        # cv2.waitKey(1)


if __name__ == "__main__":
    print("Start")
    infer_with_realsense()
    print("End")
