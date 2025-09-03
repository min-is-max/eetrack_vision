import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import open3d as o3d
import torch
from tqdm import tqdm
import shutil
from scipy.interpolate import splprep, splev


def point_cloud_from_rgbd(
    color, depth, intrinsic, extrinsic=np.eye(4),
    depth_scale=1000.0, depth_trunc=np.inf,
):
    h, w = color.shape[:2]
    color_o3d = o3d.geometry.Image(color)
    depth_o3d = o3d.geometry.Image(depth)

    # Camera intrinsics (adjust to your sensor)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=w,
        height=h,
        intrinsic_matrix=intrinsic,
    )
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=depth_scale,  # if depth is in mm
        depth_trunc=depth_trunc,      # ignore depth > 5m
    )

    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=rgbd,
        intrinsic=intrinsics,
        extrinsic=extrinsic,
    )

    return pcd


def point_cloud_from_points(points, colors=None):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if colors is not None:
        if not isinstance(colors, np.ndarray):
            colors = np.asarray(colors)
        colors = colors.reshape(-1, 3)
        if len(colors) == 1:
            colors = colors.repeat(len(points), 0)

        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def ensure_non_circular_ids(ids):
    diff = np.diff(ids)
    if (diff < 2).all():
        return ids

    end_id = diff.argmax() + 1
    ids = np.concatenate([ids[end_id:], ids[:end_id]])

    return ids


def bspline_points_down_sample(points: np.ndarray, target_num_pts: int, **kwargs):
    if points.ndim == 2:
        tck, _ = splprep(points.T, **kwargs)
        u = np.linspace(0, 1, target_num_pts)
        down_sampled_pts = np.array(splev(u, tck)).T
        return down_sampled_pts
    elif points.ndim == 3:
        B = len(points)
        down_sampled_pts_batch = []
        for i in range(B):
            down_sampled_pts = bspline_points_down_sample(points[i], target_num_pts, **kwargs)
            down_sampled_pts_batch.append(down_sampled_pts)
        down_sampled_pts_batch = np.array(down_sampled_pts_batch)

        return down_sampled_pts_batch


def fit_spline(points: np.ndarray, **kwargs):
    if points.ndim == 2:
        tck, _ = splprep(points.T, **kwargs)
        t = tck[0]
        c = np.array(tck[1])
        k = [tck[2]]
        return t, c, k
    elif points.ndim == 3:
        B = len(points)
        ts = []
        cs = []
        ks = []
        for i in range(B):
            t, c, k = fit_spline(points[i], **kwargs)
            ts.append(t)
            cs.append(c)
            ks.append(k)
        ts = np.array(ts)
        cs = np.array(cs)
        ks = np.array(ks)
        return ts, cs, ks


def generate_train_val_test_ids(num_data, train_ratio, val_ratio, test_ratio, seed=42):
    np.random.seed(seed)
    rand_ids = np.random.permutation(num_data)

    total = train_ratio + val_ratio + test_ratio
    n_train = int(num_data * train_ratio / total)
    n_val = int(num_data * val_ratio / total)

    train_ids, val_ids, test_ids = np.split(rand_ids, [n_train, n_train+n_val])

    return train_ids, val_ids, test_ids


def main(data_dir, plate_name, pillar_name, train_val_test_ratio, save_dir):
    data_dir = Path(data_dir)

    left_color_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("left/color"))], start=[])
    right_color_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("right/color"))], start=[])
    left_depth_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("left/depth"))], start=[])
    right_depth_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("right/depth"))], start=[])
    left_normal_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("left/normal"))], start=[])
    right_normal_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("right/normal"))], start=[])
    intrinsic_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("left/intrinsic"))], start=[])
    pose_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("left/pose"))], start=[])
    edge_cam_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("left/edge_cam"))], start=[])
    edge_cam_ds_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("left/edge_cam_ds"))], start=[])
    edge_vis_mask_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("left/edge_vis_mask"))], start=[])
    spline_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("left/spline"))], start=[])

    plate_mask_dirs = sum([list(sorted(dirs.glob(plate_name + "*"))) for dirs in sorted(data_dir.rglob("left/mask_gt"))], start=[])
    pillar_mask_dirs = sum([list(sorted(dirs.glob(pillar_name + "*"))) for dirs in sorted(data_dir.rglob("left/mask_gt"))], start=[])
    # Generate masks if masks do not exist.
    if len(plate_mask_dirs) == 0 or len(pillar_mask_dirs) == 0:
        # Add Grounded-SAM-2 path manually
        gsa2_path = os.path.join(os.path.dirname(__file__), "../Grounded-SAM-2")
        sys.path.append(gsa2_path)
        from gsa2_video_tracker import GSA2VideoTracker

        print(f"Masks for given plate name '{plate_name}' and pillar name '{pillar_name}' are not found.")
        if not input("Want to run segmentation? [y/n]").lower() == "y":
            return

        plate_name = input("Plate name to use:")
        pillar_name = input("Pillar name to use:")

        text_prompt = f"{plate_name}. {pillar_name}."
        print(f"Using text prompt: '{text_prompt}'")

        video_tracker = GSA2VideoTracker("cuda")

        _, _ = video_tracker.predict_video(
            video_dir = (data_dir / "left" / "color").as_posix(),
            text = text_prompt,
            mask_dir = (data_dir / "left" / "mask").as_posix(),
            result_dir = (data_dir / "left" / "result").as_posix(),
        )

        del video_tracker

        plate_mask_dirs = list((data_dir / "left" / "mask").glob(plate_name.replace(' ', '_') + "*"))
        pillar_mask_dirs = list((data_dir / "left" / "mask").glob(pillar_name.replace(' ', '_') + "*"))

    # FIXME: assume the plate mask is only one.
    plate_mask_paths = sum([list(sorted(dirs.glob("*"))) for dirs in sorted(data_dir.rglob(f"left/mask_gt/{plate_name}"))], start=[])
    pillar_mask_paths = sum([list(sorted(dirs.glob("*"))) for dirs in sorted(data_dir.rglob(f"left/mask_gt/{pillar_name}"))], start=[])


    # Generate depth estimation if they do not exist.
    left_est_depth_dirs = list(sorted(data_dir.rglob("left/est_depth")))
    if len(left_est_depth_dirs) != len(list(data_dir.rglob("left/color"))):
        print("Running depth estimation using FoudationStereo.")
        gsa2_path = os.path.join(os.path.dirname(__file__), "../FoundationStereo")
        sys.path.append(gsa2_path)

        from depth_estimator import DepthEstimator
        torch.autograd.set_grad_enabled(False)

        depth_estimator = DepthEstimator()

        traj_dirs = [lcdirs.parents[1] for lcdirs in sorted(data_dir.rglob("left/color"))]
        for traj_dir in tqdm(traj_dirs):
            est_depth_save_dir = traj_dir / "left" / "est_depth"
            if est_depth_save_dir in left_est_depth_dirs:
                continue
            est_depth_save_dir.mkdir(parents=True, exist_ok=True)

            traj_left_color_paths = sorted((traj_dir / "left" / "color").glob('*'))
            traj_right_color_paths = sorted((traj_dir / "right" / "color").glob('*'))
            traj_intrinsic_paths = sorted((traj_dir / "left" / "intrinsic").glob('*'))

            for i, (lcpath, rcpath, Kpath) in enumerate(zip(traj_left_color_paths, traj_right_color_paths, traj_intrinsic_paths)):
                left_image = imageio.imread(lcpath)
                right_image = imageio.imread(rcpath)
                K = np.loadtxt(Kpath)

                est_depth = depth_estimator.predict(
                    left_image,
                    right_image,
                    K,
                    baseline=0.063,
                )

                est_depth_save_path = os.path.join(est_depth_save_dir, f"{i:06d}.png")
                imageio.imwrite(est_depth_save_path, (1000*est_depth).astype(np.uint16))

        del depth_estimator

    left_est_depth_paths = sum([list(sorted(dirs.glob('*'))) for dirs in sorted(data_dir.rglob("left/est_depth"))], start=[])

    num_data = len(left_color_paths)
    train_ratio, val_ratio, test_ratio = map(int, train_val_test_ratio.split(':'))
    train_ids, val_ids, test_ids = generate_train_val_test_ids(
        num_data,
        train_ratio,
        val_ratio,
        test_ratio,
    )

    save_dir = Path(os.path.dirname(__file__), save_dir)
    # Remove existing data
    shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    all_dir = save_dir / "all"
    train_dir = save_dir / "train"
    val_dir = save_dir / "val"
    test_dir = save_dir / "test"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    for i, (lcpath, rcpath, ldpath, rdpath, lnpath, rnpath, edpath, Kpath, Tpath,
            ecpath, ecdpath, evmpath, spath, plpath, pipath) in tqdm(enumerate(zip(
        left_color_paths,
        right_color_paths,
        left_depth_paths,
        right_depth_paths,
        left_normal_paths,
        right_normal_paths,
        left_est_depth_paths,
        intrinsic_paths,
        pose_paths,
        edge_cam_paths,
        edge_cam_ds_paths,
        edge_vis_mask_paths,
        spline_paths,
        plate_mask_paths,
        pillar_mask_paths,
    )), total=num_data):
        left_color = np.asarray(Image.open(lcpath))
        right_color = np.asarray(Image.open(rcpath))
        left_depth = imageio.imread(ldpath)
        right_depth = imageio.imread(rdpath)
        left_normal = np.load(lnpath)
        right_normal = np.load(rnpath)
        est_depth = imageio.imread(edpath)
        intrinsic = np.loadtxt(Kpath)
        # NOTE: extrinsic = T_wc = world pose in camera frame.
        extrinsic = np.linalg.inv(np.loadtxt(Tpath)) # T_cw -> T_wc
        edge_cam = np.load(ecpath)
        edge_cam_ds = np.load(ecdpath)
        edge_vis_mask = np.load(evmpath)
        spline = dict(np.load(spath))
        plate_mask = imageio.imread(plpath) > 0
        pillar_mask = imageio.imread(pipath) > 0

        H, W = left_color.shape[:2]

        if edge_vis_mask.sum() < 5:
            continue

        valid = est_depth > 0
        plate_mask[~valid] = False
        pillar_mask[~valid] = False
        est_depth[~valid] = 1
        pcd = point_cloud_from_rgbd(left_color, est_depth, intrinsic)
        est_depth[~valid] = 0
        coord = np.asarray(pcd.points)
        coord = coord.reshape(H, W, 3)[valid]

        color = left_color[valid]
        normal = left_normal[valid]

        obj_segment = -1 * np.ones((H, W), dtype=np.int32)
        obj_segment[plate_mask] = 0
        obj_segment[pillar_mask] = 1
        obj_segment = obj_segment[valid]

        mask = obj_segment != -1
        obj_segment_onehot = np.zeros((len(obj_segment), 2), dtype=np.float32)
        obj_segment_onehot[mask, obj_segment[mask]] = 1

        edge_vis_ids = ensure_non_circular_ids(edge_vis_mask.nonzero()[0])
        dense_visible_edge = bspline_points_down_sample(edge_cam[edge_vis_ids], 500, s=0.0, k=2)
        dense_visible_edge_pcd = point_cloud_from_points(dense_visible_edge)
        edge_mask = np.array(pcd.compute_point_cloud_distance(dense_visible_edge_pcd)).reshape(H, W)[valid] < 0.003
        edge_segment = edge_mask.astype(np.int32)

        # Debugging
        # o3d.visualization.draw_geometries([point_cloud_from_points(coord[obj_segment==1], colors=color[obj_segment==1]/255)])
        # color[edge_mask] = [255,0,0]
        # o3d.visualization.draw_geometries([point_cloud_from_points(coord, colors=color/255), dense_visible_edge_pcd])

        if i in train_ids:
            data_save_dir = train_dir / f"{i:06d}"
        elif i in val_ids:
            data_save_dir = val_dir / f"{i:06d}"
        elif i in test_ids:
            data_save_dir = test_dir / f"{i:06d}"

        data_save_dir.mkdir(exist_ok=True)

        np.save(data_save_dir / "color.npy", color)
        # np.save(data_save_dir / "depth.npy", est_depth)
        np.save(data_save_dir / "coord.npy", coord)
        np.save(data_save_dir / "normal.npy", normal)
        np.save(data_save_dir / "obj_segment.npy", obj_segment)
        np.save(data_save_dir / "obj_segment_onehot.npy", obj_segment_onehot)

        np.save(data_save_dir / "edge.npy", edge_cam)
        np.save(data_save_dir / "edge_ds.npy", edge_cam_ds)
        np.save(data_save_dir / "visible_edge.npy", dense_visible_edge)
        np.save(data_save_dir / "spl_t.npy", spline['t'])
        np.save(data_save_dir / "spl_c.npy", spline['c'])
        if spline['k'].ndim == 0:
            spline['k'] = spline['k'][None]
        np.save(data_save_dir / "spl_k.npy", spline['k'])
        np.save(data_save_dir / "segment.npy", edge_segment)

        shutil.copytree(data_save_dir, all_dir / f"{i:06d}")

    print("Data is saved to", save_dir.as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing data for pointcept")
    parser.add_argument("data_dir", type=str, help="Directory to data which include color, depth, intrinsic, pose directories.")
    parser.add_argument("--plate_name", type=str, default="bottom_plate", help="Name of plate to read mask.")
    parser.add_argument("--pillar_name", type=str, default="pillar", help="Name of pillar to read mask.")
    parser.add_argument("--train_val_test_ratio", type=str, default="25:4:1", help="Ratio for train val test dataset.")
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "../Pointcept/data/welding"), help="Directory to save data.")
    args = parser.parse_args()

    main(args.data_dir, args.plate_name, args.pillar_name, args.train_val_test_ratio, args.save_dir)