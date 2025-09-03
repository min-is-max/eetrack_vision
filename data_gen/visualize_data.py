import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import imageio
import open3d as o3d
from scipy.interpolate import splprep, splev


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


def main(data_dir):
    data_dir = Path(data_dir)

    left_color_paths = sorted((data_dir / "left" / "color").glob('*'))
    right_color_paths = sorted((data_dir / "right" / "color").glob('*'))
    # left_depth_paths = sorted((data_dir / "left" / "depth").glob('*'))
    left_depth_paths = sorted((data_dir / "left" / "est_depth").glob('*'))
    right_depth_paths = sorted((data_dir / "right" / "depth").glob('*'))
    left_normal_paths = sorted((data_dir / "left" / "normal").glob('*'))
    right_normal_paths = sorted((data_dir / "right" / "normal").glob('*'))
    intrinsic_paths = sorted((data_dir / "left" / "intrinsic").glob('*'))
    pose_paths = sorted((data_dir / "left" / "pose").glob('*'))
    edge_cam_paths = sorted((data_dir / "left" / "edge_cam").glob('*'))
    edge_cam_ds_paths = sorted((data_dir / "left" / "edge_cam_ds").glob('*'))
    edge_vis_mask_paths = sorted((data_dir / "left" / "edge_vis_mask").glob('*'))
    spline_paths = sorted((data_dir / "left" / "spline").glob('*'))
    object_data = np.load((data_dir / "object_data.npz"))
    object_pose = object_data["object_pose"]
    edge = object_data["edge"]

    left_est_depth_paths = sorted((data_dir / "left" / "est_depth").glob('*'))

    for i, (lcpath, rcpath, ldpath, rdpath, lnpath, rnpath, Kpath, Tpath,
            ecpath, ecdpath, evmpath, spath, ledpath) in enumerate(zip(
        left_color_paths,
        right_color_paths,
        left_depth_paths,
        right_depth_paths,
        left_normal_paths,
        right_normal_paths,
        intrinsic_paths,
        pose_paths,
        edge_cam_paths,
        edge_cam_ds_paths,
        edge_vis_mask_paths,
        spline_paths,
        left_est_depth_paths,
    )):
        left_color = np.asarray(Image.open(lcpath))
        left_color = np.asarray(Image.open("deploy/sample_data/left/000000.png"))
        right_color = np.asarray(Image.open(rcpath))
        left_depth = imageio.imread(ldpath)
        right_depth = imageio.imread(rdpath)
        left_normal = np.load(lnpath)
        right_normal = np.load(rnpath)
        intrinsic = np.loadtxt(Kpath)
        intrinsic = np.loadtxt("deploy/sample_data/intrinsics/000000.txt")
        # NOTE: extrinsic = T_wc = world pose in camera frame.
        extrinsic = np.linalg.inv(np.loadtxt(Tpath)) # T_cw -> T_wc
        extrinsic = np.eye(4)
        edge_cam = np.load(ecpath)
        edge_cam_ds = np.load(ecdpath)
        edge_vis_mask = np.load(evmpath)
        spline = dict(np.load(spath))
        left_est_depth = imageio.imread("outputs/depth/000000.png")

        H, W = left_color.shape[:2]

        # annos = item["annotations"]
        # bboxes = np.array(annos["bboxes"])
        # T_object_world = np.array(annos["T_scan_object"])

        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

        geometry_list = [world]

        # for bbox, T_o in zip(bboxes, T_object_world):
        #     center = T_o[:3,-1]
        #     R = T_o[:3,:3]
        #     extent = np.diff(bbox)[::2]
        #     obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        #     obb.color = (1, 0, 1)
        #     geometry_list.append(obb)

        h, w = left_color.shape[:2]
        color_o3d = o3d.geometry.Image(left_color)
        depth_o3d = o3d.geometry.Image(left_est_depth)

        # Camera intrinsics (adjust to your sensor)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            intrinsic_matrix=intrinsic,
        )
        camera_vis = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=w,
            view_height_px=h,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            scale=0.2  # Adjust for visualization size
        )

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_o3d,
            depth=depth_o3d,
            convert_rgb_to_intensity=False,
            depth_scale=1000.0,  # if depth is in mm
            depth_trunc=2.0      # ignore depth > 5m
        )

        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd,
            intrinsic=intrinsics,
            extrinsic=extrinsic,
        )

        geometry_list.extend([pcd, camera_vis])

        edge_vis_ids = ensure_non_circular_ids(edge_vis_mask.nonzero()[0])
        dense_visible_edge = bspline_points_down_sample(edge[edge_vis_ids], 500, s=0.0, k=2)
        visible_edge_pcd = point_cloud_from_points(dense_visible_edge)
        edge_mask = np.array(pcd.compute_point_cloud_distance(visible_edge_pcd)) < 0.002
        np.asarray(pcd.colors)[edge_mask] = [1.,0.,0.]
        geometry_list.append(visible_edge_pcd)

        # Visualize
        o3d.visualization.draw_geometries(geometry_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data.")
    parser.add_argument("data_dir", type=str, help="Directory to data which include color, depth, intrinsic, pose directories.")
    args = parser.parse_args()

    main(args.data_dir)