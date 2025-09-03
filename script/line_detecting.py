import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import open3d as o3d
import cv2
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt


def point_cloud_from_rgbd(
    color, depth, intrinsic, extrinsic,
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

def display_inlier_outlier(pcd, ids):
    inlier_cloud = pcd.select_by_index(ids)
    outlier_cloud = pcd.select_by_index(ids, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def extract_edge_from_point_clouds(
    source_pcd,
    target_pcd,
    dist_threshold=0.002,
    color=[1,0,0],
    remove_outlier=True,
    downsample_voxel_size=None,
    **kwargs,
):
    dist = np.asarray(source_pcd.compute_point_cloud_distance(target_pcd))
    edge_ids = (dist < dist_threshold).nonzero()[0]
    edge_pcd = source_pcd.select_by_index(edge_ids)
    if remove_outlier:
        edge_pcd, _ = edge_pcd.remove_statistical_outlier(**kwargs)
    edge_pcd.paint_uniform_color(color)
    if downsample_voxel_size is not None:
        edge_pcd = edge_pcd.voxel_down_sample(voxel_size=downsample_voxel_size)

    return edge_pcd


def find_curve_end(points, radius=0.005, neighbor_thresh=2):
    """
    Find a likely endpoint in a point cloud that lies on a 1D curve.

    Parameters:
        points (np.ndarray): (N, 3) array of 3D points.
        radius (float): radius to consider neighbors.
        neighbor_thresh (int): max number of neighbors to consider as endpoint.

    Returns:
        int: index of the detected curve endpoint.
    """
    tree = cKDTree(points)
    neighbor_counts = np.array([len(tree.query_ball_point(p, r=radius)) for p in points])

    # Find indices of candidate endpoints (fewest neighbors)
    candidates = np.where(neighbor_counts <= neighbor_thresh)[0]

    if len(candidates) == 0:
        raise ValueError("No curve endpoints found. Try increasing radius.")

    # Optional: pick the candidate farthest from the centroid
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points[candidates] - centroid, axis=1)
    endpoint_index = candidates[np.argmax(distances)]

    return endpoint_index


def sort_points(points):
    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    start_idx = find_curve_end(points)
    ordered = [start_idx]
    visited[start_idx] = True
    for _ in range(1, len(points)):
        dists, idxs = tree.query(points[ordered[-1]], k=len(points))
        next_idx = next(i for i in idxs if not visited[i])
        visited[next_idx] = True
        ordered.append(next_idx)
    return points[ordered]


def fit_spline(pcd, s=0.000005, debug_vis=False):
    sorted_points = sort_points(np.asarray(pcd.points))
    if debug_vis:
        colors = np.linspace([1,0,0], [0,0,1], len(sorted_points))
        sorted_pcd = point_cloud_from_points(sorted_points, colors=colors)
        o3d.visualization.draw_geometries([sorted_pcd])

    dists = np.linalg.norm(np.diff(sorted_points, axis=0), axis=1)
    u = np.zeros(len(sorted_points))
    u[1:] = np.cumsum(dists)
    u /= u[-1]  # Normalize to [0,1]
    tck, _ = splprep(sorted_points.T, u=u, s=s)

    return tck, lambda u: np.array(splev(u, tck)).T


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


def build_knn_graph(points, k=8, symmetric=True):
    # returns adjacency sparse matrix (weights = euclidean distance)
    tree = cKDTree(points)
    dists, idx = tree.query(points, k=k+1)  # includes self at index 0
    n = len(points)
    rows, cols, data = [], [], []
    for i in range(n):
        for j in range(1, idx.shape[1]):
            ni = idx[i, j]
            rows.append(i); cols.append(ni); data.append(dists[i, j])
            if symmetric:
                rows.append(ni); cols.append(i); data.append(dists[i, j])
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    return A


def longest_path_in_mst(adj_sparse):
    # compute MST
    mst = minimum_spanning_tree(adj_sparse)  # SciPy returns csr matrix
    mst = mst + mst.T  # make symmetric (undirected)
    G = nx.from_scipy_sparse_array(mst)
    # find tree diameter (two BFS): pick arbitrary node
    u0 = 0
    lengths = nx.single_source_dijkstra_path_length(G, u0)
    far = max(lengths, key=lengths.get)
    lengths = nx.single_source_dijkstra_path_length(G, far)
    far2 = max(lengths, key=lengths.get)
    # get path between far and far2
    path = nx.shortest_path(G, source=far, target=far2, weight='weight')
    return path, G


def fit_spline_to_path(points, path_indices, s=0.0, k=3, nb_pts=500):
    path_pts = points[path_indices]
    # compute param u by cumulative euclidean distance
    d = np.linalg.norm(np.diff(path_pts, axis=0), axis=1)
    u = np.concatenate(([0.0], np.cumsum(d)))
    if u[-1] == 0:
        u = np.linspace(0, 1, len(path_pts))
    else:
        u = u / u[-1]
    # fit parametric spline
    tck, u_out = splprep([path_pts[:,0], path_pts[:,1], path_pts[:,2]], u=u, s=s, k=min(k, len(path_pts)-1))
    u_fine = np.linspace(0, 1, nb_pts)
    x_f, y_f, z_f = splev(u_fine, tck)
    curve = np.vstack([x_f, y_f, z_f]).T
    return tck, u_fine, curve, path_pts


def project_points_to_curve(points, tck, n_samples=1000):
    # approximate projection by sampling many points on the curve and finding nearest
    u_samp = np.linspace(0,1,n_samples)
    xx, yy, zz = splev(u_samp, tck)
    curve_pts = np.vstack([xx, yy, zz]).T
    tree = cKDTree(curve_pts)
    dists, idx = tree.query(points)
    # corresponding parameter approx:
    u_proj = u_samp[idx]
    return dists, u_proj, curve_pts[idx]


def robust_curve_from_pointcloud(points,
                                 voxel_size=0.01,
                                 sor_k=16,
                                 sor_std_ratio=1.0,
                                 knn_k=8,
                                 spline_s=0.5,
                                 iter_refine=2,
                                 outlier_thresh_factor=3.0,
                                 plot=False):
    ds = points
    # 1) downsample
    # ds = voxel_downsample(points, voxel_size)
    # 2) remove statistical outliers
    # ds = statistical_outlier_removal(ds, k=sor_k, std_ratio=sor_std_ratio)
    # 3) build knn graph
    adj = build_knn_graph(ds, k=knn_k)
    # 4) MST + longest path
    path, G = longest_path_in_mst(adj)
    # 5) fit initial spline to path nodes
    tck, u_fine, curve, path_pts = fit_spline_to_path(ds, path_indices=path, s=spline_s)
    # 6) iterative refine: remove points far from curve, refit
    remaining = ds.copy()
    for it in range(iter_refine):
        dists, u_proj, _ = project_points_to_curve(remaining, tck, n_samples=2000)
        med = np.median(dists)
        mad = np.median(np.abs(dists - med))  # robust dispersion
        # threshold (robust): median + factor * mad
        thresh = med + outlier_thresh_factor * mad
        mask = dists <= thresh
        remaining = remaining[mask]
        # rebuild knn & MST on remaining to get a new path
        adj = build_knn_graph(remaining, k=knn_k)
        try:
            path, G = longest_path_in_mst(adj)
            tck, u_fine, curve, path_pts = fit_spline_to_path(remaining, path, s=spline_s)
        except Exception:
            # if MST fails due to small number, just fit to remaining sorted by projection onto principle axis
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1).fit(remaining)
            t = pca.transform(remaining).ravel()
            o = np.argsort(t)
            tck, u_fine, curve, path_pts = fit_spline_to_path(remaining, path_indices=o, s=spline_s)
    # diagnostics
    dists, u_proj, _ = project_points_to_curve(points, tck, n_samples=2000)
    result = {
        'spline_tck': tck,
        'curve_points': curve,
        'curve_u': u_fine,
        'path_nodes': path_pts,
        'final_inlier_points': remaining,
        'all_point_to_curve_dists': dists,
    }
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], s=2, alpha=0.3, label='raw')
        ax.plot(curve[:,0], curve[:,1], curve[:,2], 'r-', linewidth=2, label='spline')
        ax.scatter(path_pts[:,0], path_pts[:,1], path_pts[:,2], c='k', s=10, label='path nodes')
        ax.legend()
        ax2 = fig.add_subplot(122)
        ax2.hist(dists, bins=80)
        ax2.set_title('point-to-curve distances')
        plt.show()
    return result


def project_point_cloud(points_world, colors_rgb, K, R, t, image_dims):
    """
    Projects a 3D colored point cloud into a 2D image.

    Args:
        points_world (np.ndarray): Nx3 array of 3D points in world coordinates.
        colors_rgb (np.ndarray): Nx3 array of RGB colors for each point (0-255).
        K (np.ndarray): 3x3 camera intrinsic matrix.
        R (np.ndarray): 3x3 rotation matrix (world to camera).
        t (np.ndarray): 3x1 translation vector (world to camera).
        image_dims (tuple): (height, width) of the output image.

    Returns:
        np.ndarray: The projected color image.
        np.ndarray: The corresponding depth map.
    """
    height, width = image_dims
    
    # 1. Initialize image and depth buffer
    # OpenCV uses BGR order, so we will fill with black and convert RGB colors later
    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

    # 2. Transform points from world to camera coordinates
    # The '@' operator is used for matrix multiplication
    points_camera = (R @ points_world.T) + t
    points_camera = points_camera.T  # Transpose back to shape (N, 3)

    # 3. Filter out points that are behind the camera
    in_front_of_camera = points_camera[:, 2] > 0
    points_in_front = points_camera[in_front_of_camera]
    colors_in_front = colors_rgb[in_front_of_camera]

    if points_in_front.shape[0] == 0:
        return image, depth_buffer # No points to project

    # 4. Project 3D points in camera coordinates to 2D image plane
    # Extract coordinates
    Xc, Yc, Zc = points_in_front[:, 0], points_in_front[:, 1], points_in_front[:, 2]
    
    # Perspective projection formulas
    u = (K[0, 0] * Xc / Zc) + K[0, 2]
    v = (K[1, 1] * Yc / Zc) + K[1, 2]
    
    # 5. Filter points that fall outside the image boundaries
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u_img = u[in_frame].round().astype(int)
    v_img = v[in_frame].round().astype(int)
    colors_img = colors_in_front[in_frame]
    depths_img = Zc[in_frame]
    
    # 6. Handle occlusions using the Painter's Algorithm (sort by depth)
    # Get indices that would sort the depths in descending order (far to near)
    sort_indices = np.argsort(depths_img)[::-1]
    
    # Apply this order to pixel coordinates, colors, and depths
    u_sorted = u_img[sort_indices]
    v_sorted = v_img[sort_indices]
    colors_sorted = colors_img[sort_indices]
    depths_sorted = depths_img[sort_indices]

    # Draw the points on the image and update the depth buffer
    # Because we draw from far to near, closer points will overwrite farther ones
    image[v_sorted, u_sorted] = colors_sorted
    depth_buffer[v_sorted, u_sorted] = depths_sorted
    
    return image, depth_buffer, u_sorted, v_sorted

def image_from_point_clouds(pcds, K, T, H, W, priority="none"):
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)
    if priority == "none":
        pcd_concat = sum(pcds[1:], start=pcds[0])
        rgb, depth, u, v = project_point_cloud(np.asarray(pcd_concat.points), 255*np.asarray(pcd_concat.colors), K, T[:3,:3], T[:3,3:], (H, W))
    elif priority == "asc":
        for pcd in reversed(pcds):
            rgb_i, depth_i, u, v = project_point_cloud(np.asarray(pcd.points), 255*np.asarray(pcd.colors), K, T[:3,:3], T[:3,3:], (H, W))
            rgb[v, u] = rgb_i[v, u]
            depth[v, u] = depth_i[v, u]
    elif priority == "desc":
        for pcd in pcds:
            rgb_i, depth_i, u, v = project_point_cloud(np.asarray(pcd.points), 255*np.asarray(pcd.colors), K, T[:3,:3], T[:3,3:], (H, W))
            rgb[v, u] = rgb_i[v, u]
            depth[v, u] = depth_i[v, u]
    return rgb, depth



def main(data_dir, plate_name, pillar_name, result_dir, debug_vis=False):
    data_dir = Path(data_dir)

    left_color_paths = sorted((data_dir / "left" / "color").glob('*'))
    right_color_paths = sorted((data_dir / "right" / "color").glob('*'))
    left_depth_paths = sorted((data_dir / "left" / "depth").glob('*'))
    right_depth_paths = sorted((data_dir / "right" / "depth").glob('*'))
    intrinsic_paths = sorted((data_dir / "left" / "intrinsic").glob('*'))
    pose_paths = sorted((data_dir / "left" / "pose").glob('*'))
    edge_vis_mask_paths = sorted((data_dir / "left" / "edge_vis_mask").glob('*'))
    spline_paths = sorted((data_dir / "left" / "spline").glob('*'))
    object_data = np.load((data_dir / "object_data.npz"))
    object_pose = object_data["object_pose"]
    edge = object_data["edge"]

    plate_mask_dirs = list((data_dir / "left" / "mask").glob(plate_name + "*"))
    pillar_mask_dirs = list((data_dir / "left" / "mask").glob(pillar_name + "*"))
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
    plate_mask_paths = sorted(plate_mask_dirs[0].glob("*"))
    pillar_mask_paths = sorted(pillar_mask_dirs[0].glob("*"))


    # Generate depth estimation if they do not exist.
    left_est_depth_paths = sorted((data_dir / "left" / "est_depth").glob('*'))
    if len(left_est_depth_paths) == 0:
        print("Running depth estimation using FoudationStereo.")
        gsa2_path = os.path.join(os.path.dirname(__file__), "../FoundationStereo")
        sys.path.append(gsa2_path)

        from depth_estimator import DepthEstimator
        torch.autograd.set_grad_enabled(False)

        depth_estimator = DepthEstimator()

        est_depth_save_dir = data_dir / "left" / "est_depth"
        est_depth_save_dir.mkdir(parents=True, exist_ok=True)

        for i, (lcpath, rcpath, Kpath) in enumerate(zip(left_color_paths, right_color_paths, intrinsic_paths)):
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

        left_est_depth_paths = sorted((data_dir / "left" / "est_depth").glob('*'))

    if result_dir is not None:
        result_dir = Path(__file__).parent / result_dir
        result_dir.mkdir(parents=True, exist_ok=True)

    trg_i = input(f"Which data index to detect [0 ~ {len(left_color_paths)-1}]? Press enter to detect all:")
    if trg_i.isdigit():
        trg_i = int(trg_i)
    else:
        trg_i = None

    line_errs = []
    for i, (lcpath, rcpath, ldpath, rdpath, edpath, Kpath, Tpath, evmpath, spath, plpath, pipath) in enumerate(zip(
        left_color_paths,
        right_color_paths,
        left_depth_paths,
        right_depth_paths,
        left_est_depth_paths,
        intrinsic_paths,
        pose_paths,
        edge_vis_mask_paths,
        spline_paths,
        plate_mask_paths,
        pillar_mask_paths,
    )):
        if trg_i is not None and not i == trg_i:
            continue

        left_color = np.asarray(Image.open(lcpath))
        right_color = np.asarray(Image.open(rcpath))
        left_depth = imageio.imread(ldpath)
        right_depth = imageio.imread(rdpath)
        est_depth = imageio.imread(edpath)
        intrinsic = np.loadtxt(Kpath)
        # NOTE: extrinsic = T_wc = world pose in camera frame.
        extrinsic = np.linalg.inv(np.loadtxt(Tpath)) # T_cw -> T_wc
        edge_vis_mask = np.load(evmpath)
        spline = np.load(spath)
        plate_mask = imageio.imread(plpath) > 0
        pillar_mask = imageio.imread(pipath) > 0

        valid = est_depth > 0
        px_depth_err = np.abs(left_depth[valid]/1000 - est_depth[valid]/1000).mean()
        # print("Estimated depth error:", px_depth_err)

        # Remove mask for invalid depth.
        plate_mask[~valid] = False
        pillar_mask[~valid] = False
        # Set invalid depth to 1 mm to keep the shape.
        est_depth[~valid] = 1

        H, W = left_color.shape[:2]

        pcd = point_cloud_from_rgbd(
            left_color, left_depth, intrinsic, extrinsic
        )

        est_pcd = point_cloud_from_rgbd(
            left_color, est_depth, intrinsic, extrinsic
        )
        # if debug_vis:
        #     print("Visualizing ground-truth point cloud and estimate point cloud.")
        #     o3d.visualization.draw_geometries([pcd, est_pcd])

        # ft_est_pcd, ft_est_pcd_ids = est_pcd.remove_statistical_outlier(
        #     nb_neighbors = 40,
        #     std_ratio = 1.0,
        # )
        # est_depth_outlier_uv = np.unravel_index(ft_est_pcd_ids, (H,W))
        # est_depth_outlier_uv_mask = np.zeros_like(est_depth, dtype=bool)
        # est_depth_outlier_uv_mask[est_depth_outlier_uv] = True
        # plate_mask[~est_depth_outlier_uv_mask] = False
        # pillar_mask[~est_depth_outlier_uv_mask] = False
        # if debug_vis:
        #     print("Visualizing estimate point cloud with outlier removal.")
        #     display_inlier_outlier(est_pcd, ft_est_pcd_ids)
        #     o3d.visualization.draw_geometries([pcd, ft_est_pcd])

        points = np.asarray(est_pcd.points).reshape((H, W, 3))
        colors = np.asarray(est_pcd.colors).reshape((H, W, 3))

        plate_pcd = point_cloud_from_points(points[plate_mask], [0, 1, 0]) # colors[plate_mask])
        pillar_pcd = point_cloud_from_points(points[pillar_mask], [0, 0, 1]) # colors[pillar_mask])

        # Filter outlier
        # TODO: make better outlier removal.
        # Too strict params remove too many pcd including welding line egde.
        # Too soft params cannot effectively remove outliers.
        # Doing this at full pcd would be better.
        # ft_plate_pcd, ft_plate_ids = plate_pcd.remove_statistical_outlier(
        #     nb_neighbors = 30,
        #     std_ratio = 1.0,
        # )
        # ft_pillar_pcd, ft_pillar_ids = pillar_pcd.remove_statistical_outlier(
        #     nb_neighbors = 30,
        #     std_ratio = 1.0,
        # )
        ft_plate_pcd, ft_plate_ids = plate_pcd.remove_radius_outlier(
            nb_points=25,
            radius=0.005,
        )
        ft_pillar_pcd, ft_pillar_ids = pillar_pcd.remove_radius_outlier(
            nb_points=25,
            radius=0.005,
        )
        if debug_vis:
            print("Visualizing plate pcd outlier with red color.")
            display_inlier_outlier(plate_pcd, ft_plate_ids)
            print("Visualizing pillar pcd outlier with red color.")
            display_inlier_outlier(pillar_pcd, ft_pillar_ids)

        # Extrack edge from two point clouds v1
        # edge_pcd = extract_edge_from_point_clouds(
        #     ft_plate_pcd,
        #     ft_pillar_pcd,
        #     dist_threshold=0.0015,
        #     remove_outlier=True,
        #     nb_neighbors=5,
        #     std_ratio=1.0,
        #     downsample_voxel_size=0.0075,
        # )
        # Extrack edge from two point clouds v2
        plate_dist = np.array(est_pcd.compute_point_cloud_distance(ft_plate_pcd))
        pillar_dist = np.array(est_pcd.compute_point_cloud_distance(ft_pillar_pcd))

        edge_mask = (plate_dist < 0.0015) & (pillar_dist < 0.0015)
        edge_points = np.asarray(est_pcd.points)[edge_mask]
        edge_pcd = point_cloud_from_points(edge_points, [1,0,0])
        if debug_vis:
            print("Visualizing edge pcd.")
            o3d.visualization.draw_geometries([ft_plate_pcd, ft_pillar_pcd, edge_pcd])

        # Fit edge point cloud into spline v1
        # tck, spline_func = fit_spline(edge_pcd, debug_vis=debug_vis)
        # u = np.linspace(0, 1, 1000)
        # line_points = spline_func(u)
        # line_pcd = point_cloud_from_points(line_points, [0,1,1])
        # Fit edge point cloud into spline v2
        try:
            res = robust_curve_from_pointcloud(np.asarray(est_pcd.points)[edge_mask], voxel_size=0.02, sor_k=16, sor_std_ratio=1.0,
                                                knn_k=8, spline_s=0.000025, iter_refine=2, outlier_thresh_factor=3.0, plot=False)
        except:
            continue
        line_pcd = point_cloud_from_points(res["curve_points"], [0, 1, 1])
        if debug_vis:
            print("Visualizing fitted spline from edge pcd")
            o3d.visualization.draw_geometries([edge_pcd, line_pcd])

        if debug_vis:
            # Final visualization
            world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
            geometry_list = [world, est_pcd, line_pcd]
            o3d.visualization.draw_geometries(geometry_list)

        # Compute metric
        edge_vis_ids = ensure_non_circular_ids(edge_vis_mask.nonzero()[0])
        dense_visible_edge = bspline_points_down_sample(edge[edge_vis_ids], 500, s=0.0, k=2)
        dense_visible_edge_pcd = point_cloud_from_points(dense_visible_edge, [1, 0, 0])
        line_err = np.mean(dense_visible_edge_pcd.compute_point_cloud_distance(line_pcd))
        line_errs.append(line_err)
        print(f"Line detection error {i}: {line_err:.5f}m")

        # For more detailed line visualization
        scale = 2
        intrinsic[:2] *= scale
        proj_rgb, proj_depth = image_from_point_clouds([est_pcd, dense_visible_edge_pcd, line_pcd], intrinsic, extrinsic, scale*H, scale*W, priority="desc")
        if result_dir is not None:
            save_path = result_dir / f"{i}.png"
            Image.fromarray(proj_rgb).save(save_path)

    print("Mean line error: ", np.mean(line_errs))
    print("Std line error: ", np.std(line_errs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Line detection algorithm.")
    parser.add_argument("data_dir", type=str, help="Directory to data which include left/right color, depth, intrinsic, pose directories.")
    parser.add_argument("--plate_name", type=str, default="bottom_plate", help="Name of plate to read mask.")
    parser.add_argument("--pillar_name", type=str, default="pillar", help="Name of pillar to read mask.")
    parser.add_argument("--result_dir", type=str, default="output", help="Directory to save result.")
    parser.add_argument("--debug_vis", action="store_true", default=False, help="Enable debug visualization")
    args = parser.parse_args()

    main(args.data_dir, args.plate_name, args.pillar_name, args.result_dir, args.debug_vis)
