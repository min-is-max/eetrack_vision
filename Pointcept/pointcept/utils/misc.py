"""
Misc

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import warnings
from collections import abc
import numpy as np
import torch
from importlib import import_module
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
import networkx as nx


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def build_bspline_knots(batch_size, pts_size, degree, method="uniform", t_min=0, t_max=1):
    if method == "uniform":
        knots = torch.linspace(t_min, t_max, pts_size - degree + 1).repeat(batch_size, 1)
        knots = torch.nn.functional.pad(knots, (degree,0), "constant", 0)
        knots = torch.nn.functional.pad(knots, (0,degree), "constant", 1)

    return knots


def build_bspline_fn_batch(t: torch.Tensor, c: torch.Tensor, k: int):
    dtype = t.dtype
    device = t.device

    B = len(t)
    t = t.cpu().numpy()
    c = c.cpu().numpy()

    def bspline(u: torch.Tensor):
        u = u.cpu().numpy()

        out = []
        for i in range(B):
            out.append(splev(u[i], (t[i], c[i], k)))
        out = torch.tensor(out, dtype=dtype, device=device).transpose(1,2)
        return out

    return bspline


def compute_point_clouds_distance(points1: torch.Tensor, points2: torch.Tensor, reduction="mean"):
    point_wise_dist = (points1[:,None] - points2[:,:,None]).norm(dim=-1)
    near_dist_12_mean = point_wise_dist.min(dim=1)[0].mean(dim=1)
    near_dist_21_mean = point_wise_dist.min(dim=2)[0].mean(dim=1)

    near_dist_mean = (near_dist_12_mean + near_dist_21_mean) / 2

    if reduction == "mean":
        near_dist_mean = near_dist_mean.mean()

    return near_dist_mean


def find_curve_end(points: np.ndarray, radius=0.001, neighbor_thresh=2):
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


def sort_points(points: np.ndarray):
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


def fit_spline(points: np.ndarray, s=0.000005):
    sorted_points = sort_points(points)

    dists = np.linalg.norm(np.diff(sorted_points, axis=0), axis=1)
    u = np.zeros(len(sorted_points))
    u[1:] = np.cumsum(dists)
    u /= u[-1]  # Normalize to [0,1]
    tck, _ = splprep(sorted_points.T, u=u, s=s)

    return tck, lambda u: np.array(splev(u, tck)).T

def voxel_downsample(points, voxel_size):
    if voxel_size <= 0:
        return points
    # compute voxel indices
    mins = points.min(axis=0)
    idx = np.floor((points - mins) / voxel_size).astype(np.int64)
    # unique voxels -> average points in voxel
    _, inv, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    summed = np.zeros((inv.max()+1, 3), dtype=float)
    np.add.at(summed, inv, points)
    centroids = summed / counts[:, None]
    return centroids

def statistical_outlier_removal(points, k=16, std_ratio=1.0):
    # compute mean distance to k neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    dists, _ = nbrs.kneighbors(points)
    # exclude self (first column)
    mean_dist = dists[:, 1:].mean(axis=1)
    mu = mean_dist.mean()
    sigma = mean_dist.std()
    mask = mean_dist <= (mu + std_ratio * sigma)
    return points[mask]

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


class DummyClass:
    def __init__(self):
        pass
