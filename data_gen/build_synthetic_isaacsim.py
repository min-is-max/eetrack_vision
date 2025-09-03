# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the different camera sensors that can be attached to a robot.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/sensors/cameras.py --enable_cameras

    # Usage in headless mode
    ./isaaclab.sh -p scripts/demos/sensors/cameras.py --headless --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the different camera sensor implementations.")
parser.add_argument("--object_usd_root", type=str, required=True, help="Path to root of object usd files. Any usd under this root will be used.")
parser.add_argument("--seed", type=int, default=42, help="Seed for random generator.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--path_tracing", action="store_true", default=False, help="Use path tracing for rendering.")
parser.add_argument("--baseline", type=float, default=0.063, help="Basline between left and right cameras.")
parser.add_argument("--debug_vis", action="store_true", default=False, help="Visualize camera pose for debugging.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from PIL import Image
import imageio
import torch.nn.functional as F
from scipy.interpolate import splprep, splev
import open3d as o3d
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers, FRAME_MARKER_CFG
import isaacsim.core.utils.stage as stage_utils
import omni.replicator.core as rep
import isaacsim.core.utils.torch as torch_utils
# from isaacsim.util.debug_draw import _debug_draw
from pxr import UsdGeom
from isaacsim.core.simulation_manager import SimulationManager
import Semantics
from isaaclab.sim.utils import find_matching_prims
import omni.usd
import isaacsim.core.utils.prims as prim_utils


def get_world_poses_from_view(
    eyes: torch.Tensor, targets: torch.Tensor,
):
    up_axis = stage_utils.get_stage_up_axis()
    quat = math_utils.quat_from_matrix(
        math_utils.create_rotation_matrix_from_view(
            eyes, targets, up_axis, device=eyes.device
        )
    )

    return eyes, quat


def get_usd_paths(asset):
    stage = omni.usd.get_context().get_stage()
    
    prim_paths = asset.root_physx_view.prim_paths
    usd_paths = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        stack = prim.GetPrimStack()
        layer = stack[1].layer
        usd_path = layer.GetAssetInfo()["url"]
        usd_paths.append(usd_path)

    return usd_paths


def set_semantics(scene, prim_sem_dict):
    for prim_path, sem_name in prim_sem_dict.items():
        prim_path = prim_path.format(ENV_REGEX_NS=scene.env_regex_ns)
        prims = find_matching_prims(prim_path)
        for prim in prims:
            sem = Semantics.SemanticsAPI.Apply(prim, "class_" + sem_name)
            sem.CreateSemanticTypeAttr()
            sem.CreateSemanticDataAttr()
            sem.GetSemanticTypeAttr().Set("class")
            sem.GetSemanticDataAttr().Set(sem_name)


class PosedCamera(Camera):
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "pose_visualizer"):
                self.pose_visualizer = VisualizationMarkers(self.cfg.pose_visualizer_cfg)
            self.pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "pose_visualizer"):
                self.pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._is_initialized:
            return

        self.pose_visualizer.visualize(self._data.pos_w, self._data.quat_w_ros)


@configclass
class PosedCameraCfg(CameraCfg):
    class_type: type = PosedCamera

    pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/CameraPose")

    def __post_init__(self):
        self.pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        debug_vis=False,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        # spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
        spawn=sim_utils.DistantLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)),
        # init_state=AssetBaseCfg.InitialStateCfg(
        #     rot=R.from_euler('xy', [30, 30], degrees=True).as_quat(scalar_first=True),
        # ),
        # spawn=sim_utils.CylinderLightCfg(intensity=5000.0, color=(1.0, 1.0, 1.0), radius=0.05),
        # init_state=AssetBaseCfg.InitialStateCfg(
        #     pos=(5, -5, 1),
        # ),
    )

    # weld_object
    weld_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/weld_object",
        # spawn=sim_utils.UsdFileCfg(
        #     usd_path=None,
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        # ),
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=[
                "data_gen/assets/weld_objects/usd/specimen/1/1.usd",
                "data_gen/assets/weld_objects/usd/specimen/2/2.usd",
                "data_gen/assets/weld_objects/usd/specimen/3/3.usd",
                "data_gen/assets/weld_objects/usd/specimen/4/4.usd",
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0,0.0,0.0),
        ),
    )

    # sensors
    left_camera = PosedCameraCfg(
        prim_path="{ENV_REGEX_NS}/left_cam",
        # Zed mini (TODO: fix value with real camera intrinsics)
        height=376,
        width=672,
        data_types=["rgb", "depth", "normals", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.06, focus_distance=28.0, horizontal_aperture=2.016, vertical_aperture=1.128, clipping_range=(0.1, 1.0e5)
        ),
        offset=PosedCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.5), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        colorize_semantic_segmentation=False,
        debug_vis=False,
    )
    right_camera = PosedCameraCfg(
        prim_path="{ENV_REGEX_NS}/right_cam",
        height=376,
        width=672,
        data_types=["rgb", "depth", "normals", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.06, focus_distance=28.0, horizontal_aperture=2.016, vertical_aperture=1.128, clipping_range=(0.1, 1.0e5)
        ),
        offset=PosedCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.5), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        colorize_semantic_segmentation=False,
        debug_vis=False,
    )
    # NOTE: There is a bug for camera pose update in TiledCamera


def _random_object_pose(pose_range, size, device="cuda"):
    ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
        device=device,
    )
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (size, 6), device=device)

    rand_pos = rand_samples[:, 0:3]
    rand_quat = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])

    return torch.cat([rand_pos, rand_quat], dim=-1)


def _random_spherical(phi_min, phi_max, r_min, r_max):
    """랜덤 θ, φ, r 샘플링"""
    theta = np.random.uniform(0, 2*torch.pi)
    phi   = np.random.uniform(phi_min, phi_max)
    r     = np.random.uniform(r_min, r_max)
    return theta, phi, r

def _random_target():
    """랜덤 타겟 위치 생성"""
    # return (0, 0, 0)
    return (np.random.uniform(-0.1, 0.1), 
            np.random.uniform(-0.1, 0.1), 
            np.random.uniform(-0.1, 0.1))

def _shortest_delta(a_from: float, a_to: float) -> float:
    """
    θ 보간 시, 2π 경계를 넘어가지 않도록 최단 δ 계산
    """
    raw = a_to - a_from
    return (raw + np.pi) % (2*np.pi) - np.pi


def build_camera_trajectory(n_views=60, n_anchors=4, phi_minmax=[np.radians(30), np.radians(75)], r_minmax=[0.4, 1.3]):
    anchors = [_random_spherical(*phi_minmax, *r_minmax) for _ in range(n_anchors)]
    target_anchors = [_random_target() for _ in range(n_anchors)]

    traj = []
    # 한 세그먼트당 뷰 개수
    seg_len = n_views // (n_anchors - 1)
    for i in range(n_anchors - 1):
        theta0, phi0, r0 = anchors[i]
        theta1, phi1, r1 = anchors[i+1]
        # Target anchors for smooth interpolation
        target0 = target_anchors[i]
        target1 = target_anchors[i+1]
        for t_step in range(seg_len):
            t = t_step / seg_len
            theta = theta0 + _shortest_delta(theta0, theta1) * t
            phi = phi0 + (phi1 - phi0) * t
            r = r0 + (r1 - r0) * t
            # Smooth target interpolation
            target = [
                target0[0] + (target1[0] - target0[0]) * t,
                target0[1] + (target1[1] - target0[1]) * t,
                target0[2] + (target1[2] - target0[2]) * t
            ]
            traj.append((theta, phi, r, *target))
    # 남는 뷰가 있으면 마지막 앵커 위치 반복
    while len(traj) < n_views:
        last_anchor = anchors[-1]
        last_target = target_anchors[-1]
        traj.append((last_anchor[0], last_anchor[1], last_anchor[2], *last_target))
    return traj


def init_open3d_ray_casting_scene(mesh_prim_path):
    _physics_sim_view = SimulationManager.get_physics_sim_view()
    _root_physx_view = _physics_sim_view.create_rigid_body_view(mesh_prim_path)

    transform_matrix = np.eye(4)
    pose = _root_physx_view.get_transforms().clone()
    transform_matrix[:3,3] = pose[0,:3].cpu().numpy()
    pose[:,3:7] = math_utils.convert_quat(pose[:,3:7], to="wxyz")
    R = math_utils.matrix_from_quat(pose[:,3:7])
    transform_matrix[:3,:3] = R[0].cpu().numpy()

    prims = sim_utils.get_all_matching_child_prims(
        mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
    )

    ray_scene = o3d.t.geometry.RaycastingScene()
    for prim in prims:
        mesh_prim = UsdGeom.Mesh(prim)
        points = np.asarray(mesh_prim.GetPointsAttr().Get())
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).reshape(-1,3).copy()
        points = np.matmul(points, transform_matrix[:3, :3].T)
        points += transform_matrix[:3, 3]

        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(indices))
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        ray_scene.add_triangles(mesh)
    
    return ray_scene


def get_empty_cam_data_dict():
    return {
        "colors": [],
        "depths": [],
        "normals": [],
        "semantics": [],
        "Ks": [],
        "Ts": [],
        "edge_cam": [],
        "edge_cam_ds": [],
        "edge_vis_masks": [],
        "spline": {
            "t": [],
            "c": [],
            "k": [],
        },
        "idToLabels": {},
    }


def append_cam_data(scene, data_dict, camera):
    data_dict["colors"].append(camera.data.output["rgb"].detach().clone().cpu())
    data_dict["depths"].append(camera.data.output["depth"][...,0].detach().clone().cpu())
    data_dict["normals"].append(camera.data.output["normals"].detach().clone().cpu())
    data_dict["semantics"].append(camera.data.output["semantic_segmentation"][...,0].detach().clone().cpu())
    data_dict["idToLabels"].update(camera.data.info[0]["semantic_segmentation"]["idToLabels"])
    data_dict["Ks"].append(camera.data.intrinsic_matrices.detach().clone().cpu())
    T = torch.eye(4, device=scene.device).repeat(scene.num_envs,1,1)
    T[:,:3,:3] = math_utils.matrix_from_quat(camera.data.quat_w_ros)
    T[:,:3,3] = camera.data.pos_w - scene.env_origins
    data_dict["Ts"].append(T.detach().clone().cpu())

    return data_dict


def transform_points_inv(points, pos, quat):
    return math_utils.transform_points(
        points,
        *math_utils.subtract_frame_transforms(pos, quat)
    )


def get_visible_masks(edges, camera, o3d_ray_scenes=None):
    B, N = edges.shape[:2]
    H, W = camera.data.image_shape

    depth = camera.data.output["depth"][...,0].detach().clone()
    edges_cam = transform_points_inv(edges, camera.data.pos_w, camera.data.quat_w_ros)
    edges_cam_dist = edges_cam.norm(dim=-1)
    edges_uvd = math_utils.project_points(
        edges_cam,
        camera.data.intrinsic_matrices,
    )
    if edges_uvd.dim() == 2:
        edges_uvd = edges_uvd[None]
    edges_uv, edges_d = edges_uvd[...,:2], edges_uvd[...,2]
    in_front = edges_d > 0
    # TODO: Fix in_uv.
    in_uv = (
        (torch.tensor([0., 0.], device=edges.device) < edges_uv) & 
        (edges_uv < torch.tensor([W, H], device=edges.device))
    ).all(dim=-1)

    # Check occlusion.
    if o3d_ray_scenes is None:
        norm_uv = edges_uv.clone()
        norm_uv[..., 0] = 2.0 * norm_uv[..., 0] / (W - 1) - 1.0
        norm_uv[..., 1] = 2.0 * norm_uv[..., 1] / (H - 1) - 1.0
        grid = norm_uv.view(B, N, 1, 2)
        interp_depth = F.grid_sample(depth.unsqueeze(1), grid, align_corners=True, mode='bilinear', padding_mode='border')[:,0,:,0]
        not_occ = edges_d < (interp_depth + 3e-3)
    else:
        not_occ = torch.zeros(edges.shape[:-1], dtype=torch.bool, device=edges.device)
        for i, ray_scene in enumerate(o3d_ray_scenes):
            ray_pos = camera.data.pos_w[i].repeat(edges[i].shape[0], 1)
            ray_dir = torch.nn.functional.normalize(edges[i] - camera.data.pos_w[i], dim=-1)
            rays = torch.cat([ray_pos, ray_dir], dim=-1)
            rays = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(rays.cpu()))
            ans = ray_scene.cast_rays(rays)
            hit = ans['t_hit'].isfinite()
            t_hit = torch.utils.dlpack.from_dlpack(ans['t_hit'].to_dlpack()).to(edges_d.device)
            not_occ[i] = edges_cam_dist[i] < (t_hit + 1e-3)

    vis_masks = in_front & in_uv & not_occ

    # global i
    # env_i = 0
    # rgb = camera.data.output["rgb"][env_i].clone()
    # uv = edges_uv[env_i, vis_masks[env_i]].round().int()
    # rgb[uv[:,1], uv[:,0]] = 0
    # rgb[uv[:,1], uv[:,0], 0] = 255
    # plt.imshow(rgb.cpu().numpy())
    # plt.savefig(f"{i}")
    # plt.close()
    # i += 1

    return vis_masks


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


def append_edge_data(
    edges: torch.Tensor,
    data_dict: dict,
    camera,
    o3d_ray_scenes=None,
    target_num_down_sample_pts = 16,
    **spline_kwargs,
):
    data_dict["edge_vis_masks"].append(get_visible_masks(edges, camera, o3d_ray_scenes))
    edge_cam = transform_points_inv(edges, camera.data.pos_w, camera.data.quat_w_ros)
    data_dict["edge_cam"].append(edge_cam)
    edge_cam_ds = bspline_points_down_sample(edge_cam.cpu().numpy(), target_num_down_sample_pts, **spline_kwargs)
    data_dict["edge_cam_ds"].append(edge_cam_ds)
    tck = fit_spline(edge_cam_ds, **spline_kwargs)
    for i, k in enumerate(['t', 'c', 'k']): data_dict["spline"][k].append(tck[i])


def save_object_data(
    object_poses: np.ndarray,
    edges: np.ndarray,
    save_dir: str,
):
    assert len(object_poses) == len(edges), "Batch sizes of object_poses and edges should be same."
    B = len(object_poses)

    for i in range(B):
        save_path = os.path.join(save_dir, str(i), "object_data.npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            object_pose=object_poses[i],
            edge=edges[i],
        )


def save_data(
    left_cam_data_dict: dict,
    right_cam_data_dict: dict,
    save_dir: str,
):
    for cam in ["left", "right"]:
        cam_data = left_cam_data_dict if cam == "left" else right_cam_data_dict

        colors = cam_data["colors"]
        depths = cam_data["depths"]
        normals = cam_data["normals"]
        semantics = cam_data["semantics"]
        id_to_labels = cam_data["idToLabels"]
        intrinsics = cam_data["Ks"]
        extrinsics = cam_data["Ts"]
        edge_cam = cam_data["edge_cam"]
        edge_cam_ds = cam_data["edge_cam_ds"]
        edge_vis_masks = cam_data["edge_vis_masks"]
        spline = cam_data["spline"]

        T = len(colors)
        B = len(colors[0])

        for b in range(B):
            os.makedirs(os.path.join(save_dir, str(b), cam, "color"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, str(b), cam, "depth"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, str(b), cam, "normal"), exist_ok=True)
            mask_gt_dir = os.path.join(save_dir, str(b), cam, "mask_gt")
            for sem_id, label in id_to_labels.items():
                os.makedirs(os.path.join(mask_gt_dir, label["class"]), exist_ok=True)
            os.makedirs(os.path.join(save_dir, str(b), cam, "intrinsic"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, str(b), cam, "pose"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, str(b), cam, "edge_cam"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, str(b), cam, "edge_cam_ds"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, str(b), cam, "edge_vis_mask"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, str(b), cam, "spline"), exist_ok=True)

        for t in range(T):
            fn = f"{t:06d}"
            for b in range(B):
                # Color
                rgb_save_path = os.path.join(save_dir, str(b), cam, "color", f"{fn}.jpg")
                Image.fromarray(colors[t][b].cpu().numpy()).save(rgb_save_path)
                # Depth
                depth_save_path = os.path.join(save_dir, str(b), cam, "depth", f"{fn}.png")
                imageio.imwrite(depth_save_path,
                                (1000*depths[t][b].cpu().numpy()).astype(np.uint16))
                # Normal
                normal_save_path = os.path.join(save_dir, str(b), cam, "normal", f"{fn}.npy")
                np.save(normal_save_path, normals[t][b].cpu().numpy())
                mask_gt_dir = os.path.join(save_dir, str(b), cam, "mask_gt")
                for sem_id, label in id_to_labels.items():
                    mask = semantics[t][b].cpu().numpy() == int(sem_id)
                    sem_save_path = os.path.join(mask_gt_dir, label["class"], f"{fn}.png")
                    imageio.imwrite(sem_save_path, mask.astype(np.uint8)*255)
                # Intrinsic
                K_save_path = os.path.join(save_dir, str(b), cam, "intrinsic", f"{fn}.txt")
                np.savetxt(K_save_path, intrinsics[t][b].cpu().numpy(), fmt="%.6f")
                # Extrinsic
                T_save_path = os.path.join(save_dir, str(b), cam, "pose", f"frame-{fn}.pose.txt")
                np.savetxt(T_save_path, extrinsics[t][b].cpu().numpy(), fmt="%.6f")
                # Edge in camera frame
                edge_cam_save_path = os.path.join(save_dir, str(b), cam, "edge_cam", f"{fn}.npy")
                np.save(edge_cam_save_path, edge_cam[t][b].cpu().numpy())
                # Down-sampled edge in camera frame
                edge_cam_ds_save_path = os.path.join(save_dir, str(b), cam, "edge_cam_ds", f"{fn}.npy")
                np.save(edge_cam_ds_save_path, edge_cam_ds[t][b])
                # Edge visible mask
                edge_vis_mask_save_path = os.path.join(save_dir, str(b), cam, "edge_vis_mask", f"{fn}.npy")
                np.save(edge_vis_mask_save_path, edge_vis_masks[t][b].cpu().numpy())
                # Spline
                spline_save_path = os.path.join(save_dir, str(b), cam, "spline", f"{fn}.npz")
                spline_data = {k: v[t][b] for k, v in spline.items()}
                np.savez(spline_save_path, **spline_data)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Create output directory to save images
    object_types = {Path(up).parents[1].name for up in scene["weld_object"].cfg.spawn.usd_path}
    name = "_".join(object_types)
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", name)
    os.makedirs(output_dir, exist_ok=True)

    usd_paths = get_usd_paths(scene["weld_object"])
    edge_paths = [os.path.dirname(up).replace("usd", "edges") + ".npy" for up in usd_paths]
    edge_obj = torch.tensor([np.load(ep).astype(np.float32) for ep in edge_paths], device=scene.device)

    # TODO: implement batched version
    cam_trajs = torch.tensor([build_camera_trajectory() for _ in range(scene.num_envs)], device=scene.device)
    # B x T x 6 -> T x B x 6
    cam_trajs= cam_trajs.transpose(0,1)

    # TODO: Fix hard-coded.
    set_semantics(
        scene,
        prim_sem_dict = {
        "{ENV_REGEX_NS}/weld_object/geometry/node_.*/plate": "plate",
        "{ENV_REGEX_NS}/weld_object/geometry/node_.*/pillar": "pillar",
        },
    )

    # Simulate physics
    while simulation_app.is_running():
        # Few pre-steps for clean rendering at the first frames
        for _ in range(10):
            sim.step()
            scene.update(sim_dt)
            scene["left_camera"].data.output["rgb"]
            scene["right_camera"].data.output["rgb"]

        # sample random object pose
        pose_range = {"yaw": [-torch.pi, torch.pi]}
        rand_pose_env = _random_object_pose(pose_range, scene.num_envs, device=scene.device)
        rand_pose_w = rand_pose_env.clone()
        rand_pose_w[:,:3] += scene.env_origins
        scene["weld_object"].write_root_pose_to_sim(rand_pose_w)
        edges_env = math_utils.transform_points(edge_obj, rand_pose_env[:,:3], rand_pose_env[:,3:])
        edges_w = math_utils.transform_points(edge_obj, rand_pose_w[:,:3], rand_pose_w[:,3:])
        if args_cli.num_envs == 1:
            edges_env = edges_env[None]
            edges_w = edges_w[None]
        # draw_interface = _debug_draw.acquire_debug_draw_interface()
        # debug_points = edges_w.reshape(-1,3).tolist()
        # debug_colors = [[1.0, 0.0, 0.0, 1.0] for _ in range(edges_w.reshape(-1,3).shape[0])]
        # debug_sizes = [5.0 for _ in range(edges_w.reshape(-1,3).shape[0])]
        # draw_interface.draw_points(debug_points, debug_colors, debug_sizes)
        # -- write data to sim
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # o3d_ray_scenes = None
        o3d_ray_scenes = [init_open3d_ray_casting_scene(f"/World/envs/env_{i}/weld_object") for i in range(scene.num_envs)]

        left_cam_data_dict = get_empty_cam_data_dict()
        right_cam_data_dict = get_empty_cam_data_dict()
        for cam_traj in tqdm(cam_trajs):
            # Reset
            theta = cam_traj[:,0]
            phi = cam_traj[:,1]
            r = cam_traj[:,2]
            target = cam_traj[:,3:]

            cam_pos = torch.stack([
                r * phi.sin() * theta.cos(),
                r * phi.sin() * theta.sin(),
                r * phi.cos(),
            ], dim=1)

            cam_pos += scene.env_origins
            target += scene.env_origins
            # opengl convention
            left_eyes, quat = get_world_poses_from_view(cam_pos, target)
            right_eyes = left_eyes + args_cli.baseline * math_utils.matrix_from_quat(quat)[:,:3,0]
            scene["left_camera"].set_world_poses(
                left_eyes,
                quat,
                convention = "opengl",
            )
            scene["right_camera"].set_world_poses(
                right_eyes,
                quat,
                convention = "opengl",
            )

            # perform step
            sim.step()
            count += 1
            # update buffers
            scene.update(sim_dt)

            append_edge_data(edges_w, left_cam_data_dict, scene["left_camera"], o3d_ray_scenes, s=0.0, k=2)
            append_edge_data(edges_w, right_cam_data_dict, scene["right_camera"], o3d_ray_scenes, s=0.0, k=2)
            append_cam_data(scene, left_cam_data_dict, scene["left_camera"])
            append_cam_data(scene, right_cam_data_dict, scene["right_camera"])

        save_object_data(
            object_poses=rand_pose_env.cpu().numpy(),
            edges=edges_env.cpu().numpy(),
            save_dir=output_dir,
        )

        save_data(
            left_cam_data_dict,
            right_cam_data_dict,
            save_dir=output_dir,
        )

        print("RGB-D data generation done!")
        break




def main():
    """Main function."""
    # Set seed
    seed = args_cli.seed
    if seed is not None:
        rep.set_global_seed(seed)
        torch_utils.set_seed(seed)

    # Initialize the simulation context
    if args_cli.path_tracing:
        render_cfg = sim_utils.RenderCfg(
            enable_translucency=True,
            enable_reflections=True,
            enable_global_illumination=True,
            enable_direct_lighting=True,
            enable_shadows=True,
            enable_ambient_occlusion=True,
            carb_settings={
                "rtx.rendermode": "PathTracing",
            }
        )
    else:
        render_cfg = sim_utils.RenderCfg()

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
        render=render_cfg,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=10.0,
        # Should be disabled for different assets in different env.
        replicate_physics=False,
    )
    usd_paths = [p.as_posix() for p in sorted(
        Path(args_cli.object_usd_root).rglob("*.usd"),
        key=lambda p: int(p.name.split('.')[0]),
    )]
    scene_cfg.weld_object.spawn.usd_path = usd_paths
    scene_cfg.left_camera.debug_vis = args_cli.debug_vis
    scene_cfg.right_camera.debug_vis = args_cli.debug_vis
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
