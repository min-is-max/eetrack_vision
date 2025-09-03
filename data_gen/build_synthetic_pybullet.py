#!/usr/bin/env python

import argparse, pathlib, tempfile, os, math, json
from typing import List, Tuple, Optional
import numpy as np
import trimesh
from PIL import Image
import imageio
import pybullet as pb
import pybullet_data
import zipfile, shutil, tempfile
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# 회전 매트릭스에서 직접 quaternion 추출 (올바른 방법)
def matrix_to_quaternion(R):
    """회전 매트릭스를 quaternion [w,x,y,z]로 변환"""
    trace = R[0,0] + R[1,1] + R[2,2]
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2  # s = 4 * qx
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2  # s = 4 * qy
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2  # s = 4 * qz
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    
    return [w, x, y, z]

# ────────────────────────────────────────────────────────────────────────────
# 1) Coordinate transformation matrix (Z-up → Y-up)

T_Z2YN = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
T_Z2Y = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0,-1, 0, 0],
                  [0, 0, 0, 1]], dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────────
# 2) Renderer Class (base on PyBullet)

class PyBulletRenderer:
    """
    Generate RGB and Depth images using PyBullet TinyRenderer.
    Return: (rgb[H,W,3] uint8, depth[m][H,W] float32, K[3,3], T_camW[4,4])
    """

    def __init__(self,
                 glb_path: str,
                 n_views: int = 60,
                 img_wh: int = 640,
                 near: float = 0.2,
                 far: float = 4.0) -> None:

        self.n_views = n_views
        self.img_wh = img_wh
        self.near, self.far = near, far

        # ── PyBullet Initization ────────────────────────────────────
        pb.connect(pb.DIRECT)                                       # No GUI, Headless mode
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())     # pybullet data path
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)           # Disable GUI

        # ── GLB → OBJ(temp) → PyBullet load ─────────────────────────
        mesh = trimesh.load(glb_path, force='mesh', process=False)  # GLB: Single Mesh
        tmp_obj = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)  # Pybullet requires .obj
        mesh.export(tmp_obj.name)                                   # GLB → Temporary OBJ (Trimesh)

        # ── PyBullet Collision/Visual Shape Generation ────────────────────
        scale = [0.001, 0.001, 0.001]                               # GLB: mm → m
        col_id = pb.createCollisionShape(pb.GEOM_MESH,
                                         fileName=tmp_obj.name,
                                         meshScale=scale)
        vis_id = pb.createVisualShape(pb.GEOM_MESH,
                                      fileName=tmp_obj.name,
                                      meshScale=scale)
        self.body_id = pb.createMultiBody(baseMass=0,
                                          baseCollisionShapeIndex=col_id,
                                          baseVisualShapeIndex=vis_id)
        self.position = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.1)]
        # self.position = [0, 0, 0]
        
        # Random rotation with small angles for variation
        # euler_x = np.random.uniform(-0.2, 0.2)  # ±11.5 degrees
        # euler_y = np.random.uniform(-0.2, 0.2)  # ±11.5 degrees  
        # euler_z = np.random.uniform(-0.2, 0.2)  # ±11.5 degrees
        # self.rotation = pb.getQuaternionFromEuler([euler_x, euler_y, euler_z])  # Quaternion (x,y,z,w)
        self.rotation = pb.getQuaternionFromEuler([0, 0, 0])  # TODO: No rotation
        
        pb.resetBasePositionAndOrientation(self.body_id, self.position, self.rotation)

        # ── Z-up → Y-up rotation (90°-X + 90°-Z) ─────────────────────
        #q_z2y = pb.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2])
        #pb.resetBasePositionAndOrientation(self.body_id,
        #                                   [0, 0, 0], q_z2y)

        self._tmp_obj = tmp_obj                  # handle

        self.anchors = 4
        self.r_min = 0.4
        self.r_max = 1.3
        self.n_anchors: int = 4
        self.phi_min = math.radians(30)
        self.phi_max = math.radians(75)
        self.anchors = None
        self.target_anchors = None
        self.trajectory = None
    # ──────────────────────────────────────────────────────────────
    def _random_spherical(self) -> Tuple[float, float, float]:
        """랜덤 θ, φ, r 샘플링"""
        theta = np.random.uniform(0, 2*math.pi)
        phi   = np.random.uniform(self.phi_min, self.phi_max)
        r     = np.random.uniform(self.r_min, self.r_max)
        return theta, phi, r

    def _shortest_delta(self, a_from: float, a_to: float) -> float:
        """
        θ 보간 시, 2π 경계를 넘어가지 않도록 최단 δ 계산
        """
        raw = a_to - a_from
        return (raw + math.pi) % (2*math.pi) - math.pi

    def _build_trajectory(self) -> List[Tuple[float,float,float,Tuple[float,float,float]]]:
        traj = []
        # 한 세그먼트당 뷰 개수
        seg_len = self.n_views // (self.n_anchors - 1)
        for i in range(self.n_anchors - 1):
            θ0, φ0, r0 = self.anchors[i]
            θ1, φ1, r1 = self.anchors[i+1]
            # Target anchors for smooth interpolation
            target0 = self.target_anchors[i]
            target1 = self.target_anchors[i+1]
            for t_step in range(seg_len):
                t = t_step / seg_len
                θ = θ0 + self._shortest_delta(θ0, θ1) * t
                φ = φ0 + (φ1 - φ0) * t
                r = r0 + (r1 - r0) * t
                # Smooth target interpolation
                target = [
                    target0[0] + (target1[0] - target0[0]) * t,
                    target0[1] + (target1[1] - target0[1]) * t,
                    target0[2] + (target1[2] - target0[2]) * t
                ]
                traj.append((θ, φ, r, tuple(target)))
        # 남는 뷰가 있으면 마지막 앵커 위치 반복
        while len(traj) < self.n_views:
            last_anchor = self.anchors[-1]
            last_target = self.target_anchors[-1]
            traj.append((last_anchor[0], last_anchor[1], last_anchor[2], last_target))
        return traj

    # ──────────────────────────────────────────────────────────────
    def _random_target(self) -> Tuple[float, float, float]:
        """랜덤 타겟 위치 생성"""
        # return (0, 0, 0)
        return (np.random.uniform(-0.1, 0.1), 
                np.random.uniform(-0.1, 0.1), 
                np.random.uniform(-0.1, 0.1))

    def __iter__(self):
        fov = np.random.uniform(50, 80)
        up     = [0, 0, 1]
        self.anchors = [self._random_spherical() 
                        for _ in range(self.n_anchors)]
        self.target_anchors = [self._random_target() 
                              for _ in range(self.n_anchors)]
        self.trajectory = self._build_trajectory()
        for θ, φ, r, target in self.trajectory:
            # 구면→직교 변환
            sinφ = math.sin(φ)
            cam_x = r * sinφ * math.cos(θ)
            cam_y = r * sinφ * math.sin(θ)
            cam_z = r * math.cos(φ)
            cam_pos = [cam_x, cam_y, cam_z]

            # target을 trajectory에서 받아온 값 사용

            # View, Projection
            view = pb.computeViewMatrix(cam_pos, target, up)
            proj = pb.computeProjectionMatrixFOV(fov=fov,
                                                 aspect=1.0,
                                                 nearVal=self.near,
                                                 farVal=self.far)

            w = h = self.img_wh
            flag_no_seg = getattr(pb, "ER_NO_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX", 0)
            rgba, depth_buf, *_ = pb.getCameraImage(
                    w, h, view, proj,
                    renderer=pb.ER_TINY_RENDERER,
                    flags=flag_no_seg)[2:]                                  # Get RGBA, Depth by TinyRenderer
            rgb   = np.reshape(rgba, (h, w, 4))[..., :3].astype(np.uint8)
            depth = np.reshape(depth_buf, (h, w)).astype(np.float32)
            depth_m = (self.far * self.near) / (self.far -
                       (self.far - self.near) * depth)  # OpenGL→m

            # Intrinsic (k: fov 60°)
            fx = fy = 0.5 * w / math.tan(0.5 * fov * math.pi / 180)
            cx = cy = 0.5 * w
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float32)

            # ViewMatrix(4×4) → T_camW
            view44 = np.array(view, dtype=np.float32).reshape(4, 4).T
            R = view44[:3, :3].T
            t = -R @ view44[:3, 3]
            T_camW = np.eye(4, dtype=np.float32)
            T_camW[:3, :3], T_camW[:3, 3] = R, t

            yield rgb, depth_m, K, T_camW

        pb.disconnect()
        os.remove(self._tmp_obj.name)


# ────────────────────────────────────────────────────────────────────────────
# 3) Generate PLY · bbox.json · lines.json

def mesh_bbox_line(glb_path: str, out_dir: pathlib.Path, position: Optional[List[float]] = [0, 0, 0], quat: Optional[List[float]] = [0, 0, 0, 1]) -> None:
    """GLB -> (PLY, bbox.json, lines.json)"""
    scene: trimesh.Scene = trimesh.load(glb_path, force='scene', process=False)

    # # 3-1) PLY  (Exclude Path3D, only Trimesh)
    # mesh_list = [g for g in scene.geometry.values()
    #              if isinstance(g, trimesh.Trimesh)]
    # concat = trimesh.util.concatenate(mesh_list)
    # concat.apply_transform(T_Z2Y)
    # concat.apply_scale(1 / 1000.0)
    # concat.export(out_dir / f"{out_dir.name}_vh_clean.ply")

    # # 3-2) bbox.json  (Each Trimesh OBB)
    # bboxes = []
    # for idx, g in enumerate(mesh_list):
    #     obb = g.bounding_box_oriented.copy()
    #     obb.apply_transform(T_Z2Y)
    #     center = (obb.centroid / 1000.).tolist()
    #     size   = (obb.primitive.extents / 1000.).tolist()
    #     quat   = trimesh.transformations.quaternion_from_matrix(
    #                obb.primitive.transform).tolist()  # [w,x,y,z]
    #     bboxes.append(dict(id=idx, class_id=0,
    #                        center=center, size=size, heading=quat))
    # (out_dir / 'bbox.json').write_text(json.dumps(bboxes, indent=2))

    # (1) class‑id Mapping --------------------------------------------------------
    CLASS_MAP = {
        "CBP": 0,  # Circular‑Base‑Plate
        "SBP": 1,  # Square‑Base‑Plate
        "HBP": 2,  # H‑beam‑Base‑Plate
        "ABP": 3,  # Angle‑Base‑Plate
        "ChBP": 4, # Channel‑Base‑Plate
        "CP":  5,  # Circular‑Plate
        "SP":  6,  # Square‑Plate
        "HP":  7,  # H‑beam‑Plate
        "AP":  8,  # Angle‑Plate
        "ChP": 9,  # Channel‑Plate
    }
    stem = pathlib.Path(glb_path).stem          # ex) ChBP_254
    class_id = 0                                # default
    for key, cid in CLASS_MAP.items():
        if key in stem:
            class_id = cid
            break

    # 3‑1) PLY  (Exclude Path3D, only Trimesh) - not used?
    IGNORE_FOR_BBOX = {"plate", "welding_line"}

    mesh_list = []
    for node in scene.graph.nodes_geometry:
        if node in IGNORE_FOR_BBOX:
            continue
        g = scene.geometry[node]
        if isinstance(g, trimesh.Trimesh):
            mesh_list.append(g)
    concat = trimesh.util.concatenate(mesh_list)
    concat.apply_transform(T_Z2Y)
    concat.apply_scale(1 / 1000.0)              # mm → m
    concat.export(out_dir / f"{out_dir.name}_vh_clean.ply")

    # 3‑2) bbox.json  (Each Trimesh OBB)
    verts = concat.vertices                     # [N,3]  (Y‑up, m (mm X))
    xyz_min = verts.min(axis=0)
    xyz_max = verts.max(axis=0)
    
    # 메타데이터에서 최대 높이 제한 적용 (있는 경우)
    if 'max_height_mm' in scene.metadata:
        max_height_m = scene.metadata['max_height_mm'] / 1000.0  # mm -> m
        # Y축(높이)을 제한 (Z-up -> Y-up 변환 후)
        xyz_max[1] = min(xyz_max[1], max_height_m)
    
    center_raw = ((xyz_min + xyz_max) / 2).tolist() # [x,y,z] in m
    position_transformed = (T_Z2Y[:3, :3] @ np.array(position)).tolist()
    center = [c + p for c, p in zip(center_raw, position_transformed)]
    size   = (xyz_max - xyz_min).tolist()       # [dx,dy,dz] in m

    R_zup = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
    R_yup = T_Z2Y[:3, :3] @ R_zup @ T_Z2Y[:3, :3].T
    quat = matrix_to_quaternion(R_yup)

    bbox = [dict(id=0, class_id=class_id,
                 center=center, size=size, heading=quat)]
    (out_dir / 'bbox.json').write_text(json.dumps(bbox, indent=2))

    # 3-3) lines.json  (First Path3D → polyline)
    from trimesh.path.path import Path3D
    path_objs = [o for o in scene.dump() if isinstance(o, Path3D)]
    if not path_objs:
        for node in scene.graph.nodes_geometry:
            g = scene.geometry[node]
            if isinstance(g, Path3D):
                path_objs.append(g)

    if path_objs:
        # Apply same transformation as bbox: Z-up → Y-up, then apply object transform
        verts_yup = trimesh.transform_points(path_objs[0].vertices, T_Z2Y) / 1000.  # mm → m, Z-up → Y-up
        
        # Apply object rotation
        R_zup = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
        R_yup = T_Z2Y[:3, :3] @ R_zup @ T_Z2Y[:3, :3].T
        verts_rotated = (R_yup @ verts_yup.T).T
        
        # Apply object position
        position_transformed = (T_Z2Y[:3, :3] @ np.array(position))
        verts_final = verts_rotated + position_transformed
        
        lines_dict = [dict(id=0, vertices=verts_final.tolist())]
        (out_dir / 'lines.json').write_text(json.dumps(lines_dict, indent=2))
    else:
        print(f"[!] Path3D(welding-line) not found in {glb_path}. "
              f"lines.json not written.")


# ────────────────────────────────────────────────────────────────────────────
# 4) Rendering Loop → color/depth/pose/intrinsic folders

def render_views(glb_path: str,
                 out_dir: pathlib.Path,
                 n_views: int = 60,
                 img_wh: int = 640,
                 near: float = 0.2,
                 far: float = 4.0) -> PyBulletRenderer:
    """Call PyBulletRenderer and save results"""
    for sub in ('color', 'depth', 'pose', 'intrinsic'):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    renderer = PyBulletRenderer(glb_path, n_views, img_wh, near, far)

    for idx, (rgb, depth, K, T_camW) in enumerate(renderer):
        fn = f"{idx:06d}"
        Image.fromarray(rgb).save(out_dir / 'color' / f"{fn}.jpg")
        # depth[m] -> depth[mm] 16-bit
        imageio.imwrite(out_dir / 'depth' / f"{fn}.png",
                        (depth * 1000).astype(np.uint16))
        np.savetxt(out_dir / 'intrinsic' / f"{fn}.txt", K, fmt="%.6f")

        # Z-up → Y-up transformation and camera coordinate fix
        # PyBullet uses right-handed coordinate system, we need to fix the camera direction
        T_camW_fixed = T_camW.copy()
        T_camW_fixed[:3, 1] *= -1  # Flip z-axis to look forward instead of backward
        T_camW_fixed[:3, 2] *= -1  # Flip z-axis to look forward instead of backward
        T_WC = T_Z2Y @ T_camW_fixed
        #np.savetxt(out_dir / 'pose' / f"{fn}.txt", T_WC, fmt="%.6f")
        pose_name = f"frame-{fn}.pose.txt"
        np.savetxt(out_dir/'pose'/pose_name, T_WC, fmt="%.6f")        
    
    return renderer

def process_one_wrapper(args_tuple):
    """Wrapper function for multiprocessing"""
    glb_info, scene_id, out_root, args = args_tuple
    
    if isinstance(glb_info, tuple):  # zip case: (zip_path, glb_name)
        zip_path, glb_name = glb_info
        tmp_path = pathlib.Path(tempfile.mkdtemp()) / pathlib.Path(glb_name).name
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                with zf.open(glb_name) as src, open(tmp_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
            process_one(str(tmp_path), scene_id, out_root, args)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    else:  # single glb case
        process_one(glb_info, scene_id, out_root, args)

def process_one(glb_path: str, scene_id: str, out_root: pathlib.Path,
                args):
    # Set unique random seed for each process to ensure different trajectories
    import random
    import time
    seed = int(time.time() * 1000000) % 2**32 + hash(scene_id) % 2**16
    np.random.seed(seed)
    random.seed(seed)
    
    out_dir = out_root / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer = render_views(glb_path, out_dir,
                                 n_views=args.n_views,
                                 img_wh=args.img_wh,
                                 near=args.near, far=args.far)

    mesh_bbox_line(glb_path, out_dir, position=renderer.position, quat=renderer.rotation)

# ────────────────────────────────────────────────────────────────────────────
# 5) CLI main

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--glb",      default=None, help="Path to the .glb file")
    ap.add_argument("--zip",                    help="Path to a zip archive containing multiple .glb files")
    ap.add_argument("--scene_id", default="scene0000_00", help="Output folder name")
    ap.add_argument("--out_root", default="data/scannet/scans",
                    help="Parent directory in which to create the scene directory")
    ap.add_argument("--n_views",  type=int, default=60,  help="Number of rendered views")
    ap.add_argument("--img_wh",   type=int, default=640, help="Image side length (pixels)")
    ap.add_argument("--near",     type=float, default=0.2, help="Near clipping distance (m)")
    ap.add_argument("--far",      type=float, default=4.0, help="Far clipping distance (m)")
    ap.add_argument("--workers",  type=int, default=cpu_count(), 
                    help=f"Number of worker processes (default: {cpu_count()})")
    args = ap.parse_args()

    root = pathlib.Path(args.out_root)
    
    # ---- (A) zip input, process multiple scenes in parallel ----
    if args.zip:
        with zipfile.ZipFile(args.zip, 'r') as zf:
            glb_names = sorted([n for n in zf.namelist()
                               if n.lower().endswith('.glb')])
            
            # Prepare arguments for parallel processing
            task_args = []
            for idx, name in enumerate(glb_names):
                scene_id = f"scene{idx:04d}_00"
                task_args.append(((args.zip, name), scene_id, root, args))
            
            # Process in parallel
            print(f"Processing {len(glb_names)} scenes using {args.workers} workers...")
            with Pool(processes=args.workers) as pool:
                list(tqdm(
                    pool.imap(process_one_wrapper, task_args),
                    total=len(task_args),
                    desc="Scenes"
                ))
    
    # ---- (B) single glb input, process one scene ----
    elif args.glb:
        process_one(args.glb, args.scene_id, root, args)
    else:
        raise ValueError(" --glb or --zip must be specified.")

if __name__ == "__main__":
    main()