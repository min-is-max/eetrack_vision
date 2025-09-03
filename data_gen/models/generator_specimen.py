import os
from pathlib import Path
import argparse
import numpy as np
import torch
import trimesh
import isaaclab.utils.math as math_utils


def interpolate_points_from_path3d(path3d, resolution=0.001):
    points = []
    for e in path3d.entities:
        ps, pe = e.discrete(path3d.vertices)
        dist = np.linalg.norm(pe - ps, axis=-1)
        num = int(np.round(dist / resolution))
        u = np.linspace(0, 1, num+1)[1:-1,None]
        inter_points = (1 - u) * ps[None] + u * pe
        points.append(inter_points)

    return np.concatenate(points)


def build_scene(single_mesh: trimesh.Trimesh, T: np.ndarray, debug=False):
    plate = single_mesh.copy()
    pillar = single_mesh.copy().apply_transform(T)

    z_min = plate.bounds[:,2].min()
    plate.apply_translation([0,0,-z_min])
    pillar.apply_translation([0,0,-z_min])

    scene = trimesh.Scene()
    scene.add_geometry(plate, geom_name="plate")
    scene.add_geometry(pillar, geom_name="pillar")

    plate_upper_plane_normal = [0, 0, 1]
    plate_upper_plane_origin = [0, 0, plate.vertices[:,2].max()]

    rect = pillar.section(plate_upper_plane_normal, plate_upper_plane_origin)
    vertices = rect.vertices
    edges = np.stack([rect.entities[0].points[:-1], rect.entities[0].points[1:]], axis=-1)

    edge_dists = np.linalg.norm(np.diff(vertices[edges], axis=1)[:,0], axis=-1)
    long_edge_pairs = edges[edge_dists > np.mean(edge_dists)]
    entities = [trimesh.path.entities.Line(e) for e in long_edge_pairs]

    lines = trimesh.path.Path3D(entities, vertices=vertices)
    weld_edges = interpolate_points_from_path3d(lines)

    if debug:
        debug_scene = scene.copy()
        colors = np.full((weld_edges.shape[0], 4), [255, 0, 0, 255], dtype=np.uint8)
        debug_scene.add_geometry(trimesh.PointCloud(weld_edges, colors))
        debug_scene.show()

    return scene, weld_edges


def generate(single_obj_path, count, pose_range, save_dir, seed=42, debug=False):
    np.random.seed(seed)

    single_mesh = trimesh.load(single_obj_path)

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (count, 6), device="cpu")
    rand_T = torch.eye(4).repeat(count, 1, 1)
    rand_T[:,:3,3] = rand_samples[:,:3]
    rand_T[:,:3,:3] = math_utils.matrix_from_euler(rand_samples[:,3:], convention="ZYX")
    rand_T = rand_T.numpy()

    mesh_save_dir = Path(__file__).parent / Path(save_dir)
    mesh_save_dir.mkdir(parents=True, exist_ok=True)
    edge_save_dir = Path(mesh_save_dir.as_posix().replace("meshes", "edges"))
    edge_save_dir.mkdir(parents=True, exist_ok=True)

    for i, T in enumerate(rand_T):
        scene, edges = build_scene(single_mesh, T, debug=debug)

        if not debug:
            mesh_save_path = mesh_save_dir / f"{i}.obj"
            scene.export(mesh_save_path, mtl_name=f"{i}.mtl")
            edge_save_path = edge_save_dir / f"{i}.npy"
            np.save(edge_save_path, edges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 0) 생성할 GLB 수 입력 - args 받게
    # glb_count = int(input("생성할 GLB 파일 개수 (default 1): ") or 1)
    parser.add_argument('--single_specimen_obj_path', type=str, default=f"{os.path.dirname(__file__)}/../assets/weld_objects/meshes/specimen/single.obj", help='Path to single specimen obj file')
    parser.add_argument('--count', type=int, default=1, help='Number of files to generate')
    parser.add_argument('--x_range', type=list, default=[0.0, 0.02], help="X range for upper specimen pose randomization")
    parser.add_argument('--y_range', type=list, default=[0.0, 0.0], help="Y range for upper specimen pose randomization")
    parser.add_argument('--z_range', type=list, default=[0.0, 0.0], help="Z range for upper specimen pose randomization")
    parser.add_argument('--roll_range', type=list, default=[0.0, 0.0], help="Roll deg range for upper specimen pose randomization")
    parser.add_argument('--pitch_range', type=list, default=[0.0, 0.0], help="Pitch deg range for upper specimen pose randomization")
    parser.add_argument('--yaw_range', type=list, default=[-10.0, 10.0], help="Yaw deg range for upper specimen pose randomization")
    parser.add_argument('--save_dir', type=str, default='../assets/weld_objects/meshes/specimen', help='Directory to save')
    parser.add_argument('--debug', action="store_true", default=False, help='Debug or not')
    args = parser.parse_args()
    single_obj_path = args.single_specimen_obj_path
    count = args.count
    pose_range = {
        'x': args.x_range,
        'y': args.y_range,
        'z': [z + 0.0525 for z in args.z_range], # Add default range
        'roll': [np.radians(roll) for roll in args.roll_range],
        'pitch': [np.radians(pitch + 90) for pitch in args.pitch_range],
        'yaw': [np.radians(yaw) for yaw in args.yaw_range],
    }
    save_dir = args.save_dir
    debug = args.debug

    # 2) 원하는 개수만큼 GLB 생성 후 ZIP
    generate(single_obj_path, count, pose_range, save_dir, debug=debug)
