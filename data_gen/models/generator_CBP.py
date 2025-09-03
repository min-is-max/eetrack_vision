import io
import os
import trimesh
import numpy as np
import random
import zipfile
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import trimesh.util as tu
import plotly.graph_objects  as go
from trimesh import Trimesh
from trimesh.path import entities, Path3D
from trimesh.visual import TextureVisuals
from trimesh.transformations import rotation_matrix
from trimesh.visual.texture import TextureVisuals, PBRMaterial
from trimesh.creation        import extrude_polygon
from trimesh.transformations import rotation_matrix
from shapely.geometry        import box, Point
from scipy.spatial import ConvexHull

def draw_points_order(points):
    import matplotlib.pyplot as plt
    plt.scatter(points[:,0], points[:,1])
    for i, (x_i, y_i, z_i) in enumerate(points):
        plt.text(x_i, y_i, str(i))
    plt.show()

def lerp(start: np.ndarray, end: np.ndarray, weight: np.ndarray):
    return start + weight * (end - start)

def lerp_points(points, lerp_length=0.001):
    segment_len = np.linalg.norm(np.diff(points, axis=0, append=points[:1]), axis=1)
    num_interp = np.ceil(segment_len / lerp_length).astype(np.int32)
    lerp_weight = np.concatenate([np.linspace(0, 1, num=n+1)[1:] for n in num_interp])
    vertices_lerp_start = np.repeat(points, num_interp, axis=0)
    vertices_lerp_end = np.repeat(np.concatenate([points[1:], points[:1]]), num_interp, axis=0)
    vertices_lerp = lerp(vertices_lerp_start, vertices_lerp_end, lerp_weight[:,None])

    return vertices_lerp

class CircularBodyPlateGenerator:
    def __init__(self):
        self.body_size       = float(input("판 크기 A (default 80.0): ") or 80.0)
        self.pillar_diameter = float(input("기둥 외경 (default 50.0): ") or 50.0)
        self.body_fillet      = 10.0
        self.body_thick       = 10.0
        self.pillar_rad_outer = (self.pillar_diameter + 10) / 2
        self.pillar_rad_inner = (self.pillar_diameter - 10) / 2
        self.pillar_height    = 150.0

    def generate(self, count=1, zip_filename = "Texture_CBP.zip", save_dir="assets/weld_objects/meshes"):
        # Zip 파일을 메모리에서 생성
        zip_buffer = io.BytesIO()
        save_dir = Path(__file__).parent / Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        edge_save_dir = Path(save_dir.as_posix().replace("meshes", "edges"))
        edge_save_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i in range(1, count+1):
                # 2) Mesh 생성 파트
                # 2.0) mesh_plate: 평면 사각형 (판)
                plate = box(
                    -100, -100,
                    100,  100
                )
                mesh_poly_z = extrude_polygon(plate, height=self.body_thick) # Plate 두께도 그냥 Body 두께로 설정
                mesh_plate = mesh_poly_z.copy()
                mesh_plate.apply_transform(rotation_matrix(np.radians(-90), [1, 0, 0]))
                
                # 2.1) mesh_body: 필렛 사각형 → Z축으로 Extrude → XZ 평면 위에 Y축으로 회전
                square      = box(
                    -self.body_size/2 + self.body_fillet, -self.body_size/2 + self.body_fillet,
                    self.body_size/2 - self.body_fillet,  self.body_size/2 - self.body_fillet
                )
                round_poly  = square.buffer(self.body_fillet, resolution=64)
                mesh_body_z = extrude_polygon(round_poly, height=self.body_thick)
                
                mesh_body = mesh_body_z.copy()
                mesh_body.apply_transform(rotation_matrix(np.radians(-90), [1, 0, 0]))
                mesh_body.apply_translation([0, self.body_thick, 0])
                
                # 2.2) mesh_pillar: hollow cylinder (외부 원기둥 − 내부 원기둥)
                outer_circle   = Point(0, 0).buffer(self.pillar_rad_outer, resolution=64)
                inner_circle   = Point(0, 0).buffer(self.pillar_rad_inner, resolution=64)
                annulus        = outer_circle.difference(inner_circle)
                mesh_pillar_z  = extrude_polygon(annulus, height=self.pillar_height)
                
                mesh_pillar = mesh_pillar_z.copy()
                mesh_pillar.apply_transform(rotation_matrix(np.radians(-90), [1, 0, 0]))
                mesh_pillar.apply_translation([0, self.body_thick*2, 0])
                
                # 2.3) Scene에 올리기
                scene = trimesh.Scene({
                    'plate':  mesh_plate,
                    'body':   mesh_body,
                    'pillar': mesh_pillar
                })
                
                # 3) Visualization (생략)
                
                # 4) Y축 평면 외곽 링 추출 & Welding line 생성 (mesh.body 바닥 단면)                
                # 4.1) mesh_body 참조 및 바닥 y값 찾기
                mesh_body = scene.geometry['body']
                same_y    = mesh_body.vertices[:,1].min()
                
                # 4.2) y 평면 절단 단면 vertices
                mask             = np.isclose(mesh_body.vertices[:,1], same_y, atol=1e-6)
                section_vertices = mesh_body.vertices[mask]
                
                # 4.3) 외곽 단면점: ConvexHull로 필렛된 외곽 경계 추출
                points_xz        = section_vertices[:, [0,2]]
                hull             = ConvexHull(points_xz)
                hull_indices     = hull.vertices
                outer_vertices   = section_vertices[hull_indices]
                
                # 4.4) 각도 기준 정렬 및 선분 생성
                angles  = np.arctan2(outer_vertices[:,2], outer_vertices[:,0])
                order   = np.argsort(angles)
                ordered = outer_vertices[order]
                vertices = ordered[:, [0,1,2]]
                edges    = np.column_stack([np.arange(len(vertices)), np.roll(np.arange(len(vertices)), -1)])
                lines    = vertices[edges]
                welding_line = trimesh.load_path(lines)
                
                # 4.5) Path3D 엔티티로 변환 및 씬에 추가
                n          = len(welding_line.vertices)
                segments   = np.column_stack([np.arange(n-1), np.arange(1,n)])
                line_entities = [entities.Line(tuple(seg)) for seg in segments]
                path       = Path3D(entities=line_entities, vertices=welding_line.vertices)
                new_scene  = scene.copy()
                new_scene.add_geometry(path, node_name='welding_line')
                
                # 4.6) GLB 내 PBR Material 랜덤화 (수정본)                
                def random_gray_color(min_l=0.2, max_l=0.9):
                    l = random.uniform(min_l, max_l)
                    return [l, l, l, 1.0]
                
                for name, geom in new_scene.geometry.items():
                    # Trimesh 타입이 아니면 스킵 (Path3D, Camera, Light 등 모두 건너뜀)
                    if not isinstance(geom, Trimesh):
                        continue
                
                    mesh = geom
                
                    # 1) 랜덤 PBR 파라미터
                    base_color = random_gray_color()
                    metallic   = random.uniform(0.0, 1.0)
                    roughness  = random.uniform(0.0, 1.0)
                
                    # 2) PBR 머티리얼 객체 생성
                    # PBR: Physically Based Rendering
                    pbr = PBRMaterial(
                        name=f"{name}_mat",
                        baseColorFactor=base_color,
                        metallicFactor=metallic,
                        roughnessFactor=roughness
                    )
                
                    # 3) TextureVisuals 로 교체
                    # UV Mapping: 2D 텍스처 좌표를 3D 메쉬에 적용
                    mesh.visual = TextureVisuals(
                        uv=getattr(mesh.visual, 'uv', None),
                        material=pbr
                    )
                # print(new_scene.geometry.keys())         # → ['plate','body','pillar','geometry_3']
                # print(new_scene.bounds)                  # AABB ≈ [-100,   0, -100]  ~  [100,170,100] (mm)        
                
                # GLB 파일을 메모리 버퍼에 직접 저장
                glb_buffer = io.BytesIO()
                new_scene.export(file_obj=glb_buffer, file_type='glb')
                glb_buffer.seek(0)
                    
                # Zip 파일에 직접 추가
                zipf.writestr(f"{i}.glb", glb_buffer.read())

                # Save welding line
                tf_vertices = R.from_euler('x', 90, degrees=True).apply(vertices)
                tf_vertices /= 1000
                tf_vertices_lerp = lerp_points(tf_vertices)
                np.save(edge_save_dir / f"{i}", tf_vertices_lerp)
                # Save mesh
                new_scene.delete_geometry("geometry_3")
                y2zup = np.eye(4); y2zup[:3,:3] = R.from_euler('x', 90, degrees=True).as_matrix()
                for geom in new_scene.geometry.values():
                    geom.apply_transform(y2zup)
                    geom.apply_scale(0.001)
                save_path = save_dir / f"{i}.obj"
                new_scene.export(save_path, mtl_name=f"{i}.mtl")
        
        # 메모리 버퍼를 파일로 저장
        # 파일이 이미 존재하면 새로운 이름 생성
        counter = 1
        while os.path.exists(zip_filename):
            zip_filename = f"{zip_filename}({counter}).zip"
            counter += 1
        
        with open(zip_filename, 'wb') as f:
            f.write(zip_buffer.getvalue())
        
        print(f'✅ Created {zip_filename} directly!')
        return zip_filename

""" if __name__ == "__main__":
        generator = CircularBodyPlateGenerator()
        generator.generate(1)
"""


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 0) 생성할 GLB 수 입력 - args 받게
    # glb_count = int(input("생성할 GLB 파일 개수 (default 1): ") or 1)
    parser.add_argument('--glb_count', type=int, default=1, help='Number of GLB files to generate')

    # # 1) 제너레이터 생성 (몸체·기둥 파라미터는 그대로 input 받음)
    # generator = CircularBodyPlateGenerator()
    # zip_filename = input("생성할 ZIP 파일 이름 (default Texture_CBP.zip): ") or "Texture_CBP.zip"
    parser.add_argument('--zip_filename', type=str, default='Texture_CBP.zip', help='Name of the output ZIP file')
    parser.add_argument('--save_dir', type=str, default='../assets/weld_objects/meshes/CBP', help='Directory to save')
    args = parser.parse_args()
    generator = CircularBodyPlateGenerator()
    glb_count = args.glb_count
    zip_filename = args.zip_filename
    save_dir = args.save_dir

    # 2) 원하는 개수만큼 GLB 생성 후 ZIP
    generator.generate(glb_count, zip_filename, save_dir)
