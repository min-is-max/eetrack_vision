import numpy as np
import random
import os
import io
import zipfile
from pathlib import Path

import trimesh
from trimesh import Trimesh
from trimesh.creation import extrude_polygon
from trimesh.path import entities, Path3D
from trimesh.visual import TextureVisuals
from trimesh.visual.texture import TextureVisuals, PBRMaterial
from trimesh.transformations import rotation_matrix
import trimesh.util as tu

from shapely.geometry import Polygon, box, Point
from shapely.affinity import translate

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

def generate(count=1, zip_filename='Texture_CBP.zip', save_dir="assets/weld_objects/meshes"):
    # ───────────────────────────────────────────────────────────────────
    # 1) 사용자 입력 파라미터
    # A = float(input("다리 길이 A = B (default 30.0): ") or 30.0)
    # t = float(input("두께 t (default 4.0): ") or 4.0)
    # r1 = float(input("convex fillet 반지름 r1 (default 6.5): ") or 6.5)
    # r2 = float(input("concave fillet 반지름 r2 (default 3.0): ") or 3.0)
    # pillar_height = float(input("기둥 높이 (default 100.0): ") or 100.0)
    # body_size = float(input("판 크기 A (default 50.0): ") or 50.0)
    A = 30.0
    t = 4.0
    r1 = 6.5
    r2 = 3.0
    body_size = 50.0
    NAME = "Angle_Body_Plate"

    # Plate 설정 (원하는 크기로 조정 가능)
    plate_thick = 10.0            # 판 두께
    plate_size  = 80.0           # 절반 길이 (→ 전체 가로/세로 = 2*plate_size)
    body_fillet      = 10.0
    body_thick       = 10.0

    # ───────────────────────────────────────────────────────────────────
    # 2) Plate (바닥판) 생성
    def generate_scene():
        pillar_height = np.random.uniform(50, 500)
        plate_2d = box(-plate_size * np.random.uniform(0.7, 8.3), -plate_size * np.random.uniform(0.7, 8.3),
                        plate_size * np.random.uniform(0.7, 8.3),  plate_size * np.random.uniform(0.7, 8.3))
        # plate_2d = box(-plate_size, -plate_size, plate_size, plate_size)
        mesh_plate_z = extrude_polygon(plate_2d, height=plate_thick)
        mesh_plate = mesh_plate_z.copy()

        # 2.1) Body (바닥 기둥) 생성

        square      = box(
            -body_size/2 + body_fillet, -body_size/2 + body_fillet,
            body_size/2 - body_fillet,  body_size/2 - body_fillet
        )

        # round_poly  = square.buffer(body_fillet, resolution=64)
        # mesh_body_z = extrude_polygon(round_poly, height=body_thick)
        # mesh_body = mesh_body_z.copy()
        # mesh_body.apply_transform(rotation_matrix(np.radians(-90), [1, 0, 0]))
        # mesh_body.apply_translation([0, plate_thick, 0])

        # 2.2) Triangle (바닥 기둥) 생성
        triangle_verts = [
            (-body_size/2 + body_fillet, -body_size/2 + body_fillet),
            (-body_size/2 + body_fillet, body_size/2 - body_fillet),
            ( body_size/2 - body_fillet, -body_size/2 + body_fillet)
        ]
        triangle_poly = Polygon(triangle_verts)
        round_triangle_poly  = triangle_poly.buffer(body_fillet, resolution=64)
        mesh_triangle_z = extrude_polygon(round_triangle_poly, height=body_thick)
        mesh_body = mesh_triangle_z.copy()
        mesh_body.apply_translation([0, 0, plate_thick])

        # ───────────────────────────────────────────────────────────────────
        # 3) 2D L자형 폴리곤 생성 & 필렛 처리
        # 기본 L자형 (CCW)
        verts = [(0, 0), (A, 0), (A, t), (t, t), (t, A), (0, A)]
        poly  = Polygon(verts)

        # concave fillet: 두 개의 코너 컷
        sq1 = box(A - r2, t - r2, A, t)
        sq2 = box(t - r2, A - r2, t, A)
        poly = poly.difference(sq1).difference(sq2)

        # concave fillet (사분원) 추가
        qc1 = Point(A - r2, t - r2).buffer(r2, resolution=32) \
            .intersection(box(A - r2, 0, A, t))
        qc2 = Point(t - r2, A - r2).buffer(r2, resolution=32) \
            .intersection(box(0, A - r2, t, A))
        poly = poly.union(qc1).union(qc2)

        # convex fillet: (t,t) 코너에 r1 필렛
        sq3 = box(t, t, t + r1, t + r1)
        qc3 = Point(t + r1, t + r1).buffer(r1, resolution=32) \
            .intersection(box(t, t, t + r1, t + r1))
        poly = poly.union(sq3).difference(qc3)

        # poly 전체를 (0, 0) 에서 (body_size-10, body_size-10) 로 이동
        poly = translate(poly, xoff=-body_size/2 + body_fillet, yoff=-body_size/2 + body_fillet)

        # ───────────────────────────────────────────────────────────────────
        # 4) 3D L자 기둥으로 Extrude & 위치 조정
        mesh_pillar_z = extrude_polygon(poly, height=pillar_height)
        # Plate 위에 얹히도록 Z축으로 이동
        mesh_pillar = mesh_pillar_z.copy()
        mesh_pillar.apply_translation([0, 0, plate_thick + body_thick])

        # 1) mesh_after, same_y 정의
        mesh_after = mesh_body.copy()
        same_z = mesh_after.vertices[:,2].min()

        # 2) 메시와 평면의 교차선(Section) 계산
        #    plane_origin는 평면 위의 한 점, plane_normal은 법선 벡터
        section = mesh_after.section(
            plane_origin=[0, 0, same_z],
            plane_normal=[0, 0, 1]
        )

        if section is None:
            raise RuntimeError("교차선을 찾을 수 없습니다. 메시가 y=%.6f 평면을 통과하지 않거나, 너무 얇게 접합니다." % same_z)

        # 3) 순서가 보장된 선분 리스트(discrete) 중 가장 긴 루프 선택
        loops = section.discrete   # list of (n_i × 3) np.ndarray
        # 루프들 중 점 수가 가장 많은 것을 welding line으로
        welding_loop = max(loops, key=lambda pts: pts.shape[0])

        # 4) x, y, z 배열로 분리
        x_line, y_line, z_line = welding_loop[:,0], welding_loop[:,1], welding_loop[:,2]

        # 4.5) Path3D 엔티티로 변환 및 씬에 추가
        n          = len(welding_loop)
        segments   = np.column_stack([np.arange(n-1), np.arange(1,n)])
        line_entities = [entities.Line(tuple(seg)) for seg in segments]
        path       = Path3D(entities=line_entities, vertices=welding_loop)
        
        # Scene 생성 시 명시적 이름 부여
        scene = trimesh.Scene()
        scene.add_geometry(mesh_plate, geom_name='plate')
        scene.add_geometry(mesh_body, geom_name='body') 
        scene.add_geometry(mesh_pillar, geom_name='pillar')
        scene.add_geometry(path, geom_name='welding_line')
        
        # # 최대 높이를 메타데이터에 저장 (bbox 계산 시 사용)
        # max_height = plate_thick + body_thick + PILLAR_LIMIT
        # scene.metadata['max_height_mm'] = max_height
        
        return scene




    # Zip 파일을 메모리에서 생성
    zip_buffer = io.BytesIO()
    save_dir = Path(__file__).parent / Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    edge_save_dir = Path(save_dir.as_posix().replace("meshes", "edges"))
    edge_save_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i in range(1, count+1):
            new_scene  = generate_scene()
            
            # 4.6) GLB 내 PBR Material 랜덤화 (수정본)                
            def random_gray_color(min_l=0.2, max_l=0.9):
                l = random.uniform(min_l, max_l)
                # l = {"body": 0.3, "pillar": 0.5, "plate": 0.7}.get(p, 0.5)

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

            path = new_scene.geometry["welding_line"]
            ordered_vertices = np.stack(
                [path.vertices[e.points[0]] for e in path.entities]
            )
            tf_vertices = ordered_vertices / 1000
            tf_vertices_lerp = lerp_points(tf_vertices)

            np.save(edge_save_dir / f"{i}", tf_vertices_lerp)
            new_scene.delete_geometry("welding_line")
            for geom in new_scene.geometry.values():
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
    parser.add_argument('--save_dir', type=str, default='../assets/weld_objects/meshes/ABP', help='Directory to save')
    args = parser.parse_args()
    glb_count = args.glb_count
    zip_filename = args.zip_filename
    save_dir = args.save_dir

    # 2) 원하는 개수만큼 GLB 생성 후 ZIP
    generate(glb_count, zip_filename, save_dir)
