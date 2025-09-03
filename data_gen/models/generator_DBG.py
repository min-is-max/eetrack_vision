import argparse
import trimesh
import numpy as np
import zipfile
import os


def generate_debug_boxes(count: int, zip_filename: str, size: tuple):
    """
    Generate `count` number of debug box GLB files of given `size` and
    pack them into a ZIP archive named `zip_filename`, overwriting any existing file.

    Parameters:
    - count: number of GLB files to generate
    - zip_filename: desired name for the output ZIP file
    - size: (width, depth, height) of the box in millimeters
    """
    # Create a box mesh with specified extents
    box_mesh = trimesh.creation.box(extents=size)
    scene = trimesh.Scene()
    scene.add_geometry(box_mesh, geom_name='debug_box')

    # Export GLB data once
    glb_bytes = scene.export(file_type='glb')

    # Ensure filename ends with .zip
    zip_filename = zip_filename if zip_filename.lower().endswith('.zip') else f"{zip_filename}.zip"

    # Write ZIP file (overwrite if exists)
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i in range(1, count + 1):
            zipf.writestr(f"{i}.glb", glb_bytes)

    print(f"✅ Overwritten {zip_filename} with {count} GLB files of size {size} mm.")
    return zip_filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate debug box GLBs and pack into ZIP.'
    )
    parser.add_argument(
        '--glb_count', type=int, default=1,
        help='Number of GLB files to generate'
    )
    parser.add_argument(
        '--zip_filename', type=str, default='debug_box.zip',
        help='Name of the output ZIP file (overwritten)'
    )
    parser.add_argument(
        '--width', type=float, default=100.0,
        help='Box width along X-axis (mm)'
    )
    parser.add_argument(
        '--depth', type=float, default=200.0,
        help='Box depth along Y-axis (mm)'
    )
    parser.add_argument(
        '--height', type=float, default=300.0,
        help='Box height along Z-axis (mm)'
    )
    args = parser.parse_args()

    size = (args.width, args.depth, args.height)
    generate_debug_boxes(
        count=args.glb_count,
        zip_filename=args.zip_filename,
        size=size
    )
