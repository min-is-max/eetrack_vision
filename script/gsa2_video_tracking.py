import os
import sys

# Add Grounded-SAM-2 path manually
gsa2_path = os.path.join(os.path.dirname(__file__), "../Grounded-SAM-2")
sys.path.append(gsa2_path)

import argparse
from gsa2_video_tracker import GSA2VideoTracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video segmentation with Grounded-SAM2")
    parser.add_argument("video_dir", type=str, help="Directory to image frames.")
    parser.add_argument("--text_prompt", type=str, default="pillar. bottom plate.", help="Text prompt to input Grounded-SAM2")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory to save masks")
    parser.add_argument("--result_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to load the model.")
    args = parser.parse_args()

    video_tracker = GSA2VideoTracker(args.device)

    video_segments, id_to_objects = video_tracker.predict_video(
        args.video_dir,
        text = args.text_prompt,
        mask_dir = args.mask_dir,
        result_dir = args.result_dir,
    )
