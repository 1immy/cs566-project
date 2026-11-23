"""Render every frame of a pan scene to linear 16-bit grayscale PNGs (issue #6).

Run with Blender, not Python:
  blender -b data/blender/pan_v10.blend -P scripts/blender/render_frames.py -- --out data/blender/frames_v10
  (add --frame-end 8 for a quick test)

Writes <out>/f_0001.png ... f_1024.png. The 'Raw' view transform stores scene-linear values
without a display curve, so pixel / 65535 is the radiance to hand to the burst simulator.
"""
import argparse
import os
import sys

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    scene = bpy.context.scene
    if args.frame_start is not None:
        scene.frame_start = args.frame_start
    if args.frame_end is not None:
        scene.frame_end = args.frame_end
    scene.render.use_motion_blur = False
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    try:
        scene.view_settings.view_transform = "Raw"
    except TypeError:
        print("WARNING: no Raw view transform; frames will be display-encoded, not linear")
    scene.view_settings.look = "None"
    image = scene.render.image_settings
    image.file_format = "PNG"
    image.color_mode = "BW"
    image.color_depth = "16"
    os.makedirs(args.out, exist_ok=True)
    scene.render.filepath = os.path.join(os.path.abspath(args.out), "f_")
    bpy.ops.render.render(animation=True)
    print(f"rendered frames {scene.frame_start}..{scene.frame_end} to {args.out}")


main()
