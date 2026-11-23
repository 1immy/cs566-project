"""Build small Blender scenes with a camera panning at a known, constant pixel speed (issue #6).

Run with Blender, not Python (Blender must be on PATH):
  blender -b --factory-startup -P scripts/blender/make_pan_scenes.py -- --out data/blender --speeds 0 2 5 10 20 40

For each speed it writes <out>/pan_v<speed>.blend: an emission-only textured plane (no lighting
noise, so renders are deterministic) seen by an orthographic camera that moves along x by
`speed` pixels between the first and the last frame. The last frame is the ground-truth pose,
as in the real dataset. Render it with scripts/blender/render_frames.py (visionsim's renderer
stops after one frame on Blender 5.3, so we render directly).

STATUS: written against the Blender 5.x Python API but not yet run in Blender. If a call fails,
the traceback names the line; send it back rather than working around it.
"""
import argparse
import os
import sys

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/blender")
    parser.add_argument("--speeds", nargs="+", type=float, default=[0, 2, 5, 10, 20, 40],
                        help="total camera pan in pixels over the whole animation")
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--res", type=int, default=128, help="square render size in pixels")
    parser.add_argument("--view-units", type=float, default=4.0,
                        help="world units across the image, so one pixel is view-units / res")
    return parser.parse_args(argv)


def make_material():
    material = bpy.data.materials.new("PanTexture")
    try:
        material.use_nodes = True      # deprecated and always on in Blender 5, harmless there
    except Exception:
        pass
    tree = material.node_tree
    tree.nodes.clear()
    coords = tree.nodes.new("ShaderNodeTexCoord")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 8.0       # about 4 px features at 128 px across 4 units
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.55
    ramp = tree.nodes.new("ShaderNodeValToRGB")        # stretch the noise so edges have contrast
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.65
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(coords.outputs["Object"], noise.inputs["Vector"])
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def set_linear(obj):
    """Linear interpolation on every keyframe, for both the old and the layered action API."""
    action = obj.animation_data.action
    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:
        fcurves = [fc for layer in action.layers for strip in layer.strips
                   for bag in strip.channelbags for fc in bag.fcurves]
    for fcurve in fcurves:
        for point in fcurve.keyframe_points:
            point.interpolation = "LINEAR"


def build(path, speed_px, args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False
    scene.view_settings.view_transform = "Standard"      # no tone-mapping curve on the frames
    scene.frame_start, scene.frame_end = 1, args.frames

    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0, 0, 0))
    bpy.context.object.data.materials.append(make_material())

    camera_data = bpy.data.cameras.new("PanCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = args.view_units
    camera = bpy.data.objects.new("PanCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    units_per_px = args.view_units / args.res
    camera.location = (0.0, 0.0, 5.0)
    camera.keyframe_insert("location", frame=1)
    camera.location = (speed_px * units_per_px, 0.0, 5.0)
    camera.keyframe_insert("location", frame=args.frames)
    set_linear(camera)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))
    print(f"wrote {path} (pan {speed_px:g} px over {args.frames} frames)")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    for speed in args.speeds:
        build(os.path.join(args.out, f"pan_v{speed:g}.blend"), speed, args)


main()
