"""Run Track 4 evaluation from the repository root.

Examples:
  py scripts/run_evaluation.py --samples data/samples/kitchen1_000000.npz data/samples/kitchen1_000001.npz
  py scripts/run_evaluation.py --scene-dir data/train/kitchen1 --bursts 0 1 10 --crop 250 550 248 552 --frames 1024 --reference last

Reference 'last' reproduces the mid-term report's time-reversed burst
convention. Every reconstruction uses the same fixed gain (default 2.7).
"""
import argparse
import csv
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.burst_io import load_burst, load_gt, load_sample, naive_average
from src.baseline import aligned_average
from src.taa import TAAConfig, reconstruct_taa
from src.evaluation import (DEFAULT_EVALUATION_GAIN, apply_intensity_rule,
                            evaluate_channels)


def _map_intensity(image, gain):
    return apply_intensity_rule(image, gain)


def _rgb_png(image):
    return Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8), 'RGB')


def _figure(images, output):
    width, height = images[0][1].shape[1], images[0][1].shape[0]
    scale = max(1, 280 // width)
    gap, label_h = 12, 30
    canvas = Image.new('RGB', ((width * scale + gap) * len(images) + gap,
                               height * scale + 2 * gap + label_h), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, array) in enumerate(images):
        x = gap + i * (width * scale + gap)
        draw.text((x, gap), label, fill='black')
        canvas.paste(_rgb_png(array).resize((width * scale, height * scale)),
                     (x, gap + label_h))
    canvas.save(output)


def _custom_pipeline(spec, cube, group_size):
    module_name, func_name = spec.split(':', 1)
    function = getattr(importlib.import_module(module_name), func_name)
    result = function(cube, group_size=group_size)
    if hasattr(result, 'reconstruction'):
        return result.reconstruction
    if isinstance(result, tuple):
        return result[0]
    return result


def _select_cube(cube, frames=None, crop=None, reference='first'):
    """Select frames/crop before reconstruction; reverse time for last ref."""
    if frames is not None:
        if frames > len(cube):
            raise ValueError(f'requested {frames} frames, only {len(cube)} available')
        cube = cube[:frames]
    if crop is not None:
        y0, y1, x0, x1 = crop
        if not (0 <= y0 < y1 <= cube.shape[1] and 0 <= x0 < x1 <= cube.shape[2]):
            raise ValueError(f'crop {crop} is outside the {cube.shape[1:3]} image')
        cube = cube[:, y0:y1, x0:x1]
    if reference == 'last':
        cube = cube[::-1]
    return cube


def _reconstruct(cube, group_size, extras, report_table=False):
    if report_table:
        # Table 1 compared the first and last reference and a clamp-only TAA.
        # The original last-reference convention runs the same code with time reversed.
        original = cube
        reversed_cube = original[::-1]
        return {
            'naive': naive_average(original),
            'aligned_first': aligned_average(original, group_size)[0],
            'aligned_last': aligned_average(reversed_cube, group_size)[0],
            'clamp_only_last': reconstruct_taa(
                reversed_cube, group_size, TAAConfig(clamp=True, adaptive=False)
            ).reconstruction,
            'taa_last': reconstruct_taa(
                reversed_cube, group_size, TAAConfig()
            ).reconstruction,
        }

    methods = {'naive': naive_average(cube),
               'aligned': aligned_average(cube, group_size, reference="first")[0],
               'taa': reconstruct_taa(cube, group_size, TAAConfig()).reconstruction}
    for spec in extras:
        label = spec.replace(':', '_')
        if label in methods:
            raise ValueError(f'duplicate method name: {label}')
        methods[label] = _custom_pipeline(spec, cube, group_size)
    return methods


def _sample_images(path, group_size, extras, frames=None, crop=None, reference='first', report_table=False):
    cube = load_sample(path)
    gt_path = path.with_name(path.stem + '_gt.png')
    with Image.open(gt_path) as image:
        gt = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
    cube = _select_cube(cube, frames, crop, 'first' if report_table else reference)
    if crop is not None:
        y0, y1, x0, x1 = crop
        gt = gt[y0:y1, x0:x1]
    return gt, _reconstruct(cube, group_size, extras, report_table), len(cube)


def _full_images(scene_dir, index, group_size, extras, frames, crop, reference, report_table=False):
    # The bit-packed loader crops before unpacking, avoiding a ~2 GB full cube.
    cube = load_burst(scene_dir, index=index, frames=frames, crop=crop)
    if frames is not None and len(cube) != frames:
        raise ValueError(f'burst {index}: requested {frames} frames, got {len(cube)}')
    if not len(cube) or min(cube.shape[1:3]) < 2:
        raise ValueError(f'burst {index}: empty or too-small cube')
    if reference == 'last' and not report_table:
        cube = cube[::-1]
    gt = load_gt(scene_dir, index=index, crop=crop)
    return gt, _reconstruct(cube, group_size, extras, report_table), len(cube)


def _archive_images(path):
    with np.load(path) as archive:
        gt = archive['ground_truth'].copy()
        methods = {name: archive[name].copy() for name in ('naive', 'aligned', 'taa')
                   if name in archive}
    if len(methods) < 3:
        raise ValueError(f'{path} must contain naive, aligned, and taa arrays')
    return gt, methods, None


def _check_crop(crop, parser, packed=False):
    if crop is None:
        return None
    y0, y1, x0, x1 = crop
    if not (0 <= y0 < y1 and 0 <= x0 < x1 and y1 - y0 >= 2 and x1 - x0 >= 2):
        parser.error('--crop expects y0 y1 x0 x1 with positive bounds and dimensions >= 2')
    if packed and (x0 % 8 or x1 % 8):
        parser.error('full-burst --crop x0 and x1 must be multiples of 8 (bit-packed width)')
    if packed and (y1 > 800 or x1 > 800):
        parser.error('full-burst crop must be within the 800x800 image')
    return tuple(crop)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--samples', nargs='+', type=Path)
    source.add_argument('--archives', nargs='+', type=Path)
    source.add_argument('--scene-dir', type=Path, help='full kitchen1 folder containing 000000.npy/.png, etc.')
    parser.add_argument('--bursts', nargs='+', type=int, default=[0, 1, 10],
                        help='indices in --scene-dir; default: 0 1 10')
    parser.add_argument('--crop', nargs=4, metavar=('Y0', 'Y1', 'X0', 'X1'), type=int,
                        help='pixel bounds, exclusive upper end; full-burst X bounds multiples of 8')
    parser.add_argument('--frames', type=int, help='use the first N frames (before reversing)')
    parser.add_argument('--report-table', action='store_true',
                        help='compute all five original Table 1 method variants with PSNR and SSIM')
    parser.add_argument('--reference', choices=('first', 'last'), default='first',
                        help='last reverses time exactly as in mid-term Table 1')
    parser.add_argument('--group-size', type=int, default=64)
    parser.add_argument('--gain', type=float, default=DEFAULT_EVALUATION_GAIN,
                        help='same fixed reconstruction gain for all scenes/methods; default 2.7')
    parser.add_argument('--pipeline', action='append', default=[], metavar='MODULE:FUNCTION',
                        help='additional compatible cube pipeline, sample/full-burst modes only')
    parser.add_argument('--out', type=Path, default=Path('results/evaluation'))
    args = parser.parse_args(argv)
    if not np.isfinite(args.gain) or args.gain <= 0 or args.group_size <= 0:
        parser.error('gain and group-size must be finite and positive')
    if args.frames is not None and args.frames <= 0:
        parser.error('--frames must be a positive integer')
    crop = _check_crop(args.crop, parser, packed=args.scene_dir is not None)
    if args.report_table and args.pipeline:
        parser.error('--report-table uses the five original methods; additional --pipeline not supported')
    if args.archives and args.report_table:
        parser.error('--report-table requires raw cubes, not precomputed archives')
    if args.archives and (args.pipeline or args.frames is not None or crop is not None or args.reference != 'first'):
        parser.error('archives already contain reconstructions; --pipeline, --frames, --crop and --reference require samples or --scene-dir')
    if args.scene_dir is not None:
        if not args.scene_dir.is_dir():
            parser.error(f'scene folder not found: {args.scene_dir}')
        if len(set(args.bursts)) != len(args.bursts) or any(i < 0 for i in args.bursts):
            parser.error('--bursts must contain distinct nonnegative indices')
        for idx in args.bursts:
            for ext in ('.npy', '.png'):
                path = args.scene_dir / f'{idx:06d}{ext}'
                if not path.is_file():
                    parser.error(f'file not found: {path}')
        inputs = [args.scene_dir / f'{idx:06d}.npy' for idx in args.bursts]
    else:
        inputs = args.samples or args.archives
        if len(set(x.stem for x in inputs)) != len(inputs):
            parser.error('input stems must be distinct')
        for path in inputs:
            if not path.is_file():
                parser.error(f'file not found: {path}')
    args.out.mkdir(parents=True, exist_ok=True)
    rows, counts_by_burst = [], {}
    for path in inputs:
        if args.scene_dir is not None:
            index = int(path.stem)
            gt, methods, nframes = _full_images(args.scene_dir, index, args.group_size,
                                                 args.pipeline, args.frames, crop, args.reference, args.report_table)
            sample_name = f'{args.scene_dir.name}_{index:06d}'
        elif args.samples:
            gt, methods, nframes = _sample_images(path, args.group_size, args.pipeline,
                                                   args.frames, crop, args.reference, args.report_table)
            sample_name = path.stem
        else:
            gt, methods, nframes = _archive_images(path)
            sample_name = path.stem.replace('_reconstructions', '')
        if not methods:
            raise ValueError(f'no methods for {path}')
        counts_by_burst[sample_name] = nframes
        panels = [('Ground truth', gt)]
        for method, image in methods.items():
            if image.shape != gt.shape:
                raise ValueError(f'{path}: shape mismatch for {method}: {image.shape} vs {gt.shape}')
            mapped = _map_intensity(image, args.gain)
            score = evaluate_channels(mapped, gt)
            row = {'sample': sample_name, 'method': method,
                   'gain': args.gain, 'group_size': args.group_size if nframes is not None else '',
                   'frames': nframes if nframes is not None else '',
                   'reference': (('first' if method in ('naive', 'aligned_first') else 'last')
                                 if args.report_table else args.reference) if nframes is not None else 'unknown (archive)',
                   'height': int(gt.shape[0]), 'width': int(gt.shape[1])}
            for metric in ('psnr', 'ssim'):
                for channel, value in score[metric].items():
                    row[f'{metric}_{channel}'] = value
            rows.append(row)
            panels.append((method, mapped))
        _figure(panels, args.out / (sample_name + '_comparison.png'))
    with (args.out / 'results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    info = {'source_type': 'full_bursts' if args.scene_dir else 'samples' if args.samples else 'archives',
            'scene_dir': str(args.scene_dir) if args.scene_dir else None,
            'bursts': args.bursts if args.scene_dir else None,
            'crop_y0_y1_x0_x1': list(crop) if crop is not None else None,
            'frame_selection': 'first N frames, then reverse for last reference' if args.frames is not None else 'all available frames, reverse for last reference',
            'frames_requested': args.frames,
            'frames_used': counts_by_burst,
            'group_size': args.group_size if not args.archives else None,
            'reference_group': ('mixed first/last as in Table 1' if args.report_table else args.reference) if not args.archives else 'unknown; precomputed archive',
            'report_table_variants': args.report_table,
            'reference_implementation': 'reverse full selected cube for last-group methods; preserve first group for aligned_first' if args.report_table else 'reverse the frame order; existing first-group pipeline selects original last group' if args.reference == 'last' and not args.archives else 'native first group or precomputed archive',
            'intensity_rule': 'clip(gain * reconstruction, 0, 1)',
            'intensity_policy': 'fixed gain for all methods/scenes; no per-image or GT-fitted gains',
            'gain': args.gain,
            'gain_scope': 'same fixed value for all scenes, methods and RGB channels',
            'ground_truth_mapping': 'PNG divided by 255; no gain',
            'metric_protocol': 'per-RGB-channel PSNR/SSIM and arithmetic channel means; SSIM 7x7 uniform reflect padded',
            'report_comparability': 'Matches report scene/crop/frames/reference when configured; fixed 2.7 gain differs from report least-squares fitting, so PSNR will not exactly match Table 1',
            'inputs': [str(x) for x in inputs], 'rows': len(rows)}
    (args.out / 'protocol.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(rows)} method results, figures and protocol to {args.out}')
    return rows


if __name__ == '__main__':
    main()
