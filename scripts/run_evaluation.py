"""Track 4 mid-term harness; run from the repo root.

Examples:
  py scripts/run_evaluation.py --samples data/samples/kitchen1_000000.npz data/samples/kitchen1_000001.npz --gain 2.7
  py scripts/run_evaluation.py --archives results/sample_taa/kitchen1_000000_reconstructions.npz --gain 2.7

Use one FIXED gain for every reconstruction. This is a provisional exposure
convention, not a calibrated camera response; do not mix with method-wise fitted
gains. The source arrays and the ground truth are not altered.
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
from src.burst_io import load_sample, naive_average
from src.baseline import aligned_average
from src.taa import TAAConfig, reconstruct_taa
from src.evaluation import evaluate_channels


def _map_intensity(image, gain):
    return np.clip(np.asarray(image, dtype=np.float32) * gain, 0, 1)


def _rgb_png(image):
    return Image.fromarray(np.rint(np.clip(image, 0, 1)*255).astype(np.uint8), 'RGB')


def _figure(images, output):
    width, height = images[0][1].shape[1], images[0][1].shape[0]
    scale = max(1, 280 // width)
    gap, label_h = 12, 30
    canvas = Image.new('RGB', ((width*scale + gap)*len(images) + gap,
                               height*scale + 2*gap + label_h), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, array) in enumerate(images):
        x = gap+i*(width*scale+gap)
        draw.text((x, gap), label, fill='black')
        canvas.paste(_rgb_png(array).resize((width*scale, height*scale)), (x, gap+label_h))
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


def _sample_images(path, group_size, extras):
    cube = load_sample(path)
    gt_path = path.with_name(path.stem + '_gt.png')
    with Image.open(gt_path) as image:
        gt = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
    methods = {'naive': naive_average(cube),
               'aligned': aligned_average(cube, group_size)[0],
               'taa': reconstruct_taa(cube, group_size, TAAConfig()).reconstruction}
    for spec in extras:
        methods[spec.replace(':', '_')] = _custom_pipeline(spec, cube, group_size)
    return gt, methods


def _archive_images(path):
    with np.load(path) as archive:
        gt = archive['ground_truth'].copy()
        methods = {name: archive[name].copy() for name in ('naive', 'aligned', 'taa') if name in archive}
    if len(methods) < 3:
        raise ValueError(f'{path} must contain naive, aligned, and taa arrays')
    return gt, methods


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--samples', nargs='+', type=Path)
    source.add_argument('--archives', nargs='+', type=Path)
    parser.add_argument('--group-size', type=int, default=64)
    parser.add_argument('--gain', type=float, default=2.7,
                        help='fixed shared reconstruction gain; provisional default 2.7')
    parser.add_argument('--pipeline', action='append', default=[], metavar='MODULE:FUNCTION',
                        help='additional compatible cube pipeline, sample mode only')
    parser.add_argument('--out', type=Path, default=Path('results/evaluation_midterm'))
    args = parser.parse_args(argv)
    if not np.isfinite(args.gain) or args.gain <= 0 or args.group_size <= 0:
        parser.error('gain and group-size must be finite and positive')
    if args.archives and args.pipeline:
        parser.error('--pipeline requires --samples')
    inputs = args.samples or args.archives
    if len(set(x.stem for x in inputs)) < len(inputs):
        parser.error('input stems must be distinct')
    for path in inputs:
        if not path.is_file():
            parser.error(f'file not found: {path}')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in inputs:
        gt, methods = (_sample_images(path, args.group_size, args.pipeline)
                       if args.samples else _archive_images(path))
        if not methods:
            raise ValueError(f'no reconstruction methods for {path}')
        panels = [('Ground truth', gt)]
        for method, image in methods.items():
            if image.shape != gt.shape:
                raise ValueError(f'{path}: shape mismatch for {method}')
            mapped = _map_intensity(image, args.gain)
            score = evaluate_channels(mapped, gt)
            row = {'sample': path.stem.replace('_reconstructions', ''), 'method': method,
                   'gain': args.gain, 'group_size': args.group_size if args.samples else '',
                   'height': int(gt.shape[0]), 'width': int(gt.shape[1])}
            for metric in ('psnr', 'ssim'):
                for channel, value in score[metric].items():
                    row[f'{metric}_{channel}'] = value
            rows.append(row)
            panels.append((method, mapped))
        _figure(panels, args.out / (path.stem.replace('_reconstructions', '')+'_comparison.png'))
    with (args.out / 'results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    info = {'intensity_rule': 'clip(gain * reconstruction, 0, 1)',
            'gain': args.gain, 'gain_scope': 'same fixed value for ALL scenes and methods',
            'metric_protocol': 'per-RGB-channel PSNR/SSIM, arithmetic channel means; SSIM 7x7 uniform reflect padded',
            'method_reference': 'existing first-group reference for aligned and TAA (not end-of-burst report reference)',
            'evaluation_data': 'committed development samples, not held-out evaluation' if args.samples else 'provided archives; check metadata for source/reference',
            'inputs': [str(x) for x in inputs], 'rows': len(rows)}
    (args.out/'protocol.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
    print(f'Wrote {len(rows)} method results, figures and protocol to {args.out}')
    return rows


if __name__ == '__main__':
    main()
