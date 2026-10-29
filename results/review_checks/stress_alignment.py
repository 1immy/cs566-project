"""Seeded, local synthetic checks for the Track 2 review."""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.alignment import align_warp
from src.baseline import aligned_average


def summarise(reference, source, shift, margin=24):
    warped, flow = align_warp(source[None], reference)
    sl = np.s_[margin:-margin, margin:-margin]
    f = flow[sl]
    return {"shift": shift, "exact_fraction": float(np.mean(np.all(f == shift, axis=2))),
            "median_flow": np.median(f, axis=(0,1)).tolist(),
            "interior_squared_error": float(np.nanmean((warped[sl]-reference[sl])**2)),
            "invalid_fraction": float(np.mean(~np.isfinite(warped[...,0])))}


def main():
    records = {}
    rng = np.random.default_rng(566)
    shape = (97, 99, 3)
    reference = rng.integers(0,2,shape,dtype=np.uint8)
    records["exact_texture"] = []
    for shift in [(0,0),(1,1),(-1,-1),(3,-5),(-3,5),(7,9),(9,-7),(20,0)]:
        source = np.roll(reference, shift, axis=(0,1))
        records["exact_texture"].append(summarise(reference, source, shift))
    records["tiny_identity"] = []
    for h,w in [(1,1),(1,8),(8,1),(3,5),(7,9),(15,17),(31,33),(33,31)]:
        ref = rng.integers(0,2,(h,w,3),dtype=np.uint8)
        warped,flow=align_warp(ref[None],ref)
        records["tiny_identity"].append({"shape":[h,w],"correct":bool(np.all(warped==ref) and np.all(flow==0))})
    records["lost_border"] = []
    rng_border=np.random.default_rng(566)
    ref_border=rng_border.integers(0,2,(96,96,3),dtype=np.uint8)
    for shift in [(-8,0),(8,0),(0,-8),(0,8),(-4,0),(0,-4)]:
        src_border=np.roll(ref_border,shift,axis=(0,1))
        warped,flow=align_warp(src_border[None],ref_border)
        yy,xx=np.indices(ref_border.shape[:2])
        missing=(yy+shift[0]<0)|(yy+shift[0]>=96)|(xx+shift[1]<0)|(xx+shift[1]>=96)
        records["lost_border"].append({"shift":shift,
            "missing_pixels":int(np.count_nonzero(missing)),
            "finite_among_missing":float(np.mean(np.isfinite(warped[...,0][missing]))),
            "zero_flow_among_missing":float(np.mean(np.all(flow[missing]==0,axis=1))),
            "flow_corner":flow[0,0].tolist()})
    local_rng=np.random.default_rng(566)
    local_ref=local_rng.integers(0,2,(96,128,3),dtype=np.uint8)
    local_src=local_ref.copy()
    local_src[:,:64]=np.roll(local_ref[:,:64],3,axis=1)
    local_src[:,64:]=np.roll(local_ref[:,64:],-3,axis=1)
    warped,flow=align_warp(local_src[None],local_ref)
    records["local_motion"] = {
        "left_fraction":float(np.mean(np.all(flow[24:-24,24:40]==(0,3),axis=2))),
        "right_fraction":float(np.mean(np.all(flow[24:-24,88:104]==(0,-3),axis=2)))}
    records["photon_noise"] = []
    for kind in ["flat", "smooth", "textured"]:
        local_rng=np.random.default_rng(566)
        if kind=="flat":
            p=np.full((96,96,3),.157,dtype=np.float32)
        elif kind=="smooth":
            yy,xx=np.indices((96,96))
            p2=.15+.06*np.sin(xx/7)+.05*np.cos(yy/9)
            p=np.repeat(p2[...,None],3,axis=2)
        else:
            p=.05+.3*local_rng.random((96,96,3))
        for shift in [(0,0),(1,-1),(3,-5)]:
            moving_p=np.roll(p, shift, axis=(0,1))
            first=local_rng.binomial(1,p,size=(64,*p.shape)).astype(np.uint8)
            second=local_rng.binomial(1,moving_p,size=(64,*p.shape)).astype(np.uint8)
            warped,flow=align_warp(second, first.mean(axis=0,dtype=np.float32))
            interior=np.s_[24:-24,24:-24]
            est=(first.mean(axis=0,dtype=np.float32)+warped)/2
            naive=np.concatenate([first,second]).mean(axis=0)
            records["photon_noise"].append({"scene":kind,"shift":shift,
                "exact_fraction":float(np.mean(np.all(flow[interior]==shift,axis=2))),
                "median_flow":np.median(flow[interior],axis=(0,1)).tolist(),
                "naive_mse_known_p":float(np.mean((naive[interior]-p[interior])**2)),
                "aligned_mse_known_p":float(np.nanmean((est[interior]-p[interior])**2))})
    output=Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
