"""Manual real-data CPU/native-before/native-after comparison.

Run from the repository root. The baseline DLL must be built separately from
the pre-audit revision under build/audit_baseline. Saved sessions are read only.
"""
import json
import os
from pathlib import Path
import sys
import time

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from strainx.core import cuda_native
from strainx.core.rg_dic import DICParams, run_rg_dic


def main():
    session = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'dic_results_external_vid.h5')
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    with h5py.File(session, 'r') as saved:
        ref_path = (saved.attrs['reference_image'] if frame == 0 else
                    saved[f'frame_{frame-1:04d}'].attrs['image_path'])
        cur_path = saved[f'frame_{frame:04d}'].attrs['image_path']
        roi = saved['roi_mask'][:].astype(bool)
    def load(path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        return image.astype(np.float64) / 255.0
    ref, cur = load(ref_path), load(cur_path)
    params = DICParams(hole_recovery='ncc')
    yy, xx = np.nonzero(roi)
    seed_xy = (int(xx.mean()), int(yy.mean()))
    output = dict(session=session.name, frame=frame, reference=ref_path,
                  current=cur_path, params=vars(params), seed=seed_xy,
                  note='Raw correlation only; dynamic ROI and strain are excluded.')
    fields = {}
    for name in ('before', 'after', 'cpu'):
        start = time.perf_counter()
        if name == 'cpu':
            result = run_rg_dic(ref, cur, roi, params, seed_xy=seed_xy)
            u, v, corr = result.u, result.v, result.corr
            counts = [int(result.analyzed.sum())]
        else:
            directory = 'audit_baseline' if name == 'before' else 'native_cuda'
            os.environ['STRAINX_CUDA_LIBRARY'] = str(ROOT/'build'/directory/'bin/Release/strainx_cuda.dll')
            cuda_native.native_cuda_available(refresh=True)
            solver = cuda_native.NativeCudaSolver(params)
            try:
                solver.precompute_reference(ref, roi)
                seed = int(np.argmin(np.where(solver.valid_mask,
                    (solver.gx_flat-seed_xy[0])**2 + (solver.gy_flat-seed_xy[1])**2, np.inf)))
                result = solver.solve_frame(cur, seed_idx=seed)
                counts = [int(np.isfinite(result[0]).sum())]
                for _ in range(params.hole_recovery_passes):
                    result = solver.recover_failed(strategy='ncc')
                    counts.append(int(np.isfinite(result[0]).sum()))
                    if counts[-1] <= counts[-2]:
                        break
                u, v, corr = result[0], result[1], result[6]
            finally:
                solver.close()
        valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(corr)
        fields[name] = (u, v, valid)
        output[name] = dict(accepted=int(valid.sum()), passes=counts,
                            elapsed_s=time.perf_counter()-start)
        print(name, output[name], flush=True)
    for name in ('before', 'after'):
        u, v, valid = fields[name]
        cu, cv, cvalid = fields['cpu']
        common = valid & cvalid
        error = np.hypot(u[common]-cu[common], v[common]-cv[common])
        output[name].update(cpu_only=int((cvalid & ~valid).sum()),
                            gpu_only=int((valid & ~cvalid).sum()),
                            common=int(common.sum()),
                            median_delta_px=float(np.median(error)) if error.size else None,
                            p95_delta_px=float(np.quantile(error, .95)) if error.size else None)
    dest = ROOT/'output/verification'/f'audit_{session.stem}_{frame}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
