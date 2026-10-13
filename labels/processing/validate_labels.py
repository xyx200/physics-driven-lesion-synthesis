#!/usr/bin/env python3
"""Read-only full decompression and label-value validation of this package.

Requires NumPy. Reports are written only to the explicitly chosen output path.
The expected case inventory is case_manifest.csv, not a directory-name guess.
"""
import argparse
import csv
import gzip
import json
import struct
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

def inspect(path, allowed, cropped=False):
    with gzip.open(path, 'rb') as f:
        h=f.read(348)
        endian='<' if struct.unpack('<i',h[:4])[0]==348 else '>'
        if struct.unpack(endian+'i',h[:4])[0]!=348 or h[344:348]!=b'n+1\0':
            raise ValueError('Expected single-file NIfTI-1')
        dims=struct.unpack(endian+'8h',h[40:56]); shape=dims[1:1+dims[0]]
        if cropped:
            if len(shape)!=3 or any(n<=0 or n>m for n,m in zip(shape,(240,240,155))):
                raise ValueError(f'Unexpected cropped grid: {shape}')
        elif shape!=(240,240,155): raise ValueError(f'Unexpected grid: {shape}')
        spacing=struct.unpack(endian+'8f',h[76:108])[1:4]
        if not np.allclose(spacing,[1,1,1],rtol=0,atol=1e-5): raise ValueError(f'Unexpected spacing: {spacing}')
        datatype=struct.unpack(endian+'h',h[70:72])[0]
        dtype={2:'u1',4:'i2',8:'i4',16:'f4',64:'f8',256:'i1',512:'u2',768:'u4',1024:'i8',1280:'u8'}[datatype]
        offset,slope,intercept=struct.unpack(endian+'3f',h[108:120])
        f.seek(int(offset)); data=f.read()  # Read through gzip CRC, including full voxel payload.
        expected=int(np.prod(shape))*np.dtype(dtype).itemsize
        if len(data)!=expected: raise ValueError(f'Voxel payload bytes {len(data)} != {expected}')
        a=np.frombuffer(data,dtype=endian+dtype)
        if slope not in (0.,1.) or (slope!=0 and intercept!=0): a=a*slope+intercept
        if not np.isfinite(a).all() or a.min()<0 or a.max()>7: raise ValueError('Non-finite or out-of-range values')
        integers=a.astype(np.uint8)
        if not np.array_equal(a,integers): raise ValueError('Non-integer label value')
        counts=np.bincount(integers,minlength=8)
        values=np.flatnonzero(counts).tolist()
        if not set(values).issubset(allowed): raise ValueError(f'Unexpected values: {values}')
        geometry={'qform_code':struct.unpack(endian+'h',h[252:254])[0],
                  'sform_code':struct.unpack(endian+'h',h[254:256])[0],
                  'srow':list(struct.unpack(endian+'12f',h[280:328]))}
        return dict(shape=list(shape),spacing=list(spacing),values=values,
                    voxel_counts=counts.tolist(),geometry=geometry,bytes=path.stat().st_size)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    with (args.root/'case_manifest.csv').open(encoding='utf-8-sig',newline='') as f: cases=list(csv.DictReader(f))
    if len(cases)!=3996 or len({(r['dataset'],r['case_id']) for r in cases})!=3996:
        raise ValueError('The frozen inventory must contain 3996 unique cases')
    jobs=[(f"labels/{r['dataset']}/{r['case_id']}.nii.gz",r['dataset'],r['split'],'final',
           set(range(8)) if r['dataset'] in {'GLI','MEN','WMH','MRBrainS18'} else {0,4}) for r in cases]
    def work(job):
        relative,dataset,split,kind,allowed=job
        row=dict(path=relative,dataset=dataset,split=split,variant=kind)
        try: row.update(inspect(args.root/relative,allowed,cropped=kind=='medlessynth_offline' and '_aug' in relative),status='pass')
        except Exception as e: row.update(status='error',error=str(e))
        return row
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work,jobs):
            records.append(row)
            if len(records)%250==0: print(f'Validated {len(records)}/{len(jobs)}',flush=True)
    errors=[r for r in records if r['status']!='pass']
    result=dict(status='pass' if not errors else 'failed',files=len(records),errors=errors,
                variant_counts=dict(Counter(r['variant'] for r in records)),records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({'status':result['status'],'files':len(records),'error_count':len(errors),
                      'first_errors':errors[:5],'variant_counts':result['variant_counts']},ensure_ascii=False),flush=True)
    if errors: raise SystemExit(1)

if __name__=='__main__': main()
