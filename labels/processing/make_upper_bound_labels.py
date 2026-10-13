#!/usr/bin/env python3
"""Rebuild the 3096 historical Joint Upper Bound training masks.

2490 masks are exactly main_label == 4; the remaining 606 GLI masks use
the retained historical supplements. Defaults to preflight only.
"""
import argparse,csv,shutil
from pathlib import Path

def generate(root,row,output):
    import nibabel as nib
    import numpy as np
    dataset,case=row['dataset'],row['case_id']
    supplement=root/f'reproduction_resources/upper_bound_supplements/{dataset}_{case}.nii.gz'
    prefix={'Stroke':'newCU','MS':'newMS','WMH':'newWMH'}.get(dataset,dataset)
    destination=output/f'{prefix}_{case}.nii.gz'
    if destination.exists():raise FileExistsError(destination)
    if supplement.is_file():
        shutil.copy2(supplement,destination)
    else:
        image=nib.load(str(root/'labels'/dataset/f'{case}.nii.gz'))
        data=(np.asanyarray(image.dataobj)==4).astype(np.uint8)
        header=image.header.copy();header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(data,image.affine,header),str(destination))
    return destination

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--case',action='append',help='Optional repeatable dataset/case_id, e.g. GLI/BraTS-GLI-00000-000')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    with (root/'case_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        allrows=[r for r in csv.DictReader(f) if r['split']=='train']
    if len(allrows)!=3096:raise ValueError('Expected 3096 training cases')
    rows=[r for r in allrows if not a.case or r['dataset']+'/'+r['case_id'] in a.case]
    if not rows:raise ValueError('No matching training cases')
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    if len(list((root/'reproduction_resources/upper_bound_supplements').glob('*.nii.gz')))!=606:
        raise ValueError('Expected 606 historical GLI supplements')
    for row in rows:
        label_path=root/'labels'/row['dataset']/f"{row['case_id']}.nii.gz"
        if not label_path.is_file():raise FileNotFoundError(label_path)
    print(f'Cases: {len(rows)}; execute: {a.execute}')
    if a.execute:
        a.output_dir.mkdir(parents=True)
        for row in rows:generate(root,row,a.output_dir)

if __name__=='__main__':main()
