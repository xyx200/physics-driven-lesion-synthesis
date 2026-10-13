#!/usr/bin/env python3
"""Match the frozen paper cohort to downloaded WMH/MS/Stroke/MRBrainS18 images."""
import argparse,csv
from collections import defaultdict
from pathlib import Path

def image_index(root):
    images=defaultdict(list)
    for path in root.rglob('*.nii*'):
        if path.is_file() and path.name.endswith(('.nii','.nii.gz')):
            stem=path.name[:-7] if path.name.endswith('.nii.gz') else path.stem
            images[stem].append(path)
    return images

def one(paths,description):
    if len(paths)!=1: raise ValueError(f'Expected one match: {description}; found {len(paths)}')
    return str(paths[0].resolve())

def image_pair(images,dataset,case_id):
    if dataset in {'WMH','MRBrainS18'}:
        subject=case_id if dataset=='WMH' else str(int(case_id.removeprefix('MRB18_')))
        t1_name='T1' if dataset=='WMH' else 'reg_T1'
        candidates=[p for p in images.get(t1_name,[])
                    if p.parent.name=='pre' and p.parent.parent.name==subject]
        t1=one(candidates,f'{dataset}/{case_id}/pre/{t1_name}')
        flair_name='FLAIR'
    else:
        t1_name=f'{case_id}_T1w' if dataset=='Stroke' else f'{case_id}_T1'
        t1=one(images.get(t1_name,[]),f'{dataset}/{t1_name}')
        flair_name=f'{case_id}_FLAIR'
    flair=one([p for p in images.get(flair_name,[]) if p.parent==Path(t1).parent],
              f'{Path(t1).parent}/{flair_name}')
    return t1,flair

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['WMH','MS','Stroke','MRBrainS18'])
    p.add_argument('--images-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--case-id',action='append',help='Optional repeatable subset for pilot checks')
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    with (root/'case_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        cases=[r for r in csv.DictReader(f) if r['dataset']==a.dataset and (not a.case_id or r['case_id'] in a.case_id)]
    if not cases: raise ValueError('No matching cases')
    if a.output.exists(): raise FileExistsError(a.output)
    images=image_index(a.images_root)
    rows=[]
    for r in cases:
        c=r['case_id'];t1,flair=image_pair(images,a.dataset,c)
        suffix='_Affine.mat' if a.dataset=='MS' else '_0GenericAffine.mat'
        transform=root/f'reproduction_resources/registration_transforms/{a.dataset}_{c}{suffix}'
        rows.append(dict(dataset=a.dataset,case_id=c,input_stage='raw',t1_path=t1,flair_path=flair,
                         t1_mask='',flair_mask='',
                         intra_transform='',atlas_transform=str(transform) if transform.is_file() else '',expected_t1='',expected_flair=''))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(f'Wrote {len(rows)} cases to {a.output}')

if __name__=='__main__': main()
