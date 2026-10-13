import argparse, json, os, sys, hashlib
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

RUN = Path(os.environ.get('BRAINSEG_OUTPUT_ROOT', '/path/to/brainseg_output'))
BASE = Path(os.environ.get('BRAINSEG_ASSETS_ROOT', '/path/to/brainseg_assets'))
CODE = BASE / 'code_runtime'
sys.path.insert(0, str(CODE / 'BCLIP'))
sys.path.insert(0, str(CODE))
from BCLIP import BCLIP
from BrainSeg import BrainSeg

IMG_SIZE = (224, 256, 224)
WEIGHTS = BASE / 'weights'
TISSUE_CAPTION = 'Age: Unknown, adult; Gender: Unknown; Modality: [T1w MRI, None, Flair, None, None]; Spacing: (1.0, 1.0, 1.0)'
LESION_CAPTION = TISSUE_CAPTION + '; Diagnosis: Lesion'

def process_image(image):
    image = image[None, None, :, :, :]
    tensor_image = torch.tensor(image).float()
    mask = tensor_image != 0
    if mask.any():
        vals = tensor_image[mask]
        tensor_image[mask] = (vals - vals.mean()) / (vals.std().clamp_min(1e-8))
    return tensor_image.squeeze(0)

def sobel_3d(image):
    sobel_x = torch.tensor([[[[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]], [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]], [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]]], dtype=torch.float32)
    sobel_y = torch.tensor([[[[-1, -3, -1], [0, 0, 0], [1, 3, 1]], [[-3, -6, -3], [0, 0, 0], [3, 6, 3]], [[-1, -3, -1], [0, 0, 0], [1, 3, 1]]]], dtype=torch.float32)
    sobel_z = torch.tensor([[[[-1, -3, -1], [-3, -6, -3], [-1, -3, -1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 3, 1], [3, 6, 3], [1, 3, 1]]]], dtype=torch.float32)
    image = image[None]
    grad_x = F.conv3d(image, sobel_x[None], padding=1)
    grad_y = F.conv3d(image, sobel_y[None], padding=1)
    grad_z = F.conv3d(image, sobel_z[None], padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2).squeeze(0)

def normalize_brainseg_state(state):
    fixed = {}
    for k, v in state.items():
        nk = k
        if nk.startswith('decoder1Seg_tissue.'):
            nk = 'decoder1Seg.' + nk[len('decoder1Seg_tissue.'):]
        elif nk.startswith('outSeg_tissue.'):
            nk = 'outSeg.' + nk[len('outSeg_tissue.'):]
        elif nk.startswith('attention.block.0.norm3.'):
            nk = 'attention.block.0.norm1.' + nk[len('attention.block.0.norm3.'):]
        fixed[nk] = v
    return fixed

def load_manifest(path, dataset=None, limit=None):
    rows=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r.get('status') != 'ok':
                continue
            if dataset and r['dataset'] != dataset:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows

def resize_arr(arr):
    t=torch.as_tensor(arr, dtype=torch.float32)[None, None]
    t=F.interpolate(t, size=IMG_SIZE, mode='trilinear', align_corners=False)
    return t[0,0].cpu().numpy()

def load_channel(path):
    img=nib.load(str(path))
    arr=resize_arr(img.get_fdata(dtype=np.float32))
    return process_image(arr), img.affine, img.shape

def build_tensor(subject_dir, seed_key):
    subject_dir=Path(subject_dir)
    torch.manual_seed(int(hashlib.sha1(seed_key.encode()).hexdigest()[:8], 16))
    t1, affine, orig_shape = load_channel(subject_dir / 'brain.nii.gz')
    flair, _, _ = load_channel(subject_dir / 'Flair-brain.nii.gz')
    edge = sobel_3d(t1)
    edge = process_image(edge.squeeze(0).numpy())
    noise_shape = t1.shape
    channels = [t1, torch.randn(noise_shape), flair, torch.randn(noise_shape), torch.randn(noise_shape), edge]
    x=torch.cat(channels, dim=0)[None]
    return x, affine, orig_shape

def save_prediction(logits, affine, orig_shape, out_path):
    pred=torch.argmax(logits, dim=1).float()[:, None]
    pred=F.interpolate(pred, size=orig_shape, mode='nearest')[0,0].cpu().numpy().astype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred, affine), str(out_path))

def load_models(device, modes):
    os.environ['BIOMEDCLIP_CACHE'] = str(BASE / 'hf_cache' / 'BiomedCLIP')
    os.environ['BIOMEDBERT_DIR'] = str(BASE / 'hf_cache' / 'BiomedBERT')
    os.environ['HF_HUB_OFFLINE'] = '1'
    clip=BCLIP(in_channels=6).to(device)
    clip_state=torch.load(str(WEIGHTS / 'BCLIP' / 'BCLIP.pt'), map_location=device)['model_state_dict']
    clip.load_state_dict(clip_state)
    clip.eval()
    models={}
    for mode in modes:
        m=BrainSeg(img_size=IMG_SIZE, in_channels=6, out_channel=4, feature_size=48, use_checkpoint=False).to(device)
        ckpt = WEIGHTS / 'BrainSeg_model' / ('BrainSeg_tissue.pt' if mode == 'tissue' else 'BrainSeg_lesion.pt')
        raw_state = torch.load(str(ckpt), map_location=device)['model_state_dict']
        missing, unexpected = m.load_state_dict(normalize_brainseg_state(raw_state), strict=False)
        print('model_state', mode, 'missing', len(missing), 'unexpected', len(unexpected), 'first_missing', missing[:3], 'first_unexpected', unexpected[:3], flush=True)
        m.eval()
        models[mode]=m
    return clip, models

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', default=str(RUN / 'manifests' / 'all_cases.jsonl'))
    ap.add_argument('--dataset')
    ap.add_argument('--mode', choices=['tissue','lesion','both'], default='both')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--resume', action='store_true')
    args=ap.parse_args()
    modes=['tissue','lesion'] if args.mode == 'both' else [args.mode]
    rows=load_manifest(args.manifest, args.dataset, args.limit)
    device=torch.device(args.device)
    print('rows', len(rows), 'modes', modes, 'device', device, flush=True)
    clip, models=load_models(device, modes)
    with torch.no_grad():
        for idx,r in enumerate(rows,1):
            x, affine, orig_shape = build_tensor(r['subject_dir'], r['dataset'] + '/' + r['case_id'])
            x=x.to(device)
            for mode in modes:
                out = RUN / 'predictions' / mode / r['dataset'] / r['case_id'] / f'pred_{mode}.nii.gz'
                if args.resume and out.exists():
                    print(idx, r['dataset'], r['case_id'], mode, 'skip_existing', flush=True)
                    continue
                caption = LESION_CAPTION if mode == 'lesion' else TISSUE_CAPTION
                text = clip.tokenizer(caption)[0][None].to(device)
                text_feature, _ = clip.text_encoder(text)
                logits = models[mode](x, text_feature)
                save_prediction(logits, affine, orig_shape, out)
                print(idx, r['dataset'], r['case_id'], mode, 'saved', out, flush=True)
            del x
            torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
