import numpy as np
import nibabel as nib
import json
import glob
from fun_pi import LesionGeneration3D

from tqdm import tqdm
import argparse
import os
import random
import multiprocessing as mp

def process_single_id(args):
    index, dataset_brats, dataset_wmh, img_save_path, lbl_save_path = args

    seed = (int.from_bytes(os.urandom(4), byteorder='little') + index) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)

    aug_types = ['brats', 'brats', 'wmh', 'wmh', random.choice(['brats', 'wmh'])]


    for aug_idx, a_type in enumerate(aug_types):
        if a_type == 'brats':
            t1_out, flair_out, label, affine, clean_path, param = dataset_brats[index]
        else:
            t1_out, flair_out, label, affine, clean_path, param = dataset_wmh[index]

        base_name = os.path.basename(clean_path).replace('_0000.nii.gz', '')

        img_t1 = nib.Nifti1Image(t1_out, affine=affine)
        img_flair = nib.Nifti1Image(flair_out, affine=affine)
        lbl = nib.Nifti1Image(label, affine=affine)

        out_name = f"{base_name}_aug{aug_idx}_{a_type}"
        img_prefix = os.path.join(img_save_path, out_name)
        lbl_path = os.path.join(lbl_save_path, out_name + '.nii.gz')

        nib.save(img_t1, img_prefix + '_0000.nii.gz')
        nib.save(img_flair, img_prefix + '_0001.nii.gz')
        nib.save(lbl, lbl_path)

        with open(img_prefix + '.json', 'w') as f:
            param['clean_path'] = clean_path
            param['aug_type'] = a_type
            json.dump(param, f, indent=4)
            
    return True


def generate_and_save_3d(img_list, lbl_list, img_save_path, lbl_save_path, image_size, num_workers=4, use_rician=False):
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    print("Building BraTS and WMH generation engines...")
    dataset_brats = LesionGeneration3D(
        img_path=img_list, gt_path=lbl_list, mask_path=None, which_data='brats',
        have_texture=True, have_noise=True, have_edema=True, have_smoothing=True,
        size=image_size, perturb=False, num_lesions=(1,4), semi_axis_range=[(5,10),(10,15),(15,20),(20,25)],
        centroid_scale=15, range_sampling=[0.1,0.2,0.3,0.4], num_ellipses=15, rician=use_rician
    )
    
    dataset_wmh = LesionGeneration3D(
        img_path=img_list, gt_path=lbl_list, mask_path=None, which_data='wmh',
        have_texture=True, have_noise=True, have_edema=True, have_smoothing=True,
        size=image_size, perturb=False, num_lesions=(5,10), semi_axis_range=[(2,5),(3,5)],
        centroid_scale=25, range_sampling=[0.7,0.3], num_ellipses=15, rician=use_rician
    )

    num_total_images = len(img_list)
    task_args = [(i, dataset_brats, dataset_wmh, img_save_path, lbl_save_path) for i in range(num_total_images)]
    
    print(f"Starting multiprocess generation with {num_workers} parallel workers...")
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=num_total_images) as pbar:
            for _ in pool.imap_unordered(process_single_id, task_args):
                pbar.update(1)


def create(use_rician, workers, start_idx, end_idx, **kwargs):
    img_dir = "/path/to/data/MEDATASET/nnUNet_raw/nnUNet_raw_data/Task180_PIIzhu/imagesTr"
    lbl_dir = "/path/to/data/MEDATASET/nnUNet_raw/nnUNet_raw_data/Task180_PIIzhu/labelsTr"
    
    img_save_path = "/path/to/data/MEDATASET/nnUNet_raw/nnUNet_raw_data/Task183_qita/weiimage"
    lbl_save_path = "/path/to/data/MEDATASET/nnUNet_raw/nnUNet_raw_data/Task183_qita/weilabel"

    all_t1_imgs = glob.glob(os.path.join(img_dir, "*_0000.nii.gz"))
    all_t1_imgs.sort()  

    paired_data = []

    for t1_path in all_t1_imgs:
        img_name = os.path.basename(t1_path)
        flair_path = t1_path.replace("_0000.nii.gz", "_0001.nii.gz")
        lbl_name = img_name.replace("_0000.nii.gz", ".nii.gz")
        lbl_path = os.path.join(lbl_dir, lbl_name)

        if os.path.exists(flair_path) and os.path.exists(lbl_path):
            paired_data.append(((t1_path, flair_path), lbl_path))

    end_idx = end_idx if end_idx is not None else len(paired_data)
    paired_data = paired_data[start_idx:end_idx]

    img_list = [p[0] for p in paired_data]
    lbl_list = [p[1] for p in paired_data]

    if len(img_list) == 0:
        print("Error: No data selected for processing. Check directory and indices.")
        return

    print(f"Successfully selected {len(img_list)} pairs to process (Indices {start_idx} to {end_idx}).")
    image_size = (128,128,128) 
    generate_and_save_3d(img_list, lbl_list, img_save_path, lbl_save_path, image_size, num_workers=workers, use_rician=use_rician)


if(__name__ =="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-use_rician", default=True, action='store_true')
    parser.add_argument("-workers", default=6, type=int)
    
    parser.add_argument("-start_idx", default=0, type=int, help="Start index for processing list")
    parser.add_argument("-end_idx", default=None, type=int, help="End index for processing list (None = to the end)")

    args, unknown = parser.parse_known_args()
    create(**vars(args))