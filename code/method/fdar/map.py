import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import multiprocessing as mp

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def process_single_case(args):
    label_path, output_dir = args
    
    basename = os.path.basename(label_path)
    output_path = os.path.join(output_dir, basename)
        
    try:
        lbl_nii = nib.load(label_path)
        lbl_data = lbl_nii.get_fdata()
        
        mask_float = (lbl_data == 4).astype(np.float32)
        
        prob_map = gaussian_filter(mask_float, sigma=0.8)
        
        prob_map = prob_map.astype(np.float32)
        
        identity_affine = np.eye(4)
        new_img = nib.Nifti1Image(prob_map, identity_affine)
        
        nib.save(new_img, output_path)
        return f"✅ Success: {basename}"
        
    except Exception as e:
        return f"❌ Error: {basename} ({str(e)})"

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    labels_dir = "/path/to/data/MEDATASET/nnUNet_raw/nnUNet_raw_data/Task130_zaosheng/labelsTr"
    
    output_dir = "/path/to/assets/real_probability"
    
    os.makedirs(output_dir, exist_ok=True)
    
    label_files = glob.glob(os.path.join(labels_dir, "*.nii.gz"))
    total_files = len(label_files)
    
    if total_files == 0:
        print("No label files found. Please check the labelsTr path.")
        exit()
        
    print(f"Found {total_files} label files. Preparing multiprocessing...")
    
    tasks = [(lbl_path, output_dir) for lbl_path in label_files]
    
    num_cores = max(1, mp.cpu_count() - 2)
    print(f"==> Worker processes allocated: {num_cores}")
    
    success_count = 0
    
    with mp.Pool(processes=num_cores) as pool:
        for i, result_msg in enumerate(pool.imap_unordered(process_single_case, tasks)):
            if "✅" in result_msg:
                success_count += 1
                
            if (i + 1) % 10 == 0 or (i + 1) == total_files:
                print(f"Progress: {i + 1}/{total_files} | Latest status: {result_msg}")
                
    print("=" * 50)
    print(f"🎉 All smooth probability maps from binary masks generated! Succeeded: {success_count}/{total_files}")
    print(f"Output directory: {output_dir}")