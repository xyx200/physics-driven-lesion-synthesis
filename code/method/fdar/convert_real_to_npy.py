import os
import argparse
import glob
import numpy as np
import nibabel as nib

def convert_nii_to_npy_rigorous(directory):
    search_pattern = os.path.join(directory, '*.nii.gz')
    nii_files = glob.glob(search_pattern)
    
    total_files = len(nii_files)
    print(f"Found {total_files} .nii.gz files. Starting strict conversion...\n")
    
    success_count = 0
    fail_count = 0
    
    for i, nii_path in enumerate(nii_files, 1):
        try:
            base_name = os.path.basename(nii_path)
            name_without_ext = base_name.replace('.nii.gz', '')
            npy_path = os.path.join(directory, f"{name_without_ext}.npy")
            
            img = nib.load(nii_path)
            
            data = np.asanyarray(img.dataobj)
            
            if np.issubdtype(data.dtype, np.floating):
                data = np.round(data)
                
            max_val = data.max()
            min_val = data.min()
            if max_val > 255 or min_val < 0:
                print(f"[{i}/{total_files}] Warning: {base_name} has values [{min_val}, {max_val}] outside the uint8 range; skipped!")
                fail_count += 1
                continue
                
            data_uint8 = data.astype(np.uint8)
            
            np.save(npy_path, data_uint8)
            success_count += 1
            
            if i % 10 == 0 or i == total_files:
                print(f"[{i}/{total_files}] Saved: {name_without_ext}.npy")
                
        except Exception as e:
            fail_count += 1
            print(f"[{i}/{total_files}] Conversion failed: {base_name}, error: {str(e)}")

    print("-" * 40)
    print(f"Conversion completed! Succeeded: {success_count}; failed/skipped: {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save evolved real-lesion NIfTI images as uint8 .npy files with matching names")
    parser.add_argument('directory')
    args = parser.parse_args()
    convert_nii_to_npy_rigorous(args.directory)
