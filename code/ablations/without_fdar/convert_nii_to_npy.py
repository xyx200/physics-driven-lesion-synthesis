import os
import argparse
import glob
import numpy as np
import nibabel as nib
import concurrent.futures

def convert_nii_to_npy(nii_filepath):
    try:
        img = nib.load(nii_filepath)
        
        data = np.asanyarray(img.dataobj).astype(np.uint8)
        
        npy_filepath = nii_filepath.replace('.nii.gz', '.npy')
        
        np.save(npy_filepath, data)
        
        
        return nii_filepath, True, None
    except Exception as e:
        return nii_filepath, False, str(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save binary ellipsoid NIfTI masks as uint8 .npy files with matching names")
    parser.add_argument('directory')
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.directory, '*.nii.gz'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for path, success, error in executor.map(convert_nii_to_npy, files):
            if not success:
                raise RuntimeError(f"{path}: {error}")
            print(os.path.basename(path))
