import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import concurrent.futures

def binarize_single_label(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        
        data[data > 0] = 1
        
        data = data.astype(np.uint8)
        
        new_img = nib.Nifti1Image(data, img.affine, img.header)
        new_img.header.set_data_dtype(np.uint8)
        
        nib.save(new_img, file_path)
        
        return file_path, "✅ Success"
        
    except Exception as e:
        return file_path, f"❌ Failed: {str(e)}"

def batch_binarize_labels(directories, max_workers=None):
    target_files = []
    
    print("🔍 Scanning target folder...")
    for d in directories:
        if not os.path.exists(d):
            print(f"⚠️ Warning: folder does not exist: {d}")
            continue
            
        for root, _, files in os.walk(d):
            for filename in files:
                if filename.endswith(".nii.gz"):
                    name_without_ext = filename.replace(".nii.gz", "")
                    if name_without_ext.isdigit():
                        file_path = os.path.join(root, filename)
                        target_files.append(file_path)

    if not target_files:
        print("❌ No matching files found (numeric filenames ending in .nii.gz).")
        return

    print(f"🚀 Found {len(target_files)} matching label files.")
    print("Starting parallel binarization (setting labels >0 to 1)...")
    
    success_count = 0
    failed_files = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(binarize_single_label, target_files), 
            total=len(target_files), 
            desc="Binarization progress"
        ))

    for filepath, status in results:
        if "Success" in status:
            success_count += 1
        else:
            failed_files.append((filepath, status))
            
    print("\n" + "="*50)
    print(f"🎉 Label binarization completed!")
    print(f"✅ Successfully processed: {success_count} files")
    if failed_files:
        print(f"❌ Processing failed: {len(failed_files)} files")
        for f, err in failed_files:
            print(f"   - {os.path.basename(f)}: {err}")
    print("="*50)

if __name__ == "__main__":
    TARGET_DIRS = [
        "/path/to/data/WMH/testdata",
        "/path/to/data/WMH/traindata"
    ]
    
    batch_binarize_labels(TARGET_DIRS)