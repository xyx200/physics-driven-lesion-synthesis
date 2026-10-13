import os
import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def process_single_file(args):
    input_path, output_path = args
    try:
        img = nib.load(input_path)
        
        target_ornt = axcodes2ornt(('L', 'P', 'S'))
        orig_ornt = io_orientation(img.affine)
        
        transform = ornt_transform(orig_ornt, target_ornt)
        img_lps = img.as_reoriented(transform)
        
        nib.save(img_lps, output_path)
        return True, f"Success: {os.path.basename(input_path)}"
    except Exception as e:
        return False, f"Failed: {os.path.basename(input_path)} -> {str(e)}"

def parallel_convert(input_root, output_root, max_workers=8):
    tasks = []
    
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith('.nii.gz'):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_root)
                out_dir = os.path.join(output_root, rel_path)
                output_path = os.path.join(out_dir, file)
                
                if os.path.exists(output_path):
                    continue
                    
                os.makedirs(out_dir, exist_ok=True)
                tasks.append((input_path, output_path))

    if not tasks:
        print("All files have been processed (or already exist). No new tasks to run.")
        return

    print(f"Scan completed. Found {len(tasks)} unprocessed files. Starting parallel conversion...")

    failed_tasks = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Conversion progress", unit="file"):
            success, msg = future.result()
            if not success:
                failed_tasks.append(msg)
    
    print("\n--- Processing completed ---")
    if failed_tasks:
        print(f"⚠️ Processing failed for {len(failed_tasks)} files:")
        for fail_msg in failed_tasks:
            print(fail_msg)
    else:
        print("✅ All files processed successfully!")

if __name__ == "__main__":
    IN_DIR = "/path/to/data/CUZHONG/cuzhong"
    OUT_DIR = "/path/to/data/CUZHONG/cuzhonglps"
    parallel_convert(IN_DIR, OUT_DIR, max_workers=10)