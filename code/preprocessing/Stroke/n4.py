import os
import gc
import time
import ants
from pathlib import Path
import concurrent.futures

MACHINE_ID = 0      
TOTAL_MACHINES = 2

MAX_WORKERS = 30 

INPUT_ROOT = "/path/to/data/oriCUZHONG/cuzhongHD"
OUTPUT_ROOT = "/path/to/data/newCU/cuzhongN4"
# =========================================

def run_n4_correction(task_args):
    brain_file, mask_file, output_file = task_args
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[N4 started] {brain_file.name}")
    
    img, mask, corrected_img = None, None, None
    try:
        img = ants.image_read(str(brain_file))
        mask = ants.image_read(str(mask_file))
        mask_binary = ants.threshold_image(mask, 0.5, 1e9)
        
        corrected_img = ants.n4_bias_field_correction(
            img, 
            mask=mask_binary, 
            shrink_factor=2, 
            verbose=False 
        )
        
        ants.image_write(corrected_img, str(output_file))
        return True, output_file.name
    except Exception as e:
        return False, f"{brain_file.name} | Error: {str(e)}"
    finally:
        del img, mask, corrected_img
        gc.collect()

def main():
    root_path = Path(INPUT_ROOT)
    n4_modalities = ['T1w', 'FLAIR']
    
    print(f"--- N4 processing across directories (evenly partitioned remaining tasks) ---")
    
    all_files = sorted(list(root_path.rglob("*.nii.gz")))
    remaining_tasks = []

    for f in all_files:
        if 'derivatives' in f.parts or '_bet' in f.name or 'mask' in f.name or '_n4' in f.name:
            continue
            
        if any(m in f.name for m in n4_modalities):
            expected_mask = f.parent / f"{f.name.split('.nii')[0]}_bet.nii.gz"
            
            if expected_mask.exists():
                rel_path = f.relative_to(root_path)
                out_name = f.name.replace(".nii.gz", "_n4.nii.gz")
                output_file = Path(OUTPUT_ROOT) / rel_path.parent / out_name
                
                if not output_file.exists():
                    remaining_tasks.append((f, expected_mask, output_file))

    print(f"Full scan completed. Remaining tasks without outputs: {len(remaining_tasks)}")

    if not remaining_tasks:
        print("All tasks have been processed. No files remain.")
        return

    my_tasks = [task for i, task in enumerate(remaining_tasks) if i % TOTAL_MACHINES == MACHINE_ID]
    print(f"Tasks assigned to this machine: {len(my_tasks)} (current machine ID: {MACHINE_ID}/{TOTAL_MACHINES})")

    if not my_tasks:
        print("No tasks assigned to this machine.")
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(run_n4_correction, my_tasks))
    
    success_count = sum(1 for s, _ in results if s)
    print(f"\nRemaining tasks completed! Succeeded on this machine: {success_count} / {len(my_tasks)}")

if __name__ == "__main__":
    main()