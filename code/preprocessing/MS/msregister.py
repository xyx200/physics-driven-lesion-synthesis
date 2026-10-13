import os
import glob
import time
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.ndimage import label

CUSTOM_TMP_DIR = "/path/to/data/TEST/newMS/temp_ants"
if not os.path.exists(CUSTOM_TMP_DIR):
    os.makedirs(CUSTOM_TMP_DIR)

os.environ["TMPDIR"] = CUSTOM_TMP_DIR
os.environ["TEMP"] = CUSTOM_TMP_DIR
os.environ["TMP"] = CUSTOM_TMP_DIR
# =========================================================

TEMPLATE_PATH = "/path/to/data/oriCUZHONG/spgr.nii" 

DATA_DIRS = [
    "/path/to/data/MS/MSLesSeg Dataset/train",
    "/path/to/data/MS/MSLesSeg Dataset/test"
]

N4_DIR = "/path/to/data/MS/n4"
OUTPUT_BASE_DIR = "/path/to/data/TEST/newMS"

MAX_WORKERS = 10      
ITK_THREADS = "8"    
# ================================================================

def find_all_tasks(data_dirs):
    tasks = []
    n4_file_map = {}
    if os.path.exists(N4_DIR):
        for root, dirs, files in os.walk(N4_DIR):
            for f in files:
                n4_file_map[f] = os.path.join(root, f)
    else:
        print(f"⚠️ Warning: N4 directory does not exist. Please check the path -> {N4_DIR}")
        
    for d in data_dirs:
        if not os.path.exists(d):
            continue
            
        split_name = "train" if "train" in d.lower() else "test"

        for root, dirs, files in os.walk(d):
            base_names = set()
            for f in files:
                if "_MASK.nii" in f:
                    base_names.add(f.replace("_MASK.nii.gz", "").replace("_MASK.nii", ""))
                elif "_T1.nii" in f:
                    base_names.add(f.replace("_T1.nii.gz", "").replace("_T1.nii", ""))
            
            for base_name in base_names:
                t1_path = n4_file_map.get(f"{base_name}_T1.nii.gz") or n4_file_map.get(f"{base_name}_T1.nii")
                flair_path = n4_file_map.get(f"{base_name}_FLAIR.nii.gz") or n4_file_map.get(f"{base_name}_FLAIR.nii")
                
                mask_files = [f for f in files if f.startswith(base_name) and ("_MASK.nii" in f)]
                mask_path = os.path.join(root, mask_files[0]) if mask_files else None
                
                if t1_path or flair_path:
                    tasks.append({
                        "base_name": base_name,
                        "split": split_name,
                        "t1_path": t1_path,
                        "flair_path": flair_path,
                        "mask_path": mask_path
                    })
                    
    unique_tasks = {t['base_name']: t for t in tasks}.values()
    return list(unique_tasks)

def process_subject(task_info):
    base_name = task_info["base_name"]
    split = task_info["split"]
    t1_path = task_info["t1_path"]
    flair_path = task_info["flair_path"]
    mask_path = task_info["mask_path"]
    
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = ITK_THREADS
    import ants  

    img_out_dir = os.path.join(OUTPUT_BASE_DIR, "imagesTr" if split == "train" else "imagesTs")
    lbl_out_dir = os.path.join(OUTPUT_BASE_DIR, "labelsTr" if split == "train" else "labelsTs")
    trans_out_dir = os.path.join(OUTPUT_BASE_DIR, "transforms")

    out_t1_path = os.path.join(img_out_dir, f"{base_name}_0000.nii.gz")
    out_flair_path = os.path.join(img_out_dir, f"{base_name}_0001.nii.gz")
    out_mask_path = os.path.join(lbl_out_dir, f"{base_name}.nii.gz")

    if os.path.exists(out_t1_path) and os.path.exists(out_mask_path):
        return base_name, True, "Already exists; skipping processing"

    is_special_case = (base_name == "P49_T2")

    try:
        fixed_template = ants.image_read(TEMPLATE_PATH)
        
        has_t1 = t1_path and os.path.exists(t1_path) and os.path.getsize(t1_path) > 0
        has_flair = flair_path and os.path.exists(flair_path) and os.path.getsize(flair_path) > 0
        has_mask = mask_path and os.path.exists(mask_path)
        
        moving_t1 = ants.image_read(t1_path) if has_t1 else None
        moving_flair = ants.image_read(flair_path) if has_flair else None
        moving_lesion = ants.image_read(mask_path) if has_mask else None
        
        main_img = moving_flair if is_special_case else moving_t1
        sec_img = moving_t1 if is_special_case else moving_flair
        
        # ========================================================================
        # ========================================================================
        intra_fwd_transforms = []
        if main_img is not None and sec_img is not None:
            if not is_special_case:
                tx_intra = ants.registration(fixed=main_img, moving=sec_img, type_of_transform='Rigid')
                intra_fwd_transforms = tx_intra['fwdtransforms']
            else:
                print(f"  -> Special case {base_name}: abnormal T1; skipping within-subject rigid alignment.")

        # ========================================================================
        # ========================================================================
        tx_inter = ants.registration(fixed=fixed_template, 
                                     moving=main_img, 
                                     type_of_transform='Rigid')
        inter_fwd_transforms = tx_inter['fwdtransforms']

        # ========================================================================
        # ========================================================================
        # 1. Main -> Template
        warped_main = tx_inter['warpedmovout']
        
        warped_sec = None
        if sec_img is not None:
            combined_transforms = inter_fwd_transforms + intra_fwd_transforms
            warped_sec_raw = ants.apply_transforms(fixed=fixed_template,
                                                   moving=sec_img,
                                                   transformlist=combined_transforms,
                                                   interpolator='linear')
            warped_sec = warped_sec_raw

        warped_lesion = None
        if moving_lesion is not None:
            mask_transforms = inter_fwd_transforms + intra_fwd_transforms
            warped_lesion = ants.apply_transforms(fixed=fixed_template,
                                                  moving=moving_lesion,
                                                  transformlist=mask_transforms,
                                                  interpolator='nearestNeighbor')

        warped_t1 = warped_sec if is_special_case else warped_main
        warped_flair = warped_main if is_special_case else warped_sec

        # ========================================================================
        # ========================================================================
        def get_largest_cc(binary_mask):
            labeled_array, num_features = label(binary_mask)
            if num_features == 0:
                return np.zeros_like(binary_mask, dtype=bool)
            sizes = np.bincount(labeled_array.ravel())
            sizes[0] = 0
            return labeled_array == sizes.argmax()

        t1_data = warped_t1.numpy() if warped_t1 is not None else None
        flair_data = warped_flair.numpy() if warped_flair is not None else None

        if flair_data is not None:
            flair_cc = get_largest_cc(flair_data > 0)
            flair_data[~flair_cc] = 0
            flair_data[flair_data < 0.1] = 0

        if t1_data is not None:
            if not is_special_case:
                t1_cc = get_largest_cc(t1_data > 0)
                t1_data[~t1_cc] = 0
                t1_data[t1_data < 0.1] = 0

        final_lesion = warped_lesion

        final_t1 = warped_t1.new_image_like(t1_data) if t1_data is not None else None
        final_flair = warped_flair.new_image_like(flair_data) if flair_data is not None else None

        # ========================================================================
        # ========================================================================
        if final_t1 is not None: ants.image_write(final_t1, out_t1_path)
        if final_flair is not None: ants.image_write(final_flair, out_flair_path)
        if final_lesion is not None: ants.image_write(final_lesion, out_mask_path)
            
        for i, transform_path in enumerate(inter_fwd_transforms):
            ext = os.path.splitext(transform_path)[1]
            if ext == '.mat':
                new_name = f"{base_name}_Affine.mat"
                shutil.copy(transform_path, os.path.join(trans_out_dir, new_name))

        msg = "Combined registration and cleanup completed successfully" if not is_special_case else "Combined registration and cleanup completed successfully for special case"
        return base_name, True, msg
    
    except Exception as e:
        error_msg = str(e) + "\n" + traceback.format_exc()
        return base_name, False, error_msg

def main():
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"Specified template file not found: {TEMPLATE_PATH}")

    directories_to_create = [
        os.path.join(OUTPUT_BASE_DIR, "imagesTr"),
        os.path.join(OUTPUT_BASE_DIR, "labelsTr"),
        os.path.join(OUTPUT_BASE_DIR, "imagesTs"),
        os.path.join(OUTPUT_BASE_DIR, "labelsTs"),
        os.path.join(OUTPUT_BASE_DIR, "transforms")
    ]
    for directory in directories_to_create:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ensured: {directory}")

    print("\n>>> Scanning data for processing tasks...")
    tasks = find_all_tasks(DATA_DIRS)
    total_tasks = len(tasks)
    print(f">>> Scan completed! Found {total_tasks} subjects/scan time points.")
    if total_tasks == 0:
        return

    print(f"\n>>> Starting parallel processing... (processes: {MAX_WORKERS}, threads per process: {ITK_THREADS})")
    start_time = time.time()
    
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_subject, t): t for t in tasks}
        
        for future in as_completed(future_to_task):
            base_name, success, msg = future.result()
            if success:
                success_count += 1
                print(f"✅ [{success_count + fail_count}/{total_tasks}] {base_name} -> {msg}")
            else:
                fail_count += 1
                print(f"❌ [{success_count + fail_count}/{total_tasks}] {base_name} failed:\n{msg}")

    print("\n" + "="*50)
    print("All batch processing tasks completed!")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"Succeeded: {success_count} cases | Failed: {fail_count} cases")
    print("="*50)

if __name__ == "__main__":
    main()