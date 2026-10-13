import os
import glob
import time
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

CUSTOM_TMP_DIR = "/path/to/data/TEST/newCU/temp_ants"
if not os.path.exists(CUSTOM_TMP_DIR):
    os.makedirs(CUSTOM_TMP_DIR)

os.environ["TMPDIR"] = CUSTOM_TMP_DIR
os.environ["TEMP"] = CUSTOM_TMP_DIR
os.environ["TMP"] = CUSTOM_TMP_DIR
# =========================================================

TEMPLATE_PATH = "/path/to/data/oriCUZHONG/spgr.nii"

DATA_DIR = "/path/to/data/newCU/cuzhong_registered_Indirect_CFM"

IMAGES_OUT_DIR = "/path/to/data/TEST/newCU/image"
LABELS_OUT_DIR = "/path/to/data/TEST/newCU/label"
TRANSFORMS_OUT_DIR = "/path/to/data/TEST/newCU/transforms"

# ================================================================

def check_files_ready(sub_dir):
    t1_files = glob.glob(os.path.join(sub_dir, "*_T1w.nii*"))
    flair_files = glob.glob(os.path.join(sub_dir, "*_FLAIR.nii*"))
    lesion_files = glob.glob(os.path.join(sub_dir, "*_lesion_mask.nii*"))
    
    if not t1_files:
        return False, None, None, None
        
    all_target_files = t1_files + flair_files + lesion_files
    current_time = time.time()
    
    for f in all_target_files:
        if os.path.getsize(f) == 0:
            return False, None, None, None
        
        time_diff = current_time - os.path.getmtime(f)
        
        if 0 <= time_diff < 10.0:
            return False, None, None, None

    t1_path = t1_files[0]
    flair_path = flair_files[0] if flair_files else None
    lesion_path = lesion_files[0] if lesion_files else None
    
    return True, t1_path, flair_path, lesion_path


def process_subject(subject_id, t1_path, flair_path, lesion_path):
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
    
    import ants

    try:
        print(f"[{subject_id}] Starting processing...")
        fixed_img = ants.image_read(TEMPLATE_PATH)
        moving_t1 = ants.image_read(t1_path)

        print(f"[{subject_id}] Performing rigid linear registration...")
        tx = ants.registration(fixed=fixed_img, 
                               moving=moving_t1, 
                               type_of_transform='Rigid') 
        
        print(f"[{subject_id}] Registration completed; saving results...")
        for transform_path in tx['fwdtransforms']:
            if transform_path.endswith('.mat'):
                new_name = f"{subject_id}_0GenericAffine.mat"
            elif transform_path.endswith('.nii.gz'):
                new_name = f"{subject_id}_1Warp.nii.gz"
            else:
                new_name = f"{subject_id}_UnknownTransform" + os.path.splitext(transform_path)[1]
            dest_path = os.path.join(TRANSFORMS_OUT_DIR, new_name)
            shutil.copy(transform_path, dest_path)
        
        out_t1_name = f"{subject_id}_T1_reg.nii.gz"
        out_t1_path = os.path.join(IMAGES_OUT_DIR, out_t1_name)
        ants.image_write(tx['warpedmovout'], out_t1_path)

        if flair_path and os.path.exists(flair_path):
            moving_flair = ants.image_read(flair_path)
            warped_flair = ants.apply_transforms(fixed=fixed_img,
                                                 moving=moving_flair,
                                                 transformlist=tx['fwdtransforms'],
                                                 interpolator='linear')
            out_flair_name = f"{subject_id}_FLAIR_reg.nii.gz"
            out_flair_path = os.path.join(IMAGES_OUT_DIR, out_flair_name)
            ants.image_write(warped_flair, out_flair_path)

        if lesion_path and os.path.exists(lesion_path):
            moving_lesion = ants.image_read(lesion_path)
            moving_lesion[moving_lesion > 0] = 1
            warped_lesion = ants.apply_transforms(fixed=fixed_img,
                                                  moving=moving_lesion,
                                                  transformlist=tx['fwdtransforms'],
                                                  interpolator='nearestNeighbor') 
            
            out_lesion_name = f"{subject_id}_lesion_reg.nii.gz"
            out_lesion_path = os.path.join(LABELS_OUT_DIR, out_lesion_name)
            ants.image_write(warped_lesion, out_lesion_path)

        return subject_id, True, "Success"
    
    except Exception as e:
        error_msg = str(e) + "\n" + traceback.format_exc()
        return subject_id, False, error_msg


def main():
    for directory in [IMAGES_OUT_DIR, LABELS_OUT_DIR, TRANSFORMS_OUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"Template not found at: {TEMPLATE_PATH}")

    processed_subjects = set()
    active_futures = {}

    MAX_WORKERS = 8
    print(f">>> Dynamic scan monitoring started...")
    print(f">>> Monitoring directory: {DATA_DIR}")
    print(f">>> Maximum concurrent tasks: {MAX_WORKERS}; new data will be processed automatically...\n")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        try:
            while True:
                done_futures = [f for f in active_futures.keys() if f.done()]
                for f in done_futures:
                    sub_id = active_futures.pop(f)
                    try:
                        _, success, msg = f.result()
                        if success:
                            print(f"✅ [Done] {sub_id} processed. Current queue: {len(active_futures)}/{MAX_WORKERS}")
                        else:
                            print(f"❌ [Failed] Error processing {sub_id}:\n{msg}")
                    except Exception as exc:
                        print(f"❌ [Crashed] Severe process exception for {sub_id}: {exc}")

                if len(active_futures) >= MAX_WORKERS:
                    time.sleep(3)
                    continue

                if os.path.exists(DATA_DIR):
                    sub_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
                    
                    for sub_dir in sub_dirs:
                        if len(active_futures) >= MAX_WORKERS:
                            break
                            
                        subject_id = os.path.basename(sub_dir)
                        
                        if subject_id in processed_subjects:
                            continue
                        
                        is_ready, t1_path, flair_path, lesion_path = check_files_ready(sub_dir)
                        
                        if is_ready:
                            processed_subjects.add(subject_id)
                            future = executor.submit(process_subject, subject_id, t1_path, flair_path, lesion_path)
                            active_futures[future] = subject_id
                            
                            print(f"👀 [Monitor] Ready data found: {subject_id}; assigning slot ({len(active_futures)}/{MAX_WORKERS})")

                time.sleep(3)
                
        except KeyboardInterrupt:
            print("\n>>> Interrupt received; shutting down the process pool and exiting...")
            executor.shutdown(wait=False, cancel_futures=True)

if __name__ == "__main__":
    main()