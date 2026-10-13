import os
import argparse

custom_tmp = "/path/to/data/newCU/mytmp"
os.makedirs(custom_tmp, exist_ok=True)

os.environ["TMPDIR"] = custom_tmp
os.environ["TEMP"] = custom_tmp
os.environ["TMP"] = custom_tmp

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"

import ants
import concurrent.futures

def process_subject_indirect(sub_id, base_n4, base_hd, base_mask, out_dir):
    print(f"\n========== Processing subject: {sub_id} ==========")
    
    # ==========================================
    # ==========================================
    sub_out_dir = os.path.join(out_dir, sub_id)
    out_t1   = os.path.join(sub_out_dir, f'{sub_id}_T1w.nii.gz')
    out_flair = os.path.join(sub_out_dir, f'{sub_id}_FLAIR.nii.gz')
    out_dwi = os.path.join(sub_out_dir, f'{sub_id}_DWI.nii.gz')
    out_mask = os.path.join(sub_out_dir, f'{sub_id}_lesion_mask.nii.gz')
    
    if os.path.exists(out_t1) and os.path.exists(out_flair) and \
       os.path.exists(out_dwi) and os.path.exists(out_mask):
        print(f"⏭️ Results already exist; skipping registration: {sub_id}")
        return True

    # ==========================================
    # ==========================================
    t1_path = os.path.join(base_n4, sub_id, 'anat', f'{sub_id}_T1w_n4_chong.nii.gz')
    flair_path = os.path.join(base_n4, sub_id, 'anat', f'{sub_id}_FLAIR_n4.nii.gz')
    dwi_path = os.path.join(base_hd, sub_id, 'dwi', f'{sub_id}_rec-TRACE_dwi.nii.gz')
    mask_path = os.path.join(base_mask, sub_id, 'dwi', f'{sub_id}_space-TRACE_desc-lesion_mask.nii.gz')
    
    for p in [t1_path, flair_path, dwi_path, mask_path]:
        if not os.path.exists(p):
            print(f"❌ Warning: input file not found: {p}")
            return False
            
    # ==========================================
    # ==========================================
    t1 = ants.image_read(t1_path)
    flair = ants.image_read(flair_path)
    dwi = ants.image_read(dwi_path)
    lesion_mask = ants.image_read(mask_path)
    
    # ==========================================
    # ==========================================
    print(f"-> [{sub_id}] Computing FLAIR-to-T1 transform (Rigid) ...")
    reg_flair2t1 = ants.registration(fixed=t1, moving=flair, type_of_transform='Rigid')
    tx_flair_to_t1 = reg_flair2t1['fwdtransforms']
    
    flair_in_t1 = reg_flair2t1['warpedmovout']
    
    # ==========================================
    # ==========================================
    print(f"-> [{sub_id}] Generating healthy-tissue mask (for DWI -> FLAIR) ...")
    dwi_brain_mask = ants.get_mask(dwi)
    lesion_bin = ants.threshold_image(lesion_mask, low_thresh=0.5, high_thresh=1e9, inval=1, outval=0)
    lesion_inv = ants.threshold_image(lesion_bin, low_thresh=0.5, high_thresh=1.5, inval=0, outval=1)
    healthy_mask = dwi_brain_mask * lesion_inv
    
    # ==========================================
    # ==========================================
    print(f"-> [{sub_id}] Computing DWI-to-FLAIR transform (SyN + CFM) ...")
    reg_dwi2flair = ants.registration(
        fixed=flair, 
        moving=dwi, 
        type_of_transform='SyNRA',
        moving_mask=healthy_mask
    )
    
    tx_dwi_to_flair = reg_dwi2flair['fwdtransforms'] 

    # ==========================================
    # ==========================================
    print(f"-> [{sub_id}] Composing transforms and resampling DWI and mask once (to T1 space) ...")
    combined_transforms = tx_flair_to_t1 + tx_dwi_to_flair
    
    dwi_in_t1 = ants.apply_transforms(
        fixed=t1, 
        moving=dwi, 
        transformlist=combined_transforms, 
        interpolator='linear'
    )
    
    mask_in_t1 = ants.apply_transforms(
        fixed=t1, 
        moving=lesion_mask, 
        transformlist=combined_transforms, 
        interpolator='nearestNeighbor'
    )
    
    # ==========================================
    # ==========================================
    os.makedirs(sub_out_dir, exist_ok=True)
    
    ants.image_write(t1, out_t1)
    ants.image_write(flair_in_t1, out_flair)
    ants.image_write(dwi_in_t1, out_dwi)
    ants.image_write(mask_in_t1, out_mask)
    
    print(f"✅ {sub_id} processing completed! Results saved to {sub_out_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribute subject processing tasks across devices")
    parser.add_argument('--part', type=int, choices=[0, 1], required=True,
                        help="Enter 0 for the first half of the remaining subjects, or 1 for the second half")
    args = parser.parse_args()

    BASE_N4 = "/path/to/data/newCU/cuzhongN4"
    BASE_HD = "/path/to/data/oriCUZHONG/cuzhongHD"
    BASE_MASK = "/path/to/data/oriCUZHONG/cuzhonglps/derivatives/lesion_masks"
    
    OUT_DIR = "/path/to/data/newCU/cuzhong_registered_Indirect_CFM"
    
    if os.path.exists(BASE_N4):
        all_subjects = [d for d in os.listdir(BASE_N4) if d.startswith('sub-') and os.path.isdir(os.path.join(BASE_N4, d))]
        all_subjects.sort()
        
        # ==========================================
        # ==========================================
        unprocessed_subjects = []
        for sub_id in all_subjects:
            sub_out_dir = os.path.join(OUT_DIR, sub_id)
            out_t1   = os.path.join(sub_out_dir, f'{sub_id}_T1w.nii.gz')
            out_flair = os.path.join(sub_out_dir, f'{sub_id}_FLAIR.nii.gz')
            out_dwi = os.path.join(sub_out_dir, f'{sub_id}_DWI.nii.gz')
            out_mask = os.path.join(sub_out_dir, f'{sub_id}_lesion_mask.nii.gz')
            
            if not (os.path.exists(out_t1) and os.path.exists(out_flair) and \
                    os.path.exists(out_dwi) and os.path.exists(out_mask)):
                unprocessed_subjects.append(sub_id)
        
        total_unprocessed = len(unprocessed_subjects)
        print(f"🔍 Scan completed: {len(all_subjects)} subjects in total; {total_unprocessed} unprocessed.")
        
        if total_unprocessed == 0:
            print("🎉 All subjects have been processed!")
            exit()

        midpoint = total_unprocessed // 2
        
        if args.part == 0:
            target_subjects = unprocessed_subjects[:midpoint]
            print(f"🖥️ Device 0 started: processing the first {len(target_subjects)} unprocessed subjects")
        else:
            target_subjects = unprocessed_subjects[midpoint:]
            print(f"🖥️ Device 1 started: processing the last {len(target_subjects)} unprocessed subjects")
        
        if not target_subjects:
            print("⏭️ No tasks assigned to this node.")
            exit()

        max_workers = max(1, 20)
        print(f"🚀 Starting parallel processing with {max_workers} processes...\n")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_subject_indirect, sub, BASE_N4, BASE_HD, BASE_MASK, OUT_DIR): sub 
                for sub in target_subjects
            }
            
            for future in concurrent.futures.as_completed(futures):
                sub = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Unexpected error processing {sub}: {e}")
    else:
        print(f"❌ Data root directory not found: {BASE_N4}")