import os
import shutil
from pathlib import Path

def move_lowres_subjects_with_masks(source_dir, target_dir, min_size_mb=2):
    src_path = Path(source_dir)
    tgt_path = Path(target_dir)
    
    src_masks_path = src_path / "derivatives" / "lesion_masks"
    tgt_masks_path = tgt_path / "derivatives" / "lesion_masks"
    
    min_size_bytes = min_size_mb * 1024 * 1024

    tgt_path.mkdir(parents=True, exist_ok=True)
    tgt_masks_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning directory: {src_path}")
    print(f"Filter criterion: FLAIR file < {min_size_mb} MB\n")

    moved_count = 0
    
    for sub_dir in src_path.glob("sub-*"):
        if not sub_dir.is_dir():
            continue
            
        sub_id = sub_dir.name
        
        flair_files = list(sub_dir.rglob("*T1w.nii.gz"))
        
        should_move = False
        for flair_file in flair_files:
            file_size = flair_file.stat().st_size
            if file_size < min_size_bytes:
                size_mb = file_size / (1024 * 1024)
                print(f"Found very small file: {flair_file.name} (size: {size_mb:.2f} MB)")
                should_move = True
                break
                
        if should_move:
            print(f"----------")
            main_dest = tgt_path / sub_id
            print(f"  -> Moving subject data: {sub_id} to {tgt_path}")
            try:
                shutil.move(str(sub_dir), str(main_dest))
            except Exception as e:
                print(f"  [Error] Failed to move subject data {sub_id}: {e}")
            
            src_mask_dir = src_masks_path / sub_id
            if src_mask_dir.exists() and src_mask_dir.is_dir():
                mask_dest = tgt_masks_path / sub_id
                print(f"  -> Moving mask data (lesion_masks): {sub_id} to {tgt_masks_path}")
                try:
                    shutil.move(str(src_mask_dir), str(mask_dest))
                except Exception as e:
                    print(f"  [Error] Failed to move mask data {sub_id}: {e}")
            else:
                print(f"  -> No lesion_masks folder found for {sub_id}; skipping mask move.")

            moved_count += 1

    print(f"\nCleanup completed! Raw data and corresponding masks processed for {moved_count} subjects.")

if __name__ == '__main__':
    SOURCE_DIRECTORY = "/path/to/data/wsl_data/cuzhong"
    TARGET_DIRECTORY = "/path/to/data/wsl_data/cuzhonglowres"
    
    move_lowres_subjects_with_masks(SOURCE_DIRECTORY, TARGET_DIRECTORY, min_size_mb=3)