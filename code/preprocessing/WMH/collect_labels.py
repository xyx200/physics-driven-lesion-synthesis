import os
import gzip
import shutil
from tqdm import tqdm

def extract_and_convert_wmh(src_root, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    
    subject_dirs = []
    for root, dirs, files in os.walk(src_root):
        folder_name = os.path.basename(root)
        if folder_name.isdigit():
            subject_dirs.append((root, folder_name))
            
    if not subject_dirs:
        print(f"❌ No folders with numeric names found in {src_root}.")
        return

    print(f"🔍 Found {len(subject_dirs)} subject folders with numeric names.")
    print(f"🚀 Copying and compressing to .nii.gz in: {dst_dir}")
    
    success_count = 0
    fail_count = 0

    for folder_path, subj_id in tqdm(subject_dirs, desc="Processing progress"):
        source_wmh_path = os.path.join(folder_path, "wmh.nii")
        target_wmh_path = os.path.join(dst_dir, f"{subj_id}.nii.gz")
        
        if os.path.exists(source_wmh_path):
            try:
                with open(source_wmh_path, 'rb') as f_in:
                    with gzip.open(target_wmh_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                success_count += 1
            except Exception as e:
                print(f"\n❌ Error processing subject {subj_id}: {e}")
                fail_count += 1
        else:
            source_wmh_gz_path = os.path.join(folder_path, "wmh.nii.gz")
            if os.path.exists(source_wmh_gz_path):
                try:
                    shutil.copy2(source_wmh_gz_path, target_wmh_path)
                    success_count += 1
                except Exception as e:
                    print(f"\n❌ Error copying subject {subj_id}: {e}")
                    fail_count += 1
            else:
                print(f"\n⚠️ Warning: wmh.nii or wmh.nii.gz not found in folder {subj_id}")
                fail_count += 1

    print("\n" + "="*50)
    print(f"🎉 Extraction, compression, and renaming completed!")
    print(f"✅ Successfully processed: {success_count} files")
    if fail_count > 0:
        print(f"❌ Failed or skipped: {fail_count} folders")
    print("="*50)

if __name__ == "__main__":
    SRC_DIR = "/path/to/data/WMH/wmh_data/training"
    
    DST_DIR = "/path/to/data/WMH/traindata"
    
    extract_and_convert_wmh(SRC_DIR, DST_DIR)