import os
import re
import shutil
import concurrent.futures
from tqdm import tqdm


import ants

def process_single_subject(args):
    subject_id, input_dir, images_out_dir, labels_out_dir, transforms_out_dir, template_path = args
    
    out_t1_path = os.path.join(images_out_dir, f"{subject_id}_0000.nii.gz")
    out_flair_path = os.path.join(images_out_dir, f"{subject_id}_0001.nii.gz")
    out_lesion_path = os.path.join(labels_out_dir, f"{subject_id}.nii.gz")
    
    if os.path.exists(out_t1_path) and os.path.exists(out_flair_path) and os.path.exists(out_lesion_path):
        return subject_id, "⏭️ Results already exist; Skipped"

    lesion_path = os.path.join(input_dir, f"{subject_id}.nii.gz")
    t1_path = os.path.join(input_dir, f"{subject_id}_T1.nii.gz")
    flair_path = os.path.join(input_dir, f"{subject_id}_FLAIR.nii.gz")

    if not os.path.exists(t1_path) or not os.path.exists(flair_path):
        return subject_id, "⚠️ Warning: T1 or FLAIR image is missing"

    try:
        template_img = ants.image_read(template_path)
        
        moving_t1 = ants.image_read(t1_path)
        moving_flair = ants.image_read(flair_path)
        moving_lesion = ants.image_read(lesion_path)

        tx = ants.registration(fixed=template_img, 
                               moving=moving_t1, 
                               type_of_transform='Rigid') 
        
        for transform_path in tx['fwdtransforms']:
            if transform_path.endswith('.mat'):
                new_name = f"{subject_id}_0GenericAffine.mat"
            else:
                new_name = f"{subject_id}_{os.path.basename(transform_path)}"
            dest_path = os.path.join(transforms_out_dir, new_name)
            shutil.copy(transform_path, dest_path)
        
        ants.image_write(tx['warpedmovout'], out_t1_path)

        warped_flair = ants.apply_transforms(fixed=template_img,
                                             moving=moving_flair,
                                             transformlist=tx['fwdtransforms'],
                                             interpolator='linear')
        ants.image_write(warped_flair, out_flair_path)

        moving_lesion[moving_lesion > 0] = 1
        
        warped_lesion = ants.apply_transforms(fixed=template_img,
                                              moving=moving_lesion,
                                              transformlist=tx['fwdtransforms'],
                                              interpolator='nearestNeighbor') 
        
        ants.image_write(warped_lesion, out_lesion_path)
        
        return subject_id, "✅ Success"

    except Exception as e:
        return subject_id, f"❌ Failed: {str(e)}"


def process_dataset_multiprocess(input_dir, output_dir, template_path, max_workers=None):
    print("\n" + "="*50)
    print(f"🚀 Processing dataset: {input_dir}")
    print(f"📂 Results will be saved to: {output_dir}")
    print("="*50)

    images_out_dir = os.path.join(output_dir, "images")
    labels_out_dir = os.path.join(output_dir, "labels")
    transforms_out_dir = os.path.join(output_dir, "transforms")
    
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)
    os.makedirs(transforms_out_dir, exist_ok=True)

    subject_ids = []
    if os.path.exists(input_dir):
        for filename in os.listdir(input_dir):
            if re.match(r'^\d+\.nii\.gz$', filename):
                sub_id = filename.split('.')[0]
                subject_ids.append(sub_id)
                
    subject_ids.sort(key=lambda x: int(x))

    if not subject_ids:
        print(f"❌ No label files with numeric names found in {input_dir}!")
        return

    print(f"🔍 Found {len(subject_ids)} subjects; processing in parallel with {max_workers or 'automatically allocated'} processes...")

    tasks = [
        (sub_id, input_dir, images_out_dir, labels_out_dir, transforms_out_dir, template_path) 
        for sub_id in subject_ids
    ]

    success_count = 0
    skip_count = 0
    fail_records = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_subject, tasks), 
            total=len(tasks), 
            desc="Registration progress"
        ))

    for sub_id, status in results:
        if "Success" in status:
            success_count += 1
        elif "Skipped" in status:
            skip_count += 1
        else:
            fail_records.append((sub_id, status))

    print("\n📊 Batch processing report:")
    print(f"✅ Successfully processed: {success_count}")
    print(f"⏭️ Skipped existing results: {skip_count}")
    if fail_records:
        print(f"❌ Failed or raised exceptions: {len(fail_records)}")
        for sub, err in fail_records:
            print(f"   - {sub}: {err}")


def main():
    TEMPLATE_PATH = "/path/to/data/oriCUZHONG/spgr.nii"
    
    MAX_WORKERS = 16 
    
    tasks = [
        (
            "/path/to/data/WMH/testdata",
            "/path/to/data/TEST/newWMHtest"
        ),
        (
            "/path/to/data/WMH/traindata",
            "/path/to/data/TEST/newWMHtrain"
        )
    ]

    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"Template not found at: {TEMPLATE_PATH}")

    for input_dir, output_dir in tasks:
        process_dataset_multiprocess(input_dir, output_dir, TEMPLATE_PATH, max_workers=MAX_WORKERS)

    print("\n🎉 Parallel processing completed for all datasets!")

if __name__ == "__main__":
    main()