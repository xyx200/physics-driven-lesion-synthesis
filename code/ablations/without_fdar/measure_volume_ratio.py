import os
import argparse
import glob
import numpy as np
import nibabel as nib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

def calculate_label_ratio(file_path):
    try:
        img = nib.load(file_path)
        
        data = np.asanyarray(img.dataobj)
        
        label_1_count = int(np.sum(data == 1))
        
        total_voxels = int(data.size)
        
        ratio = float(label_1_count / total_voxels) if total_voxels > 0 else 0.0
            
        return file_path, ratio, label_1_count, total_voxels
        
    except Exception as e:
        return file_path, None, None, str(e)

def main(folder_path, output_json_path):
    
    file_pattern = os.path.join(folder_path, '*.nii.gz')
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"No .nii.gz files found in {folder_path}.")
        return

    print(f"Found {len(file_list)} files. Starting parallel computation...")
    
    file_results = {}
    failed_files = {}
    
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(calculate_label_ratio, fp): fp for fp in file_list}
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            filename = os.path.basename(file_path)
            try:
                file_path, ratio, label_1_count, total_voxels = future.result()
                
                if ratio is not None:
                    file_results[filename] = {
                        "ratio": ratio,
                        "label_1_count": label_1_count,
                        "total_voxels": total_voxels
                    }
                    print(f"Processed: {filename} | Ratio: {ratio:.6f}")
                else:
                    error_msg = total_voxels 
                    failed_files[filename] = error_msg
                    print(f"Failed: {filename} | Error: {error_msg}")
                    
            except Exception as exc:
                print(f"Exception in file {filename}: {exc}")

    # =========================================
    # =========================================
    
    valid_ratios = [data["ratio"] for data in file_results.values()]
    
    average_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0.0
    
    final_output = {
        "summary": {
            "total_files_processed": len(file_list),
            "successful_count": len(file_results),
            "failed_count": len(failed_files),
            "average_ratio": average_ratio
        },
        "file_details": file_results,
        "failed_files": failed_files
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\nAll computations completed!")
    print(f"Successfully processed: {len(file_results)} files; mean ratio: {average_ratio:.6f}")
    print(f"Results saved to: {os.path.abspath(output_json_path)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate lesion voxel fractions in the shape library")
    parser.add_argument('directory')
    parser.add_argument('--output', default='volume_ratios.json')
    args = parser.parse_args()
    main(args.directory, args.output)
