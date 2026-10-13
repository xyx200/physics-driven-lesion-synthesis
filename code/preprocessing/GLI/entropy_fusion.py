import os
import glob
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor
import time
import re

# ==========================================
# ==========================================
NPZ_ROOT = "/path/to/data/Task002_Tumor/GILplus"
EXPERT_ROOT = "/path/to/data/wsl_data/LIT/DATA/brats2023/Goldlabel"
OUTPUT_ROOT = "/path/to/data/Task002_Tumor/GILfinal"
NUM_WORKERS = 8
LESION_CLASS_INDEX = 4 

EXCLUDE_CONFIG = {
    "0845": [0, 1, 3],
    "1138": [1],
    "1149": [0, 1, 2],
    "1158": [0, 2],
    "1182": [0, 1, 3]
}

# ==========================================
# ==========================================
class SimpleEntropyFusion:
    def __init__(self, epsilon=1e-8, ignore_background=False): 
        self.epsilon = epsilon
        self.ignore_background = ignore_background

    def _calculate_voxel_wise_entropy(self, probability_map):
        if self.ignore_background:
            target_probs = probability_map[1:, ...]
        else:
            target_probs = probability_map
        probs = np.clip(target_probs, self.epsilon, 1.0)
        entropy_map = -np.sum(probs * np.log(probs), axis=0)
        return entropy_map

    def fuse(self, model_outputs):
        if not model_outputs:
            raise ValueError("Input model list is empty (all modalities were excluded or not found)")
        
        stacked_probs = np.stack(model_outputs, axis=0)
        entropy_maps = [self._calculate_voxel_wise_entropy(p) for p in model_outputs]
        entropy_stack = np.stack(entropy_maps, axis=0)
        
        raw_weights = np.exp(-entropy_stack)
        sum_weights = np.sum(raw_weights, axis=0, keepdims=True)
        weights = raw_weights / (sum_weights + self.epsilon) 
        
        weights_expanded = weights[:, np.newaxis, ...]
        weighted_probs = weights_expanded * stacked_probs
        fused_prob = np.sum(weighted_probs, axis=0)
        fused_label = np.argmax(fused_prob, axis=0)
        return fused_prob, fused_label

# ==========================================
# ==========================================
def load_npz_prob(path):
    try:
        data = np.load(path)
        for key in ['softmax', 'arr_0', 'prob', 'vol_data']:
            if key in data:
                return data[key]
        return data[list(data.keys())[0]]
    except Exception as e:
        raise IOError(f"Cannot read NPZ file {path}: {e}")

def apply_expert_constraints(prob_map, expert_mask, lesion_class_index=4):
    modified_prob = prob_map.copy()
    lesion_mask_area = (expert_mask > 0)
    modified_prob[:, lesion_mask_area] = 0.0
    modified_prob[lesion_class_index, lesion_mask_area] = 1.0
    bg_mask_area = (expert_mask == 0)
    modified_prob[lesion_class_index, bg_mask_area] = 0.0
    current_sums = np.sum(modified_prob, axis=0)
    safe_sums = current_sums.copy()
    safe_sums[safe_sums == 0] = 1.0 
    modified_prob[:, bg_mask_area] /= safe_sums[bg_mask_area]
    return modified_prob

# ==========================================
# ==========================================
def process_single_case(expert_path):
    try:
        filename = os.path.basename(expert_path)
        
        base_name = filename.replace('.nii.gz', '')
        if '-' in base_name:
            case_id = base_name.split('-')[0]
        else:
            case_id = base_name
        
        # -------------------------------------------------
        # -------------------------------------------------
        nii_obj = nib.load(expert_path)
        expert_data_orig = nii_obj.get_fdata().astype(np.uint8) 
        expert_affine = nii_obj.affine
        expert_header = nii_obj.header
        
        expert_data = np.transpose(expert_data_orig, (2, 1, 0))

        # -------------------------------------------------
        # -------------------------------------------------
        prob_maps = []
        
        skip_list = EXCLUDE_CONFIG.get(case_id, [])
        
        for i in range(4):
            if i in skip_list:
                continue 

            npz_name = f"{case_id}_{i:04d}_aggregated_probs.npz"
            npz_path = os.path.join(NPZ_ROOT, npz_name)
            
            if not os.path.exists(npz_path):
                return f"[Error] ID {case_id}: missing file {npz_name}"
            
            prob = load_npz_prob(npz_path)
            if prob.shape[1:] != expert_data.shape:
                return f"[Error] Dimension mismatch: {npz_name}"
            
            prob_mod = apply_expert_constraints(prob, expert_data, lesion_class_index=LESION_CLASS_INDEX)
            prob_maps.append(prob_mod)

        if len(prob_maps) == 0:
            return f"[Warning] ID {case_id}: all modalities were excluded; fusion is unavailable"

        # -------------------------------------------------
        # -------------------------------------------------
        fusion_engine = SimpleEntropyFusion(ignore_background=False)
        _, fused_label = fusion_engine.fuse(prob_maps)
        
        fused_label_transposed = np.transpose(fused_label, (2, 1, 0))
        fused_label_to_save = fused_label_transposed.astype(np.uint8)
        
        new_header = expert_header.copy()
        new_header.set_data_dtype(np.uint8)
        
        save_filename = f"{case_id}.nii.gz"
        save_path = os.path.join(OUTPUT_ROOT, save_filename)
        
        new_img = nib.Nifti1Image(fused_label_to_save, expert_affine, header=new_header)
        nib.save(new_img, save_path)
        
        return f"[Success] {case_id} (modalities fused: {len(prob_maps)})"
        
    except Exception as e:
        return f"[Exception] ID {case_id if 'case_id' in locals() else filename}: {str(e)}"

# ==========================================
# ==========================================
def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_nii_files = glob.glob(os.path.join(EXPERT_ROOT, "*.nii.gz"))
    
    expert_files = []
    pattern = re.compile(r'^\d{4}(-seg)?\.nii\.gz$')

    for f in all_nii_files:
        fname = os.path.basename(f)
        if pattern.match(fname):
            expert_files.append(f)
    
    expert_files.sort()

    total_files = len(expert_files)
    print(f"NPZ source directory: {NPZ_ROOT}")
    print(f"Expert label directory: {EXPERT_ROOT}")
    print(f"Total files: {total_files}")
    
    print(f"Specific exclusion rules active: {len(EXCLUDE_CONFIG)} cases affected")
    print("-" * 50)
    
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = executor.map(process_single_case, expert_files)
        
        for i, res in enumerate(results):
            if "[Error]" in res or "[Exception]" in res or "[Warning]" in res or (i+1) % 50 == 0:
                print(f"[{i+1}/{total_files}] {res}")

    duration = time.time() - start_time
    print("-" * 50)
    print(f"All tasks completed! Elapsed time: {duration:.2f} seconds")

if __name__ == "__main__":
    main()