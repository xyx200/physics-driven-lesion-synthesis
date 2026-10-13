import os
import glob
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor
import time

# ==========================================
# ==========================================
PROB_ROOT = "/path/to/data/TEST/newWMHtrain/twolabel"
EXPERT_ROOT = "/path/to/data/TEST/newWMHtrain/labels"
OUTPUT_ROOT = "/path/to/data/TEST/newWMHtrain/finallabel"

DO_FLIP = True
FLIP_AXIS = 1 

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
            raise ValueError("Input model list is empty")
        
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

def load_nifti_prob(path):
    try:
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        data = np.moveaxis(data, -1, 0)
        return data
    except Exception as e:
        raise IOError(f"Cannot read NIfTI probability file {path}: {e}")

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
        case_id = filename.replace('.nii.gz', '')
        
        mod0_filename = f"{case_id}_0000_seg_aggregated_probs.nii.gz"
        mod1_filename = f"{case_id}_0001_seg_aggregated_probs.nii.gz"
        
        mod0_path = os.path.join(PROB_ROOT, mod0_filename)
        mod1_path = os.path.join(PROB_ROOT, mod1_filename)
        
        if not os.path.exists(mod0_path) or not os.path.exists(mod1_path):
            return f"[Skipped] ID {case_id}: missing corresponding 0000 or 0001 probability file"
            
        nii_obj = nib.load(expert_path)
        expert_data = nii_obj.get_fdata().astype(np.uint8) 
        expert_affine = nii_obj.affine
        expert_header = nii_obj.header
        
        mod0_prob = load_nifti_prob(mod0_path)      
        mod1_prob = load_nifti_prob(mod1_path)
        
        if DO_FLIP:
            mod0_prob = np.flip(mod0_prob, axis=FLIP_AXIS)
            mod1_prob = np.flip(mod1_prob, axis=FLIP_AXIS)

        if mod0_prob.shape[1:] != expert_data.shape:
            return f"[Error] ID {case_id}: shape mismatch Prob{mod0_prob.shape} vs Expert{expert_data.shape}"
        
        mod0_prob_mod = apply_expert_constraints(mod0_prob, expert_data, lesion_class_index=4)
        mod1_prob_mod = apply_expert_constraints(mod1_prob, expert_data, lesion_class_index=4)
        
        fusion_engine = SimpleEntropyFusion(ignore_background=False)
        _, fused_label = fusion_engine.fuse([mod0_prob_mod, mod1_prob_mod])
        
        fused_label_to_save = fused_label.astype(np.uint8)
        
        new_header = expert_header.copy()
        new_header.set_data_dtype(np.uint8)
        
        new_img = nib.Nifti1Image(fused_label_to_save, expert_affine, header=new_header)
        
        save_filename = f"{case_id}.nii.gz"
        save_path = os.path.join(OUTPUT_ROOT, save_filename)
        nib.save(new_img, save_path)
        
        return f"[Success] ID {case_id} -> {save_filename}"
        
    except Exception as e:
        return f"[Exception] Failed to process ID {case_id}: {str(e)}"

# ==========================================
# ==========================================
def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
        print(f"Creating output directory: {OUTPUT_ROOT}")

    expert_files = glob.glob(os.path.join(EXPERT_ROOT, "*.nii.gz"))
    total_files = len(expert_files)
    print(f"Found {total_files} expert label files. Preparing parallel processing...")
    print("-" * 50)

    with ProcessPoolExecutor(max_workers=2) as executor:
        results = executor.map(process_single_case, expert_files)
        
        for i, res in enumerate(results):
            print(f"[{i+1}/{total_files}] {res}")

    print("-" * 50)
    print("All tasks processed.")

if __name__ == "__main__":
    main()