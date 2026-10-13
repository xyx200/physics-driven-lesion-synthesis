import numpy as np
import os
import glob

class UNALesionAugmenter:
    def __init__(self, 
                 prob_dir='/path/to/assets/una_probability', 
                 p_per_sample=1.0):
        self.p_per_sample = p_per_sample
        self.prob_dir = prob_dir
        self.prob_files = glob.glob(os.path.join(prob_dir, "*.npy"))
        
        if not self.prob_files:
            print(f"Warning: No probability maps found in {prob_dir}")

    @staticmethod
    def align_to_mri_shape(mri_shape, input_data):
        target_d, target_h, target_w = mri_shape
        src_d, src_h, src_w = input_data.shape
        aligned_data = np.zeros(mri_shape, dtype=input_data.dtype)
        
        ct_d, ct_h, ct_w = target_d // 2, target_h // 2, target_w // 2
        cs_d, cs_h, cs_w = src_d // 2, src_h // 2, src_w // 2
        
        len_d = min(target_d, src_d)
        td_s = ct_d - len_d // 2; td_e = td_s + len_d
        sd_s = cs_d - len_d // 2; sd_e = sd_s + len_d
        
        len_h = min(target_h, src_h)
        th_s = ct_h - len_h // 2; th_e = th_s + len_h
        sh_s = cs_h - len_h // 2; sh_e = sh_s + len_h

        len_w = min(target_w, src_w)
        tw_s = ct_w - len_w // 2; tw_e = tw_s + len_w
        sw_s = cs_w - len_w // 2; sw_e = sw_s + len_w
        
        aligned_data[td_s:td_e, th_s:th_e, tw_s:tw_e] = \
            input_data[sd_s:sd_e, sh_s:sh_e, sw_s:sw_e]
            
        return aligned_data

    @staticmethod
    def encode_pathology_strict(I0, Pprob, seg_map):
        I_min = np.min(I0)
        I_shifted = I0 - I_min
        
        gm_mask = (seg_map == 1) | (seg_map == 2)
        wm_mask = (seg_map == 3)
        
        if np.sum(wm_mask) == 0 or np.sum(gm_mask) == 0:
            mu_w = np.mean(I_shifted[seg_map > 0]) if np.sum(seg_map > 0) > 0 else np.mean(I_shifted)
            mu_g = mu_w 
        else:
            mu_w = np.mean(I_shifted[wm_mask])
            mu_g = np.mean(I_shifted[gm_mask])
            
        if mu_w > mu_g:
            dist_mean = -mu_w / 2.0
        else:
            dist_mean = mu_w / 2.0
            
        dist_std = mu_w / 2.0
        
        Omega_P = Pprob > 0
        num_lesion_voxels = np.sum(Omega_P)
        
        if num_lesion_voxels == 0:
            return I0
            
        Delta_I = np.zeros_like(I_shifted)
        Delta_I[Omega_P] = np.random.normal(loc=dist_mean, scale=dist_std, size=num_lesion_voxels)
        
        if np.random.rand() < 0.20:
            Delta_I = -Delta_I
            
        I_shifted_new = I_shifted + Delta_I * Pprob
        
        I_shifted_new = np.clip(I_shifted_new, 0.0, None)
        
        I_final = I_shifted_new + I_min
        
        return I_final

    def apply(self, data, seg):
        if (np.random.uniform() > self.p_per_sample) or (not self.prob_files):
            return data, seg

        data_aug = data.copy()
        seg_aug = seg.copy()
        label_vol = seg_aug[0]
        
        try:
            prob_path = np.random.choice(self.prob_files)
            raw_prob = np.load(prob_path)
            
            if raw_prob.dtype == np.uint8:
                raw_prob = raw_prob.astype(np.float32) / 255.0
            else:
                raw_prob = raw_prob.astype(np.float32)
                if raw_prob.max() > 1.0:
                    raw_prob = raw_prob / raw_prob.max()
            # =========================================================
            
            mri_shape = data_aug[0].shape
            Pprob_aligned = self.align_to_mri_shape(mri_shape, raw_prob)
            
            brain_mask = (label_vol >= 1) & (label_vol <= 7)
            not_existing_lesion = label_vol != 4
            valid_area = brain_mask & not_existing_lesion
            
            Pprob_aligned[~valid_area] = 0.0 
            
            max_p = Pprob_aligned.max()
            if max_p < 1e-5:
                return data, seg
                
            thres = 0.2 * max_p
            P_binary = Pprob_aligned > thres
            
            if not np.any(P_binary):
                return data, seg

            num_channels = data_aug.shape[0]
            for c in range(num_channels):
                I0 = data_aug[c]
                I_aug = self.encode_pathology_strict(I0, Pprob_aligned, label_vol)
                data_aug[c] = I_aug
                
            seg_aug[0][P_binary] = 4

        except Exception as e:
            print(f"UNA strict formula fusion failed: {e}")
            import traceback
            traceback.print_exc()
            return data, seg

        return data_aug.astype(np.float32), seg_aug.astype(np.float32)