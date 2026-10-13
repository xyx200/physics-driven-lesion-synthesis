import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.ndimage import convolve, binary_erosion, generate_binary_structure, distance_transform_edt, label, gaussian_filter, median_filter
import os
import glob
import random

class PoissonLesionAugmenter:
    def __init__(self, lesion_dir, noise_dir, p_per_sample=1.0, t1_idx=0, flair_idx=1):
        self.p_per_sample = p_per_sample
        self.t1_idx = t1_idx
        self.flair_idx = flair_idx
        
        self.lesion_files = glob.glob(os.path.join(lesion_dir, "*.npy"))
        self.noise_files = glob.glob(os.path.join(noise_dir, "*.npy"))
        
        if not self.lesion_files:
            print(f"Warning: No lesion files found in {lesion_dir}")
        if not self.noise_files:
            print(f"Warning: No noise files found in {noise_dir}")

    # ==========================================
    # ==========================================
    @staticmethod
    def build_3d_laplacian_sparse(shape):
        mz, my, mx = shape
        
        def make_1d_diff(n):
            if n == 1: return sp.csc_matrix([-2.0])
            return sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csc')

        Dz = make_1d_diff(mz)
        Dy = make_1d_diff(my)
        Dx = make_1d_diff(mx)
        
        Iz = sp.eye(mz, format='csc')
        Iy = sp.eye(my, format='csc')
        Ix = sp.eye(mx, format='csc')
        
        L3d = sp.kron(Dz, sp.kron(Iy, Ix)) + \
              sp.kron(Iz, sp.kron(Dy, Ix)) + \
              sp.kron(Iz, sp.kron(Iy, Dx))
              
        return L3d

    # ==========================================
    # ==========================================
    @staticmethod
    def align_to_mri_shape(mri_shape, input_data):
        target_d, target_h, target_w = mri_shape
        src_d, src_h, src_w = input_data.shape
        
        aligned_data = np.zeros(mri_shape, dtype=input_data.dtype)
        
        ct_d, ct_h, ct_w = target_d // 2, target_h // 2, target_w // 2
        cs_d, cs_h, cs_w = src_d // 2, src_h // 2, src_w // 2
        
        len_d = min(target_d, src_d)
        td_s = ct_d - len_d // 2
        td_e = td_s + len_d
        sd_s = cs_d - len_d // 2
        sd_e = sd_s + len_d
        
        len_h = min(target_h, src_h)
        th_s = ct_h - len_h // 2
        th_e = th_s + len_h
        sh_s = cs_h - len_h // 2
        sh_e = sh_s + len_h

        len_w = min(target_w, src_w)
        tw_s = ct_w - len_w // 2
        tw_e = tw_s + len_w
        sw_s = cs_w - len_w // 2
        sw_e = sw_s + len_w
        
        aligned_data[td_s:td_e, th_s:th_e, tw_s:tw_e] = \
            input_data[sd_s:sd_e, sh_s:sh_e, sw_s:sw_e]
            
        return aligned_data

    @staticmethod
    def get_lesion_percentile_bounds(label_data, image_data, is_t1, label_val=4, 
                                     lower_p=None, upper_p=None):
        mask = label_data == label_val
        
        if np.sum(mask) == 0:
            valid_voxels = image_data[label_data > 0]
            if len(valid_voxels) == 0: return 0.0, 1.0
            mean_val = np.mean(valid_voxels)
            std_val = np.std(valid_voxels)
            
            if is_t1:
                return mean_val - 2.5 * std_val, mean_val - 1.0 * std_val
            else:
                return mean_val + 1.0 * std_val, mean_val + 3.0 * std_val

        vals = image_data[mask]
        
        if is_t1:
            lp = lower_p if lower_p is not None else 1
            up = upper_p if upper_p is not None else 95
        else:
            lp = lower_p if lower_p is not None else 5
            up = upper_p if upper_p is not None else 99
            
        lower_bound = np.percentile(vals, lp)
        upper_bound = np.percentile(vals, up)
        
        return lower_bound, upper_bound

    @staticmethod
    def adjust_noise_intensity(noise_vol, target_min, target_max):
        noise_min, noise_max = noise_vol.min(), noise_vol.max()
        
        if noise_max - noise_min > 1e-6:
            norm_noise = (noise_vol - noise_min) / (noise_max - noise_min)
        else:
            norm_noise = np.zeros_like(noise_vol)

        scaled_noise = norm_noise * (target_max - target_min) + target_min
        return scaled_noise

    def poisson_edit_3d_fast(self, src, dst_in, mask, band_width=3, alpha=1.0, sigma=0.8):
        if not np.any(mask): return dst_in
        dst = dst_in.copy()
        
        struct = generate_binary_structure(3, 1)
        labeled_array, num_features = label(mask > 0, structure=struct)
        
        if num_features == 0: return dst

        eroded_global = binary_erosion(mask > 0, structure=struct, iterations=band_width)
        surviving_ids = np.unique(labeled_array[eroded_global])
        surviving_ids = surviving_ids[surviving_ids != 0]

        is_large = np.isin(labeled_array, surviving_ids)
        mask_small = (mask > 0) & (~is_large) 
        mask_large = (mask > 0) & (is_large)  

        if np.any(mask_small):
            soft_mask = mask_small.astype(np.float32)
            soft_mask = gaussian_filter(soft_mask, sigma=sigma)
            blend_weights = soft_mask * alpha
            active_region = blend_weights > 1e-3
            
            if np.any(active_region):
                src_pixels = src[active_region]
                dst_pixels = dst[active_region]
                weights = blend_weights[active_region]
                dst[active_region] = dst_pixels + weights * (src_pixels - dst_pixels)

        if not np.any(mask_large):
            return dst

        
        large_blobs_labeled, num_large_blobs = label(mask_large, structure=struct)
        
        for region_idx in range(1, num_large_blobs + 1):
            current_blob_mask = (large_blobs_labeled == region_idx)
            
            
            coords = np.argwhere(current_blob_mask)
            
            pad = 2 
            min_z, min_y, min_x = np.maximum(coords.min(axis=0) - pad, 0)
            max_z, max_y, max_x = np.minimum(coords.max(axis=0) + pad + 1, dst.shape)
            
            sub_src = src[min_z:max_z, min_y:max_y, min_x:max_x]
            sub_dst = dst[min_z:max_z, min_y:max_y, min_x:max_x]
            sub_mask = current_blob_mask[min_z:max_z, min_y:max_y, min_x:max_x]
            
            h, w, d = sub_dst.shape

            inner_mask = binary_erosion(sub_mask, structure=struct, iterations=band_width)
            band_mask = (sub_mask > 0) & (~inner_mask)
            band_flat = band_mask.flatten()

            if not np.any(band_flat):
                roi_res = sub_dst.copy()
                roi_res[sub_mask > 0] = sub_src[sub_mask > 0] * alpha + \
                                        sub_dst[sub_mask > 0] * (1 - alpha)
                dst[min_z:max_z, min_y:max_y, min_x:max_x] = roi_res
                continue

            composite_bg = sub_dst.copy()
            core_src = sub_src[inner_mask]
            core_dst = sub_dst[inner_mask]
            composite_bg[inner_mask] = core_src * alpha + core_dst * (1 - alpha)


            L = self.build_3d_laplacian_sparse((h, w, d)).tocsr()
            
            idx_band = np.flatnonzero(band_flat)
            idx_boundary = np.flatnonzero(~band_flat)

            A = L[idx_band, :][:, idx_band]
            # ==============================================================

            lap_kernel = np.zeros((3, 3, 3))
            lap_kernel[1, 1, 1] = -6.0
            lap_kernel[0, 1, 1] = lap_kernel[2, 1, 1] = 1.0
            lap_kernel[1, 0, 1] = lap_kernel[1, 2, 1] = 1.0
            lap_kernel[1, 1, 0] = lap_kernel[1, 1, 2] = 1.0
            
            div_g = convolve(sub_src.astype(float), lap_kernel, mode='reflect')
            
            b_source = div_g.flatten()[idx_band] 

            u_out = composite_bg.flatten()[idx_boundary]
            L_in_out = L[idx_band, :][:, idx_boundary]
            # ======================================
            
            b_boundary = -L_in_out.dot(u_out)
            b_total = b_source + b_boundary

            x, info = cg(-A, -b_total, rtol=1e-5, atol=0.0, maxiter=500)
            
            res_sub = composite_bg.flatten()
            res_sub[band_flat] = x
            res_sub = res_sub.reshape((h, w, d))
            
            dst[min_z:max_z, min_y:max_y, min_x:max_x] = res_sub
        
        return dst.astype(np.float32)

    @staticmethod
    def get_valid_fusion_mask(label_vol, pseudo_lesion_mask):
        brain_mask = label_vol > 0
        dist_map = distance_transform_edt(brain_mask)
        boundary_safe_mask = dist_map > 2
        
        not_existing_lesion = label_vol != 4
        
        valid_brain_area = brain_mask & boundary_safe_mask & not_existing_lesion
        final_fusion_mask = (pseudo_lesion_mask > 0.5) & valid_brain_area
        
        return final_fusion_mask

    # ==========================================
    # ==========================================
    def apply(self, data, seg):
        if (np.random.uniform() > self.p_per_sample) or \
           (not self.lesion_files) or \
           (not self.noise_files):
            return data, seg

        data_aug = data.copy()
        seg_aug = seg.copy()
        
        if data_aug.shape[0] <= max(self.t1_idx, self.flair_idx):
            return data, seg
            
        t1_vol = data_aug[self.t1_idx]
        flair_vol = data_aug[self.flair_idx]
        label_vol = seg_aug[0]

        try:
            lesion_path = np.random.choice(self.lesion_files)
            noise_path = np.random.choice(self.noise_files)
            raw_lesion = np.load(lesion_path).astype(np.float32)
            raw_noise = np.load(noise_path).astype(np.float32)
        except Exception as e:
            return data, seg

        mri_shape = t1_vol.shape
        aligned_lesion = self.align_to_mri_shape(mri_shape, raw_lesion)
        aligned_noise = self.align_to_mri_shape(mri_shape, raw_noise)
        
        t1_lp = np.random.uniform(0.5, 3)
        t1_up = np.random.uniform(93.0, 97.0)
        t1_alpha = np.random.uniform(0.8, 0.90)
        
        fl_lp = np.random.uniform(3.0, 7.0)
        fl_up = np.random.uniform(97.0, 99.5)
        fl_alpha = np.random.uniform(0.92, 0.98)
        
        rand_bw = np.random.randint(2, 6)
        rand_sigma = np.random.uniform(0.5, 1.1)
        
        rand_y0_percentile = np.random.uniform(12, 18)
        rand_hard_threshold = np.random.uniform(0.45, 0.55)
        # ------------------------

        fusion_mask = self.get_valid_fusion_mask(label_vol, aligned_lesion)
        
        if np.sum(fusion_mask) == 0:
            return data, seg

        try:
            t1_min, t1_max = self.get_lesion_percentile_bounds(
                label_vol, t1_vol, is_t1=True, label_val=4, 
                lower_p=t1_lp, upper_p=t1_up)
            noise_t1 = self.adjust_noise_intensity(aligned_noise, t1_min, t1_max)
            final_t1 = self.poisson_edit_3d_fast(
                noise_t1, t1_vol, fusion_mask, band_width=rand_bw, alpha=t1_alpha, sigma=rand_sigma)
            data_aug[self.t1_idx] = final_t1

            flair_original = flair_vol.copy() 
            fl_min, fl_max = self.get_lesion_percentile_bounds(
                label_vol, flair_vol, is_t1=False, label_val=4, 
                lower_p=fl_lp, upper_p=fl_up)
            noise_fl = self.adjust_noise_intensity(aligned_noise, fl_min, fl_max)
            final_flair = self.poisson_edit_3d_fast(
                noise_fl, flair_vol, fusion_mask, band_width=rand_bw, alpha=fl_alpha, sigma=rand_sigma)
            data_aug[self.flair_idx] = final_flair
            
            diff_vol = np.maximum(final_flair - flair_original, 0)
            valid_diffs = diff_vol[fusion_mask > 0] 

            if valid_diffs.size > 0:
                active_diffs = valid_diffs[valid_diffs > 1e-6]
                
                if active_diffs.size == 0:
                    y_0_adaptive = 0.0
                    diff_std = 0.0
                else:
                    y_0_adaptive = np.percentile(active_diffs, rand_y0_percentile) 
                    diff_std = np.std(active_diffs)
                
                if diff_std < 1e-8: 
                    k_adaptive = 100.0 
                else:
                    k_adaptive = 4.0 / diff_std

                y_binary_base = (diff_vol > 1e-7).astype(np.float32)
                y_continuous = diff_vol
                
                exponent = -k_adaptive * (y_continuous - y_0_adaptive)
                exponent = np.clip(exponent, -50, 50) 
                y_logistic_raw = y_binary_base / (1 + np.exp(exponent))

                y_logistic_filtered = median_filter(y_logistic_raw, size=3)

                final_pseudo_mask = y_logistic_filtered > rand_hard_threshold
                final_pseudo_mask = final_pseudo_mask & (fusion_mask > 0)
                
                seg_aug[0][final_pseudo_mask] = 4
                
        except Exception as e:
            print(f"Poisson fusion failed: {e}")
            import traceback
            traceback.print_exc()
            return data, seg

        return data_aug.astype(np.float32), seg_aug.astype(np.float32)