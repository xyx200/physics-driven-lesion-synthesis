import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.ndimage import binary_erosion, generate_binary_structure, distance_transform_edt, label, gaussian_filter, median_filter, convolve
import os
import glob
import random
import time


class ClassicalPoissonSolveError(RuntimeError):
    """Raised when a Classical Full-region Poisson solve is not trustworthy."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or []

class PoissonLesionAugmenter:
    blend_mode = "CLASSICAL_FULL_POISSON"

    def __init__(self, lesion_dir, noise_dir, p_per_sample=1.0, t1_idx=0, flair_idx=1):
        self.p_per_sample = p_per_sample
        self.t1_idx = t1_idx
        self.flair_idx = flair_idx
        self.last_solver_diagnostics = []
        self.last_apply_diagnostics = {"t1": [], "flair": []}
        self.last_sampled_parameters = None
        self.lesion_files = glob.glob(os.path.join(lesion_dir, "*.npy"))
        self.noise_files = glob.glob(os.path.join(noise_dir, "*.npy"))




    # ==========================================
    # ==========================================
    @staticmethod
    def build_3d_laplacian_sparse(shape):
        """
        Build the frozen Ours fixed-minus-six 3D discrete operator.
        """
        mz, my, mx = shape
        
        def make_1d_diff(n):
            if n == 1:
                return sp.csc_matrix([-2.0])
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
        """
        Classical Poisson editing over every complete connected-component mask.

        ``band_width``, ``alpha`` and ``sigma`` remain in the signature only so
        this deployment overlay can replace the legacy module without changing
        its caller. Classical mode deliberately does not use them.
        """
        del band_width, alpha, sigma

        src = np.asarray(src)
        dst_in = np.asarray(dst_in)
        mask = np.asarray(mask) > 0
        if src.shape != dst_in.shape or mask.shape != dst_in.shape:
            raise ValueError("src, dst_in and mask must have identical shapes")
        if dst_in.ndim != 3:
            raise ValueError("Classical Poisson blending requires 3D volumes")

        self.last_solver_diagnostics = []
        if not np.any(mask):
            return dst_in.copy()

        dst = dst_in.copy()
        struct = generate_binary_structure(3, 1)
        labeled_array, num_features = label(mask, structure=struct)

        for region_idx in range(1, num_features + 1):
            current_blob_mask = labeled_array == region_idx
            coords = np.argwhere(current_blob_mask)
            roi_start = np.maximum(coords.min(axis=0) - 2, 0)
            roi_stop = np.minimum(coords.max(axis=0) + 3, dst.shape)
            slices = tuple(slice(int(start), int(stop))
                           for start, stop in zip(roi_start, roi_stop))
            sub_src = np.asarray(src[slices], dtype=np.float64)
            # Match frozen Ours: every component reads the cumulative output.
            sub_dst = dst[slices].copy()
            sub_mask = current_blob_mask[slices]
            mask_flat = sub_mask.ravel()
            idx_in = np.flatnonzero(mask_flat)
            idx_out = np.flatnonzero(~mask_flat)

            diagnostic = {
                "component_id": int(region_idx),
                "voxel_count": int(idx_in.size),
                "roi_start": [int(value) for value in roi_start],
                "roi_stop": [int(value) for value in roi_stop],
                "roi_shape": [int(value) for value in sub_dst.shape],
                "cg_info": None,
                "iterations": 0,
                "relative_residual": None,
                "guidance_mode": "source_gradient",
                "laplacian_boundary_mode": "legacy_fixed_minus_6",
                "source_divergence_mode": "legacy_reflect_convolution",
                "target_boundary_mode": "cumulative_dst",
                "touches_volume_boundary": bool(
                    np.any(coords == 0)
                    or np.any(coords == (np.asarray(dst.shape) - 1))
                ),
                "dirichlet_edge_count": 0,
                "runtime_ms": None,
                "status": "running",
            }
            solve_started = time.perf_counter()

            try:
                laplacian = self.build_3d_laplacian_sparse(sub_dst.shape).tocsr()
                matrix_in = laplacian[idx_in, :][:, idx_in]
                matrix_in_out = laplacian[idx_in, :][:, idx_out]
                diagnostic["dirichlet_edge_count"] = int(matrix_in_out.nnz)

                lap_kernel = np.zeros((3, 3, 3), dtype=np.float64)
                lap_kernel[1, 1, 1] = -6.0
                lap_kernel[0, 1, 1] = lap_kernel[2, 1, 1] = 1.0
                lap_kernel[1, 0, 1] = lap_kernel[1, 2, 1] = 1.0
                lap_kernel[1, 1, 0] = lap_kernel[1, 1, 2] = 1.0
                div_g = convolve(sub_src, lap_kernel, mode='reflect')

                b_source = div_g.ravel()[idx_in]
                b_boundary = -matrix_in_out.dot(sub_dst.ravel()[idx_out])
                system_matrix = -matrix_in
                rhs = -(b_source + b_boundary)

                def count_iteration(_):
                    diagnostic["iterations"] += 1

                solution, info = cg(
                    system_matrix,
                    rhs,
                    rtol=1e-5,
                    atol=0.0,
                    maxiter=500,
                    callback=count_iteration,
                )
                residual = system_matrix.dot(solution) - rhs
                rhs_norm = float(np.linalg.norm(rhs))
                relative_residual = float(
                    np.linalg.norm(residual) / max(rhs_norm, np.finfo(np.float64).eps)
                )
                solution_finite = bool(np.all(np.isfinite(solution)))
                diagnostic["cg_info"] = int(info)
                diagnostic["solution_finite"] = solution_finite
                diagnostic["relative_residual"] = (
                    relative_residual if np.isfinite(relative_residual) else None
                )

                if info != 0:
                    raise ClassicalPoissonSolveError(
                        f"CG did not converge for component {region_idx}: info={info}",
                        self.last_solver_diagnostics + [diagnostic],
                    )
                if not solution_finite:
                    raise ClassicalPoissonSolveError(
                        f"CG returned non-finite values for component {region_idx}",
                        self.last_solver_diagnostics + [diagnostic],
                    )
                if not np.isfinite(relative_residual) or relative_residual > 1e-5:
                    raise ClassicalPoissonSolveError(
                        "CG residual exceeded 1e-5 for component "
                        f"{region_idx}: {relative_residual}",
                        self.last_solver_diagnostics + [diagnostic],
                    )

                sub_result = sub_dst.ravel().copy()
                sub_result[idx_in] = solution
                sub_result = sub_result.reshape(sub_dst.shape)
                dst_view = dst[slices]
                dst_view[sub_mask] = sub_result[sub_mask]
                diagnostic["status"] = "converged"
            except ClassicalPoissonSolveError:
                diagnostic["status"] = "failed"
                raise
            except Exception as exc:
                diagnostic["status"] = "failed"
                diagnostic["exception_type"] = type(exc).__name__
                diagnostic["exception_message"] = str(exc)
                raise ClassicalPoissonSolveError(
                    f"Classical Poisson solve failed for component {region_idx}: {exc}",
                    self.last_solver_diagnostics + [diagnostic],
                ) from exc
            finally:
                diagnostic["runtime_ms"] = float(
                    (time.perf_counter() - solve_started) * 1000.0
                )
                self.last_solver_diagnostics.append(diagnostic)

        if not np.array_equal(dst[~mask], dst_in[~mask], equal_nan=True):
            raise ClassicalPoissonSolveError(
                "Classical Poisson solve changed voxels outside the fusion mask",
                self.last_solver_diagnostics,
            )
        return dst

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
    def apply(self, data, seg, case_id=None):
        """
        Execute one Classical Full-region Poisson augmentation.

        All legacy random draws are retained in their original order. Values
        that belong only to the improved method are retained but unused.
        """
        gate_draw = float(np.random.uniform())
        self.last_apply_diagnostics = {"t1": [], "flair": []}
        self.last_solver_diagnostics = []
        self.last_sampled_parameters = None

        if gate_draw > self.p_per_sample:
            return data, seg

        if not self.lesion_files or not self.noise_files:
            exc = RuntimeError(
                "Classical Poisson augmentation requires at least one lesion and noise asset"
            )
            raise exc

        data_aug = data.copy()
        seg_aug = seg.copy()

        if data_aug.shape[0] <= max(self.t1_idx, self.flair_idx):
            exc = ValueError(
                "Input data does not contain the configured T1 and FLAIR channels"
            )
            raise exc
        if seg_aug.ndim < 4 or seg_aug.shape[0] == 0:
            exc = ValueError("seg must contain a 3D label volume in channel 0")
            raise exc

        t1_vol = data_aug[self.t1_idx]
        flair_vol = data_aug[self.flair_idx]
        label_vol = seg_aug[0]

        try:
            lesion_path = os.fspath(np.random.choice(self.lesion_files))
            noise_path = os.fspath(np.random.choice(self.noise_files))
            raw_lesion = np.load(lesion_path).astype(np.float32)
            raw_noise = np.load(noise_path).astype(np.float32)

            mri_shape = t1_vol.shape
            aligned_lesion = self.align_to_mri_shape(mri_shape, raw_lesion)
            aligned_noise = self.align_to_mri_shape(mri_shape, raw_noise)

            # Keep the exact legacy global NumPy RNG draw order.
            t1_lp = float(np.random.uniform(0.5, 3))
            t1_up = float(np.random.uniform(93.0, 97.0))
            t1_alpha = float(np.random.uniform(0.8, 0.90))
            fl_lp = float(np.random.uniform(3.0, 7.0))
            fl_up = float(np.random.uniform(97.0, 99.5))
            fl_alpha = float(np.random.uniform(0.92, 0.98))
            rand_bw = int(np.random.randint(2, 6))
            rand_sigma = float(np.random.uniform(0.5, 1.1))
            rand_y0_percentile = float(np.random.uniform(12, 18))
            rand_hard_threshold = float(np.random.uniform(0.45, 0.55))
            sampled_parameters = {
                "t1_lower_percentile": t1_lp,
                "t1_upper_percentile": t1_up,
                "t1_alpha_legacy_unused": t1_alpha,
                "flair_lower_percentile": fl_lp,
                "flair_upper_percentile": fl_up,
                "flair_alpha_legacy_unused": fl_alpha,
                "band_width_legacy_unused": rand_bw,
                "small_lesion_sigma_legacy_unused": rand_sigma,
                "label_center_percentile": rand_y0_percentile,
                "label_hard_threshold": rand_hard_threshold,
            }
            self.last_sampled_parameters = sampled_parameters

            fusion_mask = self.get_valid_fusion_mask(label_vol, aligned_lesion)
            fusion_voxels = int(np.count_nonzero(fusion_mask))
            if fusion_voxels == 0:
                return data, seg

            current_modality = "t1"
            t1_min, t1_max = self.get_lesion_percentile_bounds(
                label_vol, t1_vol, is_t1=True, label_val=4, 
                lower_p=t1_lp, upper_p=t1_up)
            noise_t1 = self.adjust_noise_intensity(aligned_noise, t1_min, t1_max)
            final_t1 = self.poisson_edit_3d_fast(
                noise_t1, t1_vol, fusion_mask, band_width=rand_bw, alpha=t1_alpha, sigma=rand_sigma)
            self.last_apply_diagnostics["t1"] = [
                dict(item) for item in self.last_solver_diagnostics
            ]

            current_modality = "flair"
            self.last_solver_diagnostics = []
            flair_original = flair_vol.copy()
            fl_min, fl_max = self.get_lesion_percentile_bounds(
                label_vol, flair_vol, is_t1=False, label_val=4, 
                lower_p=fl_lp, upper_p=fl_up)
            noise_fl = self.adjust_noise_intensity(aligned_noise, fl_min, fl_max)
            final_flair = self.poisson_edit_3d_fast(
                noise_fl, flair_vol, fusion_mask, band_width=rand_bw, alpha=fl_alpha, sigma=rand_sigma)
            self.last_apply_diagnostics["flair"] = [
                dict(item) for item in self.last_solver_diagnostics
            ]

            if not np.array_equal(
                    final_t1[~fusion_mask], t1_vol[~fusion_mask], equal_nan=True):
                raise ClassicalPoissonSolveError(
                    "T1 output changed outside the fusion mask",
                    self.last_apply_diagnostics["t1"],
                )
            if not np.array_equal(
                    final_flair[~fusion_mask], flair_vol[~fusion_mask], equal_nan=True):
                raise ClassicalPoissonSolveError(
                    "FLAIR output changed outside the fusion mask",
                    self.last_apply_diagnostics["flair"],
                )

            # Preserve the legacy pseudo-label computation order exactly.
            diff_vol = np.maximum(final_flair - flair_original, 0)
            valid_diffs = diff_vol[fusion_mask > 0]
            active_diffs = np.empty(0, dtype=diff_vol.dtype)
            final_pseudo_mask = np.zeros_like(fusion_mask, dtype=bool)

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

            # Commit both modalities only after both solves and labels succeed.
            data_aug[self.t1_idx] = final_t1
            data_aug[self.flair_idx] = final_flair

        except Exception as exc:
            if "current_modality" in locals():
                self.last_apply_diagnostics[current_modality] = [
                    dict(item) for item in self.last_solver_diagnostics
                ]
            raise

        return data_aug.astype(np.float32), seg_aug.astype(np.float32)
