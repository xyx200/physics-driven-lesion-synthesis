import os
import numpy as np
import torch
import nibabel as nib
import torch.multiprocessing as mp
import time
import sys
import datetime
from config_randomizer import Config

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
LOG_FILE = "./gpu_generation_random.log"

def write_log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pid = os.getpid()
    log_msg = f"[{timestamp}] [PID:{pid}] {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding='utf-8') as f:
            f.write(log_msg)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    from FluidAnomaly.DiffEqs.pde import AdvDiffPDE
    from FluidAnomaly.DiffEqs.adjoint import odeint_adjoint as odeint
    from FluidAnomaly.perlin3d import generate_velocity_3d, generate_shape_3d
except ImportError:
    write_log("Error: Please ensure the 'FluidAnomaly' folder is in the current directory.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class LesionGenerator:
    def __init__(self, config):
        self.cfg = config
        self.device = config.device
        
    def setup_pde(self):
        self.t = torch.from_numpy(np.arange(self.cfg.max_nt + 1) * self.cfg.dt).to(self.device)
        with torch.no_grad():
            self.adv_pde = AdvDiffPDE(
                data_spacing=[1., 1., 1.],
                perf_pattern='adv_diff',
                V_type='vector_div_free',
                D_type='constant',
                V_dict={},
                D_dict={},
                BC=self.cfg.bc,
                dt=self.cfg.dt,
                device=self.device
            )

    def generate(self):
        with torch.no_grad():
            self.setup_pde() 
            current_shape = self.cfg.target_shape
            
            # Padding
            res = self.cfg.perlin_res
            pad_shape = [int(np.ceil(s / r) * r) for s, r in zip(current_shape, res)]
            
            # Shape
            percentile = np.random.uniform(self.cfg.mask_percentile_min, self.cfg.mask_percentile_max)
            _, Pp_pad = generate_shape_3d(pad_shape, self.cfg.perlin_res, percentile, self.device)
            
            # Velocity & Diffusion
            self.adv_pde.V_dict = generate_velocity_3d(pad_shape, self.cfg.perlin_res, self.cfg.V_multiplier, self.device)
            self.adv_pde.D_dict = {'D': self.cfg.D_multiplier}
            
            # Evolve
            nt = self.cfg.max_nt
            if nt > 0:
                Pprob_evolved_pad = odeint(self.adv_pde, Pp_pad[None], self.t[:nt], 
                                        self.cfg.dt, 
                                        method=self.cfg.integ_method)[-1, 0]
            else:
                Pprob_evolved_pad = Pp_pad
            
            # Crop
            Pprob_evolved = Pprob_evolved_pad[:current_shape[0], :current_shape[1], :current_shape[2]]
            prob_map = torch.clamp(Pprob_evolved, 0, 1)
            return prob_map.cpu().numpy()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def process_single_sample(task_args):
    idx, output_dir = task_args
    task_id = f"Random_{idx:03d}"
    torch.set_num_threads(1)
    
    try:
        local_cfg = Config(use_gpu=True)
        seed_base = int(time.time() * 1000) % 2**32 + idx
        np.random.seed(seed_base)
        generator = LesionGenerator(local_cfg)
    except Exception as e:
        return f"[Init Error] {task_id}: {e}"

    global_attempt = 0
    
    while True:
        global_attempt += 1
        if global_attempt > 30: 
            err_msg = f"Skipped {task_id} after global attempts."
            write_log(f"[{task_id}] {err_msg}")
            return err_msg

        np.random.seed(seed_base + global_attempt * 999)
        local_cfg.randomize() 
        
        mean_res = np.mean(local_cfg.perlin_res)
        current_min_vol_ratio = 0.002 if mean_res > 15 else 0.005
        current_max_vol_ratio = 0.015 if mean_res > 15 else 0.03

        for fine_tune_step in range(10):
            try:
                temp_prob = generator.generate() 

                max_val_raw = temp_prob.max()
                min_val_raw = temp_prob.min()
                
                if max_val_raw - min_val_raw > 1e-8:
                    temp_prob_norm = (temp_prob - min_val_raw) / (max_val_raw - min_val_raw)
                else:
                    temp_prob_norm = np.zeros_like(temp_prob)

                lesion_voxels = np.sum(temp_prob_norm > 0.5)
                current_ratio = lesion_voxels / temp_prob_norm.size
                
                is_valid_ratio = current_min_vol_ratio <= current_ratio <= current_max_vol_ratio
                
                if max_val_raw >= 0.1 and is_valid_ratio:
                    identity_affine = np.eye(4)
                    mask_binary = (temp_prob_norm > 0.5).astype(np.uint8)
                    
                    filename_nii = os.path.join(output_dir, f"{task_id}.nii.gz")
                    filename_npy = os.path.join(output_dir, f"{task_id}.npy")
                    
                    nib.save(nib.Nifti1Image(mask_binary, identity_affine), filename_nii) 
                    np.save(filename_npy, mask_binary)
                    
                    write_log(f"Saved: {task_id} (fine-tuning:{fine_tune_step}, ratio:{current_ratio:.5f}, limits:[{current_min_vol_ratio}, {current_max_vol_ratio}])")
                    
                    del temp_prob, temp_prob_norm, mask_binary
                    torch.cuda.empty_cache()
                    return f"OK: {task_id}"
                
                else:
                    if fine_tune_step < 9:
                        step_size = np.random.uniform(3.0, 6.0)
                        
                        if current_ratio < current_min_vol_ratio:
                            local_cfg.mask_percentile_min = max(0.1, local_cfg.mask_percentile_min - step_size)
                            local_cfg.mask_percentile_max = max(1.0, local_cfg.mask_percentile_max - step_size)
                            
                        elif current_ratio > current_max_vol_ratio:
                            local_cfg.mask_percentile_min = min(98.0, local_cfg.mask_percentile_min + step_size)
                            local_cfg.mask_percentile_max = min(99.9, local_cfg.mask_percentile_max + step_size)
                            
                        continue
                    else:
                        write_log(f"[{task_id}] Shape abandoned (Res:{local_cfg.perlin_res}, Ratio:{current_ratio:.5f}); trying a new shape...")
            
            except Exception as e:
                err_str = str(e)
                if "out of memory" in err_str.lower():
                    torch.cuda.empty_cache()
                    time.sleep(0.5)
                    break 
                
                write_log(f"[{task_id}] PDE failed: {err_str} -> trying new parameters")
                torch.cuda.empty_cache()
                break

# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(f"\n=== Robust random generation (resume mode) started at {datetime.datetime.now()} ===\n")
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    output_dir = "/path/to/assets/morphology"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==========================================
    START_INDEX = 1127   
    END_INDEX   = 3001  
    
    tasks = [(i, output_dir) for i in range(START_INDEX, END_INDEX)]
            
    num_processes = 5 
    
    write_log("=" * 60)
    write_log(f"Starting robust generation (resume mode)")
    write_log(f"Task range: Random_{START_INDEX:03d} to Random_{END_INDEX-1:03d}")
    write_log(f"Tasks in this run: {len(tasks)}")
    write_log("=" * 60)

    start_time = time.time()
    
    with mp.Pool(processes=num_processes, maxtasksperchild=5) as pool:
        for result in pool.imap_unordered(process_single_sample, tasks):
            if "Skipped" in result:
                write_log(f"!!! Final failure: {result}")
            elif "Init Error" in result:
                write_log(f"!!! Initialization failed: {result}")

    end_time = time.time()
    write_log(f"Completed! Elapsed time: {end_time - start_time:.2f} seconds")
    print(f"Run completed. Generation range: {START_INDEX}-{END_INDEX}; log: {LOG_FILE}")