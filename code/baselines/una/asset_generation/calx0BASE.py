import os
import numpy as np
import torch
import nibabel as nib
import torch.multiprocessing as mp
import time
import sys
import datetime

from config_randomizerBASE import Config

LOG_FILE = "./gpu_generation_baseline.log"

def write_log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pid = os.getpid()
    log_msg = f"[{timestamp}] [PID:{pid}] {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding='utf-8') as f:
            f.write(log_msg)
    except Exception:
        pass

try:
    from FluidAnomaly.DiffEqs.pde import AdvDiffPDE
    from FluidAnomaly.DiffEqs.adjoint import odeint_adjoint as odeint
    from FluidAnomaly.perlin3d import generate_velocity_3d, generate_shape_3d
except ImportError:
    write_log("Error: Please ensure the 'FluidAnomaly' folder is in the current directory.")
    sys.exit(1)



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
                D_dict={'D': self.cfg.D_multiplier}, 
                BC=self.cfg.bc,
                dt=self.cfg.dt,
                device=self.device
            )

    def generate(self):
        with torch.no_grad():
            self.setup_pde() 
            current_shape = self.cfg.target_shape
            res = self.cfg.perlin_res
            pad_shape = [int(np.ceil(s / r) * r) for s, r in zip(current_shape, res)]
            
            use_external_prob = False
            Pp_pad = None
            
            if len(self.cfg.prob_map_files) > 0 and np.random.rand() < 0.5:
                try:
                    chosen_file = np.random.choice(self.cfg.prob_map_files)
                    if chosen_file.endswith('.npy'):
                        ext_data = np.load(chosen_file)
                    else:
                        ext_data = nib.load(chosen_file).get_fdata()
                        
                    ext_data_aligned = align_to_mri_shape(current_shape, ext_data)
                    
                    pad_widths = [(0, p - c) for c, p in zip(current_shape, pad_shape)]
                    ext_data_padded = np.pad(ext_data_aligned, pad_widths, mode='constant', constant_values=0)
                    
                    Pp_pad = torch.from_numpy(ext_data_padded).float().to(self.device)
                    use_external_prob = True
                except Exception as e:
                    write_log(f"-> Failed to load external probability map: {e}; falling back to noise-only mode")
                    use_external_prob = False

            if not use_external_prob:
                percentile = np.random.uniform(self.cfg.mask_percentile_min, self.cfg.mask_percentile_max)
                _, Pp_pad = generate_shape_3d(pad_shape, self.cfg.perlin_res, percentile, self.device)
            
            self.adv_pde.V_dict = generate_velocity_3d(pad_shape, self.cfg.perlin_res, self.cfg.V_multiplier, self.device)
            
            nt = self.cfg.max_nt
            if nt > 0:
                Pprob_evolved_pad = odeint(self.adv_pde, Pp_pad[None], self.t[:nt], 
                                        self.cfg.dt, 
                                        method=self.cfg.integ_method)[-1, 0]
            else:
                Pprob_evolved_pad = Pp_pad
            
            Pprob_evolved = Pprob_evolved_pad[:current_shape[0], :current_shape[1], :current_shape[2]]
            prob_map = torch.clamp(Pprob_evolved, 0, 1)
            return prob_map.cpu().numpy()

def process_single_sample(task_args):
    idx, output_dir = task_args
    task_id = f"Baseline_{idx:03d}"
    torch.set_num_threads(1)
    
    try:
        local_cfg = Config(use_gpu=True)
        seed_base = int(time.time() * 1000) % 2**32 + idx
        generator = LesionGenerator(local_cfg)
    except Exception as e:
        return f"[Init Error] {task_id}: {e}"

    global_attempt = 0
    
    while True:
        global_attempt += 1
        if global_attempt > 30: 
            err_msg = f"Skipped {task_id} after 30 attempts due to strict tol."
            write_log(f"[{task_id}] {err_msg}")
            return err_msg

        np.random.seed(seed_base + global_attempt * 999)
        torch.manual_seed(seed_base + global_attempt * 999)
        
        try:
            temp_prob = generator.generate() 

            pathol_mean = temp_prob.mean()
            if pathol_mean < local_cfg.pathol_tol:
                continue
                
            prob_map_save = temp_prob.astype(np.float32)

            identity_affine = np.eye(4)
            filename_nii = os.path.join(output_dir, f"{task_id}.nii.gz")
            filename_npy = os.path.join(output_dir, f"{task_id}.npy")
            
            nib.save(nib.Nifti1Image(prob_map_save, identity_affine), filename_nii) 
            np.save(filename_npy, prob_map_save)
            
            write_log(f"Probability map saved: {task_id} (Attempt:{global_attempt}, Mean:{pathol_mean:.6f})")
            
            del temp_prob, prob_map_save
            torch.cuda.empty_cache()
            return f"OK: {task_id}"

        except Exception as e:
            err_str = str(e)
            if "out of memory" in err_str.lower():
                torch.cuda.empty_cache()
                time.sleep(0.5)
            continue

if __name__ == "__main__":
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(f"\n=== Baseline comparison started at {datetime.datetime.now()} ===\n")
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    output_dir = "/path/to/assets/una_probability"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    START_INDEX = 0   
    END_INDEX   = 1000  
    
    tasks = [(i, output_dir) for i in range(START_INDEX, END_INDEX)]
            
    num_processes = 5 
    
    dummy_cfg = Config(use_gpu=False)
    
    write_log("=" * 60)
    write_log(f"Starting original Baseline generation (saving continuous probability maps)")
    write_log(f"External probability map files found: {len(dummy_cfg.prob_map_files)}")
    write_log(f"Task range: Baseline_{START_INDEX:03d} to Baseline_{END_INDEX-1:03d}")
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
    print(f"Baseline run completed. Log: {LOG_FILE}")