import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import torch.multiprocessing as mp
import time
import sys
import datetime
from config_random_true import Config

LOG_FILE = "./lesion_evolution.log"

def write_log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pid = os.getpid()
    log_msg = f"[{timestamp}] [PID:{pid}] {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding='utf-8') as f:
            f.write(log_msg)
    except Exception:
        pass

# =========================================================================
# =========================================================================
def center_crop_or_pad(data, target_shape=(147, 184, 147)):
    d, h, w = data.shape
    td, th, tw = target_shape
    
    pad_d = max(0, td - d)
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        data = np.pad(data, (
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2)
        ), mode='constant', constant_values=0.0)
        
    d, h, w = data.shape
    start_d = (d - td) // 2
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    
    cropped_data = data[
        start_d : start_d + td,
        start_h : start_h + th,
        start_w : start_w + tw
    ]
    
    return cropped_data

try:
    from FluidAnomaly.DiffEqs.pde import AdvDiffPDE
    from FluidAnomaly.DiffEqs.adjoint import odeint_adjoint as odeint
    from FluidAnomaly.perlin3d import generate_velocity_3d
except ImportError:
    write_log("Error: Please ensure the 'FluidAnomaly' folder is in the current directory.")
    sys.exit(1)

class LesionEvolver:
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

    def evolve(self, init_prob_map_tensor):
        with torch.no_grad():
            self.setup_pde() 
            
            current_shape = init_prob_map_tensor.shape
            self.cfg.target_shape = current_shape
            
            res = self.cfg.perlin_res
            pad_shape = [int(np.ceil(s / r) * r) for s, r in zip(current_shape, res)]
            
            pad_z = pad_shape[2] - current_shape[2]
            pad_y = pad_shape[1] - current_shape[1]
            pad_x = pad_shape[0] - current_shape[0]
            
            Pp_pad = F.pad(init_prob_map_tensor.unsqueeze(0), (0, pad_z, 0, pad_y, 0, pad_x), mode='constant', value=0.0)
            
            self.adv_pde.V_dict = generate_velocity_3d(pad_shape, self.cfg.perlin_res, self.cfg.V_multiplier, self.device)

            self.adv_pde.D_dict = {'D': self.cfg.D_multiplier}
            
            nt = self.cfg.max_nt
            if nt > 0:
                Pprob_evolved_pad = odeint(self.adv_pde, Pp_pad, self.t[:nt], 
                                        self.cfg.dt, 
                                        method=self.cfg.integ_method)[-1, 0]
            else:
                Pprob_evolved_pad = Pp_pad[0]
            
            Pprob_evolved = Pprob_evolved_pad[:current_shape[0], :current_shape[1], :current_shape[2]]
            prob_map = torch.clamp(Pprob_evolved, 0, 1)
            
            return prob_map.cpu().numpy()

def process_single_sample(task_args):
    input_path, output_dir = task_args
    basename = os.path.basename(input_path)
    
    TARGET_SHAPE = (147, 184, 147)
    MAX_OVERLAP_RATIO = 0.5 
    TARGET_NUM = 1
    
    try:
        img_nii = nib.load(input_path)
        raw_prob_map = img_nii.get_fdata().astype(np.float32)
        
        orig_prob_map = center_crop_or_pad(raw_prob_map, TARGET_SHAPE)
        
        affine = np.eye(4)
        
        orig_mask = orig_prob_map > 0.01
        orig_volume = np.sum(orig_mask)
        
        if orig_volume < 10:
            return f"Skipped {basename}: lesion in the central region is too small or missing (volume={orig_volume})"

        init_tensor = torch.from_numpy(orig_prob_map).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        local_cfg = Config(use_gpu=True)
        seed_base = int(time.time() * 1000) % 2**32 + hash(basename) % 10000
        np.random.seed(seed_base)
        evolver = LesionEvolver(local_cfg)
        
    except Exception as e:
        return f"[Init Error] {basename}: {e}"

    global_attempt = 0
    success_count = 0
    
    while success_count < TARGET_NUM:
        global_attempt += 1
        
        if global_attempt > 25: 
            err_msg = f"Partially abandoned {basename}: only {success_count}/{TARGET_NUM} samples generated after 75 attempts."
            write_log(err_msg)
            return err_msg

        np.random.seed(seed_base + global_attempt * 999)
        local_cfg.randomize() 

        try:
            temp_prob = evolver.evolve(init_tensor) 

            new_mask = (temp_prob > local_cfg.binarize_threshold).astype(np.uint8)
            new_volume = np.sum(new_mask)
            
            if new_volume < orig_volume * 0.3:
                continue
                
            intersection = np.sum((orig_mask == True) & (new_mask == 1))
            overlap_ratio = intersection / float(orig_volume)
            
            if overlap_ratio > MAX_OVERLAP_RATIO:
                write_log(f"[{basename}] Attempt {global_attempt}: overlap {overlap_ratio:.2f} > {MAX_OVERLAP_RATIO}; too close, retrying...")
                continue
            
            success_count += 1
            
            filename_nii = os.path.join(output_dir, f"evolved_{success_count}_{basename}")
            nib.save(nib.Nifti1Image(new_mask, affine), filename_nii) 
            
            write_log(f"✅ Extraction and evolution succeeded: {basename} (sample {success_count}/{TARGET_NUM}, attempt:{global_attempt}, overlap:{overlap_ratio:.3f}, volume change:{new_volume/orig_volume:.2f}x)")
            
            del temp_prob, new_mask
            torch.cuda.empty_cache()
            
        except Exception as e:
            err_str = str(e)
            if "out of memory" in err_str.lower():
                torch.cuda.empty_cache()
                time.sleep(0.5)
            continue

    del init_tensor
    torch.cuda.empty_cache()
    return f"OK: {basename} ({TARGET_NUM} samples generated successfully)"

if __name__ == "__main__":
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(f"\n=== Center-cropped lesion evolution started at {datetime.datetime.now()} ===\n")
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    input_dir = "/path/to/assets/real_probability"
    output_dir = "/path/to/assets/real_evolved_nifti"
    os.makedirs(output_dir, exist_ok=True)

    input_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    if not input_files:
        print("No input files found. Please check input_dir.")
        sys.exit(1)

    tasks = [(path, output_dir) for path in input_files]
            
    num_processes = 5
    
    write_log("=" * 60)
    write_log(f"Starting center cropping and evolution")
    write_log(f"Total tasks: {len(tasks)}")
    write_log("=" * 60)

    start_time = time.time()
    
    with mp.Pool(processes=num_processes, maxtasksperchild=5) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_sample, tasks)):
            if "Skipped" in result or "abandoned" in result:
                print(f"[{i+1}/{len(tasks)}] ⚠️ {result}")
            else:
                print(f"[{i+1}/{len(tasks)}] {result}")

    end_time = time.time()
    write_log(f"Completed! Elapsed time: {end_time - start_time:.2f} seconds")
    print(f"All evolution tasks completed. Outputs saved to: {output_dir}")