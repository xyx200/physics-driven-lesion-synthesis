import numpy as np
import torch
import glob
import os

class Config:
    def __init__(self, use_gpu=True):
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.target_shape = [147, 184, 147]
        
        # ==========================================
        # ==========================================
        self.perlin_res = [2, 2, 2]
        self.mask_percentile_min = 90.0
        self.mask_percentile_max = 99.6
        self.integ_method = 'dopri5'
        self.bc = 'neumann'
        self.V_multiplier = 500
        self.dt = 0.1
        self.max_nt = 10
        
        self.pathol_thres = 0.2
        self.pathol_tol = 0.000001
        
        self.D_multiplier = 1.0
        
        # ==========================================
        # ==========================================
        self.prob_map_dir = "/path/to/assets/real_probability"
        self.prob_map_files = []
        if os.path.exists(self.prob_map_dir):
            self.prob_map_files.extend(glob.glob(os.path.join(self.prob_map_dir, "*.nii.gz")))
            self.prob_map_files.extend(glob.glob(os.path.join(self.prob_map_dir, "*.npy")))

    def randomize(self):
        pass

    def __repr__(self):
        return (f"Baseline_Cfg(Res={self.perlin_res}, V={self.V_multiplier}, "
                f"dt={self.dt}, Thres={self.pathol_thres}, D={self.D_multiplier})")