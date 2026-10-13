import numpy as np
import torch

class Config:
    def __init__(self, use_gpu=True):
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.target_shape = [147, 184, 147]
        
        self.integ_method = 'dopri5' 
        self.bc = 'neumann'
        
        self.perlin_res = [4, 4, 4] 
        self.dt = 0.1
        self.mask_percentile_min = 90
        self.mask_percentile_max = 99.0
        self.V_multiplier = 500
        self.max_nt = 10
        
        self.D_multiplier = 1.0

    def randomize(self):
        # =====================================================
        # =====================================================
        valid_divisors = np.array([2, 4, 5, 8, 10, 20, 25])

        dice_shape = np.random.rand()
        
        if dice_shape < 0.4: 
            base_res = np.random.choice(valid_divisors)
            self.perlin_res = [int(base_res), int(base_res), int(base_res)]
            
        elif dice_shape < 0.7: 
            base_idx = np.random.randint(0, len(valid_divisors) - 2)
            res1 = valid_divisors[base_idx]
            res2 = valid_divisors[base_idx + np.random.randint(1, 3)] 
            self.perlin_res = [int(res1), int(res1), int(res2)]
            np.random.shuffle(self.perlin_res)
            
        else: 
            base_idx = np.random.randint(4, len(valid_divisors) - 1)
            r1 = valid_divisors[base_idx]
            r2 = valid_divisors[base_idx + 1]
            self.perlin_res = [int(np.random.choice([r1, r2])) for _ in range(3)]

        # =====================================================
        # =====================================================
        mean_res = np.mean(self.perlin_res)
        safe_velocity_cap = 2500 / (mean_res + 1e-5)
        risk_factor = 1.0 if np.random.rand() > 0.2 else 1.5 
        
        max_v = max(50, int(safe_velocity_cap * risk_factor))
        min_v = max(20, int(max_v * 0.2))
        
        if min_v >= max_v:
            self.V_multiplier = min_v
        else:
            self.V_multiplier = np.random.randint(min_v, max_v)

        if mean_res > 15 or self.V_multiplier > 400:
            self.dt = np.random.uniform(0.05, 0.1) 
            self.max_nt = np.random.randint(10, 20)
        else:
            self.dt = np.random.uniform(0.1, 0.2)  
            self.max_nt = np.random.randint(4, 15)

        if np.random.rand() < 0.1: self.max_nt = 0
        
        if mean_res > 15:
            self.D_multiplier = np.random.uniform(1.5, 2.5)
        else:
            self.D_multiplier = np.random.uniform(0.5, 1.5)

        # =====================================================
        # =====================================================
        center_p = np.random.uniform(40.0, 98.5)
        max_possible_width = min(center_p - 0, 100 - center_p) * 2
        bandwidth = np.random.uniform(1.0, min(15.0, max_possible_width))
        
        self.mask_percentile_min = center_p - (bandwidth / 2)
        self.mask_percentile_max = center_p + (bandwidth / 2)

        # if mean_res > 15 and self.mask_percentile_min > 75:
        #      shift = np.random.uniform(5, 10)
        #      self.mask_percentile_min = max(50, self.mask_percentile_min - shift)
        #      self.mask_percentile_max = max(55, self.mask_percentile_max - shift)

    def __repr__(self):
        return (f"Cfg(Res={self.perlin_res}, V={self.V_multiplier}, dt={self.dt:.2f}, D={self.D_multiplier:.2f})")