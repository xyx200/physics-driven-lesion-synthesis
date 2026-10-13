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
        
        self.perlin_res = [2, 2, 2] 
        self.dt = 0.1
        self.V_multiplier = 300
        self.max_nt = 50
        
        self.binarize_threshold = 0.3 
        self.D_multiplier = 1.0

    def randomize(self):

        valid_divisors = np.array([2, 4, 5, 8, 10, 20])
        dice_shape = np.random.rand()
        
        if dice_shape < 0.65: 
            base_idx = np.random.choice([0, 1]) 
            self.perlin_res = [
                int(valid_divisors[base_idx]), 
                int(valid_divisors[base_idx]), 
                int(valid_divisors[base_idx + 1]) 
            ]
            np.random.shuffle(self.perlin_res)
        elif dice_shape < 0.85: 
            base_idx = np.random.choice([1, 2]) 
            self.perlin_res = [
                int(valid_divisors[base_idx]), 
                int(valid_divisors[base_idx]), 
                int(valid_divisors[base_idx + 2]) 
            ]
            np.random.shuffle(self.perlin_res)
        else: 
            base_res = int(np.random.choice([8, 10]))
            self.perlin_res = [base_res, base_res, base_res]

        mean_res = np.mean(self.perlin_res)
        
        if mean_res <= 5.0:
            self.V_multiplier = np.random.randint(400, 700) 
            self.max_nt = np.random.randint(25, 70)           
            self.dt = np.random.uniform(0.05, 0.1) 
            self.D_multiplier = np.random.uniform(0.8, 1.4)
        else:
            self.V_multiplier = np.random.randint(250, 450)  
            self.max_nt = np.random.randint(30, 60)           
            self.dt = np.random.uniform(0.02, 0.05)
            self.D_multiplier = np.random.uniform(1.5, 2.1)


        self.binarize_threshold = np.random.uniform(0.3, 0.5) 
        
    def __repr__(self):
        return (f"Cfg(Res={self.perlin_res}, V={self.V_multiplier}, nt={self.max_nt}, D={self.D_multiplier:.2f}, th={self.binarize_threshold:.2f})")