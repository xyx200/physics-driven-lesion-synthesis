import os
import subprocess
import sys
import threading
import concurrent.futures
import shutil
from pathlib import Path

class DynamicScanner:
    def __init__(self, input_path, output_path, target_modalities):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.target_modalities = target_modalities
        self.generator = self.input_path.rglob("*.nii.gz")
        self.lock = threading.Lock()
        self.processed = set() 

    def get_next(self):
        with self.lock:
            for f in self.generator:
                if 'derivatives' in f.parts: 
                    continue
                    
                filename = f.name
                if any(m in filename for m in self.target_modalities) and '_bet' not in filename and 'mask' not in filename:
                    rel = f.relative_to(self.input_path)
                    expected_mask = self.output_path / rel.parent / f"{filename.split('.nii')[0]}_bet.nii.gz"
                    
                    if not expected_mask.exists() and f not in self.processed:
                        self.processed.add(f) 
                        return f
            return None 

def worker(gpu_id, scanner):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    while True:
        nii_file = scanner.get_next()
        
        if nii_file is None:
            print(f"[GPU {gpu_id}] Scan completed or no pending files; exiting task.")
            break
            
        print(f"[GPU {gpu_id}] New task received; extracting: {nii_file.name}")
        
        rel = nii_file.relative_to(scanner.input_path)
        target_dir = scanner.output_path / rel.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        
        temp_work_dir = target_dir / f".tmp_hdbet_{nii_file.name.replace('.nii.gz', '')}"
        temp_work_dir.mkdir(parents=True, exist_ok=True)
        
        temp_out = temp_work_dir / nii_file.name 
        
        cmd_hdbet = [
            "hd-bet", 
            "-i", str(nii_file), 
            "-o", str(temp_out), 
            "--save_bet_mask"
        ]
        
        res_bet = subprocess.run(cmd_hdbet, stdout=sys.stdout, stderr=sys.stderr, env=env)
        
        if res_bet.returncode == 0:
            for generated_file in temp_work_dir.glob("*"):
                dest_path = target_dir / generated_file.name
                if dest_path.exists():
                    dest_path.unlink()
                shutil.move(str(generated_file), str(dest_path))
                
            temp_work_dir.rmdir()
            print(f"[GPU {gpu_id}] [Success] Extraction completed: {nii_file.name}\n")
        else:
            print(f"[GPU {gpu_id}] [Failed] HD-BET crashed: {nii_file.name}\n")

def run_hdbet_pipeline(input_root, output_root):
    target_modalities = ['T1w', 'FLAIR', 'TRACE']
    
    scanner = DynamicScanner(input_root, output_root, target_modalities)
    
    gpu_ids = [0,  3, 4]
    
    print("--- Starting HD-BET dynamic scanning with 6 GPUs and conflict prevention ---")
    print("Dispatching tasks in parallel...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, gpu_id, scanner) for gpu_id in gpu_ids]
        concurrent.futures.wait(futures)
        
    print("All files processed.")

if __name__ == "__main__":
    src = "/path/to/data/CUZHONG/cuzhonglps"
    dst = "/path/to/data/CUZHONG/cuzhongHD"
    run_hdbet_pipeline(src, dst)