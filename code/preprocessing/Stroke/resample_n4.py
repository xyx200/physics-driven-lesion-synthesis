import os
import re
import SimpleITK as sitk
import concurrent.futures

def resample_image(input_path):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(8)
    
    output_path = re.sub(r'_T1w_n4\.nii\.gz$', '_T1w_n4_chong.nii.gz', input_path)

    # if os.path.exists(output_path):
    #     return True

    try:
        image = sitk.ReadImage(input_path)
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_spacing = [max(1.0, sp) for sp in original_spacing]

        if new_spacing == list(original_spacing):
            print(f"[Resampling skipped] All spacings >=1; saving directly: {input_path}")
            sitk.WriteImage(image, output_path)
            return True

        new_size = [
            int(round(osz * osp / nsp))
            for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkLinear) 
        resampler.SetDefaultPixelValue(0)

        resampled_image = resampler.Execute(image)

        
        
        resampled_image = sitk.Cast(resampled_image, image.GetPixelID())

        resampled_image = sitk.Cast(resampled_image, image.GetPixelID())

        sitk.WriteImage(resampled_image, output_path)
        print(f"[Success] {os.path.basename(input_path)} \n      Original spacing: {[round(s,3) for s in original_spacing]} -> New spacing: {new_spacing}")
        
        return True

    except Exception as e:
        print(f"[Error] Failed to process file {input_path}: {str(e)}")
        return False

def main():
    base_dir = '/path/to/data/newCU/cuzhongN4'
    
    print("Scanning target folder...")
    target_files = []
    for root, dirs, files in os.walk(base_dir):
        if 'derivatives' in dirs:
            dirs.remove('derivatives')
            
        for file in files:
            if file.endswith('_T1w_n4.nii.gz'):
                target_files.append(os.path.join(root, file))
                
    total_files = len(target_files)
    print(f"Found {total_files} _T1w_n4.nii.gz files.")
    
    if total_files == 0:
        return

    max_workers = max(1, 40)
    print(f"Preparing multiprocessing with {max_workers} processes ...")
    
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(resample_image, target_files)
        
        for res in results:
            if res:
                success_count += 1

    print("=" * 50)
    print(f"All processing completed! Succeeded: {success_count} / {total_files}")
    print("=" * 50)

if __name__ == '__main__':
    main()