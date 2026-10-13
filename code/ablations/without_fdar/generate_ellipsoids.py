import os
import argparse
import numpy as np
import skimage.measure
from scipy.ndimage import gaussian_filter
import nibabel as nib
import concurrent.futures

class LesionShapeGenerator:
    def __init__(self, perturb=False):
        self.perturb = perturb

    def create_ellipsoid(self, coord=(1,2,1), semi_a=4, semi_b=34, semi_c=34, alpha=np.pi/4, beta=np.pi/4, gamma=np.pi/4, img_dim=(147,147,184)):
        img_dim = list(img_dim)
        img_dim[0], img_dim[1] = img_dim[1], img_dim[0]
        x = np.linspace(-img_dim[0]//2, np.ceil(img_dim[0]//2), img_dim[0])
        y = np.linspace(-img_dim[1]//2, np.ceil(img_dim[1]//2), img_dim[1])
        z = np.linspace(-img_dim[2]//2, np.ceil(img_dim[2]//2), img_dim[2])
        x, y, z = np.meshgrid(x, y, z)

        x = (x - coord[0] + img_dim[0]//2)
        y = (y - coord[1] + img_dim[1]//2)
        z = (z - coord[2] + img_dim[2]//2)

        ellipsoid_std_axes = np.stack([x, y, z], 0)

        alpha, beta, gamma = -alpha, -beta, -gamma    

        rotation_x = np.array([[1, 0, 0],
                               [0, np.cos(alpha), -np.sin(alpha)],
                               [0, np.sin(alpha), np.cos(alpha)]])
        
        rotation_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                               [0, 1, 0],
                               [-np.sin(beta), 0, np.cos(beta)]])
        
        rotation_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                               [np.sin(gamma), np.cos(gamma), 0],
                               [0, 0, 1]])

        rot_matrix = rotation_x @ rotation_y @ rotation_z
        ellipsoid_rot_axes = np.tensordot(rot_matrix, ellipsoid_std_axes, axes=([1,0]))

        x, y, z = ellipsoid_rot_axes
        x **= 2
        y **= 2
        z **= 2

        a, b, c = semi_a**2, semi_b**2, semi_c**2

        ellipsoid = x/a + y/b + z/c - 1
        return ellipsoid < 0 

    def localise_pert(self, scale_centroids, centroid_main, num_ellipses, semi_axes_range, perturb_sigma=[1,1,1], image_mask=1):
        random_centroids = centroid_main.T + (np.random.random((num_ellipses,3)) * scale_centroids)

        random_major_axes = np.random.uniform(semi_axes_range[0], semi_axes_range[1], (num_ellipses, 1))
        random_minor_axes = np.concatenate([np.ones((num_ellipses,1)), np.random.uniform(0.3, 1, size=(num_ellipses,2))], 1)
        random_semi_axes = random_major_axes * random_minor_axes

        rng = np.random.default_rng()
        random_semi_axes = rng.permuted(random_semi_axes, axis=1)
        random_rot_angles = np.random.uniform(-np.pi/2, np.pi/2, size=(num_ellipses,3))
        
        out = []
        img_shape = image_mask.shape if hasattr(image_mask, 'shape') else (147, 184, 147) 

        for i in range(num_ellipses):
            out.append(self.create_ellipsoid(random_centroids[i], *random_semi_axes[i], *random_rot_angles[i], img_dim=img_shape))

        out = np.logical_or.reduce(out) * image_mask
        regions = skimage.measure.regionprops(np.int16(out * 2))

        if not regions:
            return np.zeros_like(out), -1

        if not self.perturb:
            return out, 1

        min_x, min_y, min_z, max_x, max_y, max_z = regions[0].bbox
        margin = 4
        
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        min_z = max(0, min_z - margin)
        max_x = min(img_shape[0], max_x + margin)
        max_y = min(img_shape[1], max_y + margin)
        max_z = min(img_shape[2], max_z + margin)

        noise_b_box = np.random.normal(size=img_shape)
        noise_b_box = gaussian_filter(noise_b_box, sigma=perturb_sigma) 
        noise_b_box -= noise_b_box.min()
        noise_b_box /= noise_b_box.max()

        thresholded_b_box = np.zeros_like(out)
        
        thresholded_b_box[min_x:max_x, min_y:max_y, min_z:max_z] = \
            (noise_b_box > 0.6)[min_x:max_x, min_y:max_y, min_z:max_z]
        # ====================================================

        labelled_threshold_b_box, nlabels = skimage.measure.label(thresholded_b_box, return_num=True)
        labels_in_big_lesion = np.union1d(out * labelled_threshold_b_box, [])
        labels_tob_removed = np.setdiff1d(np.arange(1, nlabels+1), labels_in_big_lesion)
        for i in labels_tob_removed:
            labelled_threshold_b_box[labelled_threshold_b_box == i] = 0

        final_region = out + labelled_threshold_b_box > 0
        return final_region, 1

def generate_single_mask(task_id, output_dir):
    seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(seed)

    generator = LesionShapeGenerator(perturb=False)
    image_mask = np.ones((147, 184, 147), dtype=np.uint8)
    
    final_mask = np.zeros((147, 184, 147), dtype=bool)
    
    centroid_scaling = 10         
    max_num_ellipses = 15         

    ranges = [(3, 6), (6, 11), (11, 16)]  
    range_sampling = [0.5, 0.35, 0.15]
    
    num_lesions_in_image = np.random.randint(14, 25) 

    for lesion_idx in range(num_lesions_in_image):
        while True:
            cx = np.random.randint(0, 147)
            cy = np.random.randint(0, 184)
            cz = np.random.randint(0, 147)
            
            centroid_main = np.array([cy, cx, cz]) 
            
            scale_centroid = np.random.uniform(2, centroid_scaling)
            num_ellipses = np.random.randint(5, max_num_ellipses)
            semi_axes_range = ranges[np.random.choice(len(ranges), p=range_sampling)]
            perturb_sigma = np.random.uniform(0.5, 5)
            
            out_mask, shape_status = generator.localise_pert(
                scale_centroids=scale_centroid,
                centroid_main=centroid_main,
                num_ellipses=num_ellipses,
                semi_axes_range=semi_axes_range,
                perturb_sigma=[perturb_sigma, perturb_sigma, perturb_sigma],
                image_mask=image_mask
            )
            
            if shape_status != -1 and np.any(out_mask):
                final_mask = np.logical_or(final_mask, out_mask)
                break

    binary_mask = final_mask.astype(np.uint8)

    affine = np.eye(4)
    nii_img = nib.Nifti1Image(binary_mask, affine)

    filename = f"lesion_mask_{task_id:02d}.nii.gz"
    filepath = os.path.join(output_dir, filename)
    nib.save(nii_img, filepath)
    
    return filepath, num_lesions_in_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an ellipsoid shape library for ablation")
    parser.add_argument('--output', required=True, help='Output directory for NIfTI masks')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=4820, help='Exclusive end index')
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_single_mask, i, args.output): i
                   for i in range(args.start, args.end)}
        for future in concurrent.futures.as_completed(futures):
            task_id = futures[future]
            filepath, lesion_count = future.result()
            print(f"[{task_id}] {os.path.basename(filepath)}; lesion groups={lesion_count}")
