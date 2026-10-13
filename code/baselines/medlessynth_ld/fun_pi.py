import skimage
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import skimage.morphology
import skimage.transform as skiform
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from scipy.stats import rice

class LesionGeneration3D(Dataset):
    def __init__(self, img_path, gt_path=None, mask_path = None, roi_path=None, type_of_imgs='nifty',have_texture=True,have_noise=True, have_smoothing=True, have_small=True, have_edema=True, return_param=True, transform=None, which_data='wmh',perturb=False, size=(128,128,128),semi_axis_range=[(5,10)],centroid_scale=10,num_lesions=5,range_sampling=[1],num_ellipses=15,rician=False):
        self.paths = img_path 
        self.gt_path = gt_path
        self.mask_path = mask_path
        self.roi_path = roi_path
        self.transform = transform
        self.size = size
        self.have_noise = have_noise
        self.have_smoothing = have_smoothing
        self.have_small = have_small
        self.have_edema = have_edema
        self.img_type = type_of_imgs
        self.return_param = return_param
        self.which_data = which_data
        self.perturb = perturb
        self.ranges = semi_axis_range
        self.num_lesions = num_lesions
        self.centroid_scaling = centroid_scale
        self.range_sampling = range_sampling
        self.num_ellipses = num_ellipses        
        self.use_rician = rician

    def gaussian_small_shapes(self,image_mask,small_sigma = [1,1,1]):
        image_mask = skimage.morphology.binary_erosion(image_mask, np.ones((5, 5, 5)))
        index = -1
        while(index==-1):
            noise = np.random.normal(size = image_mask.shape)
            smoothed_noise = gaussian_filter(noise,sigma=small_sigma) 
            smoothed_noise -= smoothed_noise.min()
            smoothed_noise /= smoothed_noise.max()

            bg_mask = (smoothed_noise>0.3)*(image_mask)
            mask = (1-bg_mask)*(image_mask)

            labelled = skimage.measure.label(mask*2,background = 0)         
            regions = skimage.measure.regionprops(labelled)

            total_brain_area = np.sum(image_mask)
            count = 0
            old_area = 0 

            for region in regions:
                if(region.area<total_brain_area*0.01 and region.area>total_brain_area*(0.0001)):
                    if(region.area >old_area):
                        old_area = region.area
                        index = count
                count+=1

            if(index!=-1):
                label = regions[index].label
                label_sim = labelled == label
                return label_sim
        
    def create_ellipsoid(self,coord=(1,2,1),semi_a = 4, semi_b = 34, semi_c = 34, alpha=np.pi/4, beta=np.pi/4, gamma=np.pi/4,img_dim=(128,128,128)):
        img_dim = list(img_dim)
        img_dim[0],img_dim[1] = img_dim[1],img_dim[0]
        x = np.linspace(-img_dim[0]//2,np.ceil(img_dim[0]//2),img_dim[0])
        y = np.linspace(-img_dim[1]//2,np.ceil(img_dim[1]//2),img_dim[1])
        z = np.linspace(-img_dim[2]//2,np.ceil(img_dim[2]//2),img_dim[2])
        x,y,z = np.meshgrid(x,y,z)
  
        x=(x - coord[0] + img_dim[0]//2)
        y=(y - coord[1] + img_dim[1]//2)
        z=(z - coord[2] + img_dim[2]//2)

        ellipsoid_std_axes = np.stack([x,y,z],0)

        alpha = -alpha
        beta = -beta
        gamma = -gamma    

        rotation_x = np.array([[1, 0, 0],
                                [0, np.cos(alpha), -np.sin(alpha)],
                                [0, np.sin(alpha), np.cos(alpha)]])
        
        rotation_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                                [0, 1, 0],
                                [-np.sin(beta), 0, np.cos(beta)]])
        
        rotation_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                                [np.sin(gamma), np.cos(gamma), 0],
                                [0, 0, 1]])

        rot_matrix = rotation_x@rotation_y@rotation_z
        ellipsoid_rot_axes = np.tensordot(rot_matrix,ellipsoid_std_axes,axes=([1,0]))

        x,y,z = ellipsoid_rot_axes
        x**=2
        y**=2
        z**=2

        a = semi_a**2
        b = semi_b**2
        c = semi_c**2

        ellipsoid = x/a + y/b + z/c - 1
        ellipsoid = ellipsoid<0 
        return ellipsoid
        
    def create_pert(self,inter_image, sigma=1.0,size=None, range_min=-0.3, range_max=1.0, pregenerated_noise=None):
        if pregenerated_noise is not None:
            noise = pregenerated_noise.copy()
        else:
            noise = np.random.randn(*size)
            if(self.use_rician):
                img = np.ones(size)
                noise = rice.rvs(img,scale=1)
            noise -=noise.min()
            noise /=noise.max()
            
        noise = gaussian_filter(noise+inter_image,sigma) + 0.5*gaussian_filter(noise+inter_image,sigma/2) + 0.25*gaussian_filter(noise+inter_image,sigma/4)
        noise_min = noise.min()
        noise_max = noise.max()

        tex_noise = ((noise - noise_min)*(range_max-range_min)/(noise_max-noise_min)) + range_min 
        return tex_noise
        
    def localise_pert(self,scale_centroids,centroid_main,num_ellipses,semi_axes_range,perturb_sigma=[1,1,1],image_mask=1):
        random_centroids = centroid_main.T + (np.random.random((num_ellipses,3))*scale_centroids)

        random_major_axes = np.random.uniform(semi_axes_range[0],semi_axes_range[1],(num_ellipses,1))
        random_minor_axes = np.concatenate([np.ones((num_ellipses,1)),np.random.uniform(0.3,1,size = (num_ellipses,2))],1) 
        random_semi_axes = random_major_axes*random_minor_axes

        rng = np.random.default_rng()
        random_semi_axes = rng.permuted(random_semi_axes, axis=1)
        random_rot_angles = np.random.uniform(-np.pi/2,np.pi/2,size = (num_ellipses,3))
        
        out = []
        for i in range(num_ellipses):
            out.append(self.create_ellipsoid(random_centroids[i],*random_semi_axes[i],*random_rot_angles[i],img_dim=image_mask.shape) )

        out = np.logical_or.reduce(out)*image_mask
            
        regions = skimage.measure.regionprops(np.int16(out*2))

        if(regions==[]):
            return np.zeros_like(out),-1

        if(not self.perturb):
            return out,1

        bounding_box = regions[0].bbox

        noise_b_box = np.random.normal(size = image_mask.shape)
        noise_b_box = gaussian_filter(noise_b_box,sigma=perturb_sigma) 
        noise_b_box -= noise_b_box.min()
        noise_b_box /= noise_b_box.max()

        thresholded_b_box = np.zeros_like(out)
        thresholded_b_box[bounding_box[0]:bounding_box[3],bounding_box[1]:bounding_box[4],bounding_box[2]:bounding_box[5]] = (noise_b_box>0.6)[bounding_box[0]:bounding_box[3],bounding_box[1]:bounding_box[4],bounding_box[2]:bounding_box[5]]

        labelled_threshold_b_box, nlabels = skimage.measure.label(thresholded_b_box, return_num=True)
        labels_in_big_lesion = np.union1d(out * labelled_threshold_b_box, [])
        labels_tob_removed = np.setdiff1d(np.arange(1, nlabels+1), labels_in_big_lesion)
        for i in labels_tob_removed:
            labelled_threshold_b_box[labelled_threshold_b_box == i] = 0

        final_region = out + labelled_threshold_b_box >0
        return final_region,1

    def blend_intensity(self,semi_axes_range,tex_sigma_edema,image_mask,range_min,range_max,alpha,beta,gamma,out_edema,out,tex_noise,smoothing_mask,inter_image,smoothing_image,image,output_image,output_mask, dark=True, base_noise_edema=None):
        if(self.have_smoothing and self.have_edema and semi_axes_range !=(2,5) and semi_axes_range !=(3,5) and self.which_data!='lits'):
            tex_noise_edema = self.create_pert(inter_image,tex_sigma_edema,image_mask.shape,range_min,range_max, pregenerated_noise=base_noise_edema)
            if(not self.have_noise):
                tex_noise_edema = 1.0

            if not dark:
                smoothed_les = beta*gaussian_filter(out_edema*(gamma*(1-out)*tex_noise_edema + 4.0*gamma*out*tex_noise), sigma=smoothing_mask)
                smoothed_out = gaussian_filter(output_image,sigma=smoothing_image) 
                
                smoothed_out-=smoothed_out.min()
                smoothed_out/= (smoothed_out.max() + 1e-7)

                image1 = alpha*output_image*(1-out) + smoothed_les
                image2 = alpha*smoothed_out*(1-out) + smoothed_les

                image1[out_edema>0]=image2[out_edema>0]
                image1[image1<0] = 0

                output_mask = np.logical_or(output_mask,out_edema)
                output_image = image1
                output_image -= output_image.min()
                output_image /= (output_image.max() + 1e-7)
            else:
                core_mult = 2.0 
                soft_out = gaussian_filter(out.astype(np.float32), sigma=smoothing_mask)
                
                smoothed_les = beta*gaussian_filter(out_edema*(-0.25*gamma*(1-out)*tex_noise_edema + core_mult*gamma*out*tex_noise), sigma=smoothing_mask)
                smoothed_out = gaussian_filter(output_image,sigma=smoothing_image) 
                
                image1 = alpha*output_image*(1-soft_out) + smoothed_les
                image2 = alpha*smoothed_out*(1-soft_out) + smoothed_les

                image1[out_edema>0]=image2[out_edema>0]
                image1[image1<0] = 0

                output_mask = np.logical_or(output_mask,out_edema)
                output_image = image1
                output_image -= output_image.min()
                output_image /= (output_image.max() + 1e-7)
            
        if(self.have_smoothing and (semi_axes_range==(2,5) or semi_axes_range==(3,5)) or self.which_data=='lits'):
            smoothed_les = gaussian_filter(out*tex_noise, sigma=smoothing_mask)
            if(self.which_data!='wmh'):
                smoothed_les = out*(tex_noise + output_image)
            
            smoothed_out = gaussian_filter(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)

            if not dark:
                image1 = alpha*output_image + beta*smoothed_les
                image2 = alpha*smoothed_out + beta*smoothed_les
            else:
                t1_subtract_ratio_wmh_my = 0.25
                image1 = alpha*output_image - t1_subtract_ratio_wmh_my*beta*smoothed_les
                image2 = alpha*smoothed_out - t1_subtract_ratio_wmh_my*beta*smoothed_les

            image1[out>0] = image2[out>0]
            image1[image1 < 0] = 0.0 

        image_stuff = image>0.01
        image1[image_stuff==0] = 0
        
        return image1

    def simulation(self, image_t1, image_flair, inter_t1, inter_flair, image_mask, num_lesions=3, gt_mask=None, roi_mask=None):
        param_dict = {}
        if(image_t1.shape != inter_t1.shape):
            inter_t1 = skiform.resize(image_t1, image_t1.shape, order=1, preserve_range=True ,anti_aliasing=True)
            inter_flair = skiform.resize(image_flair, image_flair.shape, order=1, preserve_range=True ,anti_aliasing=True)
            
        roi = ndimage.binary_erosion(image_mask,skimage.morphology.ball(5)) 

        if roi_mask is not None:
            roi = roi_mask*roi
        roi_with_masks = roi*(1-gt_mask)
        
        output_image_t1 = image_t1.copy()
        output_image_flair = image_flair.copy()
        output_mask = gt_mask.copy()
        
        total_param_list = ['scale_centroid','num_ellipses','semi_axes_range','alpha','beta','gamma','smoothing_mask',
                            'tex_sigma','range_min','range_max','tex_sigma_edema','pertub_sigma']
        
        num_lesions = np.random.randint(low=num_lesions[0],high=num_lesions[1])
        for i in range(num_lesions):
            gamma = 0
            tex_sigma_edema = 0
            x_corr,y_corr,z_corr = np.nonzero(roi_with_masks[:,:,:]*image_mask)
            if len(x_corr) == 0:
                continue
                
            random_coord_index = np.random.choice(len(x_corr),1)
            centroid_main = np.array([y_corr[random_coord_index],x_corr[random_coord_index],z_corr[random_coord_index]])

            scale_centroid = np.random.uniform(2,self.centroid_scaling)
            num_ellipses = np.random.randint(5,self.num_ellipses)
            semi_axes_range = self.ranges[int(np.random.choice(len(self.ranges),p=self.range_sampling))]
                
            if(semi_axes_range==(2,5)):
                alpha = np.random.uniform(0.5,0.65)
                beta = 1-alpha
            if(semi_axes_range!=(2,5)):
                alpha = np.random.uniform(0.5,0.65)
                beta = 1-alpha

            if(self.which_data=='wmh'):
                alpha = np.random.uniform(0.5,0.6)
                beta = 1-alpha

            smoothing_mask = np.random.uniform(0.6,0.8)
            smoothing_image = np.random.uniform(0.05,0.1) 
                
            tex_sigma = np.random.uniform(0.4,0.7)
            perturb_sigma = np.random.uniform(0.5,5)

            out_edema = np.zeros_like(image_t1)
            if(semi_axes_range == (2,5)):
                small_sigma = np.random.uniform(1,2)
                out = self.gaussian_small_shapes(image_mask,small_sigma)
            else:   
                out,shape_status = self.localise_pert(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask) 
                while(shape_status==-1):
                    random_coord_index = np.random.choice(len(x_corr),1)
                    centroid_main = np.array([y_corr[random_coord_index],x_corr[random_coord_index],z_corr[random_coord_index]])
                    out,shape_status = self.localise_pert(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask)
            if roi_mask is not None:
                out *= roi_mask            

            if(semi_axes_range !=(2,5) and semi_axes_range !=(3,5) and self.which_data!='lits'):
                semi_axes_range_edema = (semi_axes_range[0]+5,semi_axes_range[1]+5)
                tex_sigma_edema = np.random.uniform(1,1.5)
                beta = 1-alpha
                gamma = np.random.uniform(0.5,0.7)
                out_edema,shape_status = self.localise_pert(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)
                while(shape_status==-1):
                    random_coord_index = np.random.choice(len(x_corr),1)
                    centroid_main = np.array([y_corr[random_coord_index],x_corr[random_coord_index],z_corr[random_coord_index]])
                    out_edema,shape_status = self.localise_pert(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)

                out_edema = np.logical_or(out_edema, out)

            output_mask = np.logical_or(output_mask,out)
            output_mask = np.logical_or(output_mask, out_edema)

            # =========================================================================
            # =========================================================================
            if(self.have_noise and semi_axes_range !=(2,5)):
                base_noise_core = np.random.randn(*image_mask.shape)
                if(self.use_rician):
                    base_noise_core = rice.rvs(np.ones(image_mask.shape),scale=1)
                base_noise_core -= base_noise_core.min()
                base_noise_core /= base_noise_core.max()
            else:
                base_noise_core = None

            if(self.have_smoothing and self.have_edema and semi_axes_range !=(2,5) and semi_axes_range !=(3,5) and self.which_data!='lits'):
                base_noise_edema = np.random.randn(*image_mask.shape)
                if(self.use_rician):
                    base_noise_edema = rice.rvs(np.ones(image_mask.shape),scale=1)
                base_noise_edema -= base_noise_edema.min()
                base_noise_edema /= base_noise_edema.max()
            else:
                base_noise_edema = None
            # =========================================================================


            range_min_f = np.random.uniform(0.0, 0.5)
            range_max_f = np.random.uniform(0.7, 1.0)
            if(self.have_noise and semi_axes_range !=(2,5)):
                tex_noise_f = self.create_pert(inter_flair,tex_sigma,image_mask.shape,range_min_f,range_max_f, pregenerated_noise=base_noise_core)
            else:
                tex_noise_f = 1.0
            
            output_image_flair = self.blend_intensity(semi_axes_range,tex_sigma_edema,image_mask,range_min_f,range_max_f,alpha,beta,gamma,out_edema,out,tex_noise_f,smoothing_mask,inter_flair,smoothing_image,image_flair,output_image_flair,output_mask, dark=False, base_noise_edema=base_noise_edema)
            output_image_flair -= output_image_flair.min()
            output_image_flair /= (output_image_flair.max() + 1e-7)

            range_min_t = np.random.uniform(0.0, 0.35)
            range_max_t = np.random.uniform(0.40, 0.6)
            if(self.have_noise and semi_axes_range !=(2,5)):
                tex_noise_t = self.create_pert(inter_t1,tex_sigma,image_mask.shape,range_min_t,range_max_t, pregenerated_noise=base_noise_core)
            else:
                tex_noise_t = 0.3
            
            output_image_t1 = self.blend_intensity(semi_axes_range,tex_sigma_edema,image_mask,range_min_t,range_max_t,alpha,beta,gamma,out_edema,out,tex_noise_t,smoothing_mask,inter_t1,smoothing_image,image_t1,output_image_t1,output_mask, dark=True, base_noise_edema=base_noise_edema)
            output_image_t1 -= output_image_t1.min()
            output_image_t1 /= (output_image_t1.max() + 1e-7)

            roi_with_masks *= (1-output_mask)>0
            
            total_params = [scale_centroid,num_ellipses,semi_axes_range,alpha,beta,gamma,smoothing_mask,
                            tex_sigma,range_min_f,range_max_f,tex_sigma_edema,perturb_sigma]
            
            for j in range(len(total_params)):
                param_dict[str(i)+'_'+total_param_list[j]] = total_params[j]
        
        param_dict['num_lesions'] = num_lesions

        return output_image_t1, output_image_flair, output_mask, param_dict

    def read_image(self,index,inter_image=False):
        t1_path, flair_path = self.paths[index]
        
        nii_t1 = nib.load(t1_path)            
        nii_flair = nib.load(flair_path)
        
        affine = nii_t1._affine
        
        img_data_t1 = nii_t1.get_fdata()
        img_data_flair = nii_flair.get_fdata()
        
        row_sum = np.sum(np.sum(img_data_t1, axis=1), axis=1)
        col_sum = np.sum(np.sum(img_data_t1, axis=0), axis=1)
        stack_sum = np.sum(np.sum(img_data_t1, axis=1), axis=0)
        rsid, reid, rlen = self.cut_zeros1d(row_sum)
        csid, ceid, clen = self.cut_zeros1d(col_sum)
        ssid, seid, slen = self.cut_zeros1d(stack_sum)
        img_crop_para = [rsid, rlen, csid, clen, ssid, slen]
        
        image_t1 = img_data_t1[rsid:rsid + rlen, csid:csid + clen, ssid:ssid + slen]
        sub_min_t1 = min(0, img_data_t1.min())
        image_t1 -= image_t1.min()
        image_t1 /= image_t1.max() + 1e-7
        
        image_flair = img_data_flair[rsid:rsid + rlen, csid:csid + clen, ssid:ssid + slen]
        sub_min_flair = min(0, img_data_flair.min())
        image_flair -= image_flair.min()
        image_flair /= image_flair.max() + 1e-7

        if(self.gt_path!=None):
            gt_img = nib.load(self.gt_path[index]).get_fdata()
            gt_img = gt_img[rsid:rsid + rlen, csid:csid + clen, ssid:ssid + slen]
            original_gt = gt_img.copy() 
            gt_mask = (gt_img == 4).astype(np.float32) 
            roi_mask = None
        else:
            gt_mask = np.zeros_like(image_t1)
            original_gt = np.zeros_like(image_t1)
            roi_mask = None

        if(inter_image and self.which_data!='lits'):
            angle = np.random.uniform(0, 180)
            axes = np.random.choice([0,1,2],2,replace=False)
            image_t1 = ndimage.rotate(image_t1, float(angle), axes=axes, reshape=False, mode='nearest')
            image_flair = ndimage.rotate(image_flair, float(angle), axes=axes, reshape=False, mode='nearest')
        
        return image_t1, image_flair, affine, img_crop_para, gt_mask, roi_mask, original_gt
    
    def __getitem__(self, index):
        image_t1, image_flair, affine, img_crop_para, gt_mask, roi_mask, original_gt = self.read_image(index)

        interpolation_choice = np.random.choice(len(self.paths))
        inter_t1, inter_flair, _, _, _, _, _ = self.read_image(interpolation_choice, inter_image=True)

        if(self.mask_path):
            brain_mask_img = nib.load(self.mask_path[index]).get_fdata()
            brain_mask_img = brain_mask_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1],img_crop_para[2]:img_crop_para[2] + img_crop_para[3],img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
        else:
            brain_mask_img = ndimage.binary_fill_holes(image_t1>0,structure=np.ones((3,3,3)))
        
        out_t1, out_flair, output_mask, param_dict = self.simulation(
            image_t1, image_flair, inter_t1, inter_flair, brain_mask_img, 
            self.num_lesions, gt_mask=gt_mask, roi_mask=roi_mask
        )

        final_label = original_gt.copy()
        final_label[output_mask > 0] = 4 

        t1_path, _ = self.paths[index]
        return out_t1.astype(np.single), out_flair.astype(np.single), final_label.astype(np.single), affine, t1_path, param_dict

    def __len__(self):
        return len(self.paths)

    def cut_zeros1d(self, im_array):
        im_list = list(im_array > 0)
        start_index = im_list.index(1)
        end_index = im_list[::-1].index(1)
        length = len(im_array[start_index:]) - end_index
        return start_index, end_index, length

    def tight_crop_data(self, img_data):
        pass