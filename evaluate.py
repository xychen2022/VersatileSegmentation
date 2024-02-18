#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18, 2023

@author: xychen
"""

import os
import torch
import timeit
import argparse
import numpy as np
import SimpleITK as sitk
from networks.vit_seg_configs import get_vit_3d_config
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
 
def get_arguments():
    parser = argparse.ArgumentParser(description="Versatile Model toward Universal Segmentation")
    parser.add_argument("--data_dir", type=str, default='/gpfs/fs001/cbica/home/chexiao/MultiOrgan/awesome/MM')
    parser.add_argument("--datasets_for_testing", nargs="+", type=str, default=["amos", "btcv", "flare22", "abdomenct1k", "totalsegmentator", "nihpancreas", "word", "urogram122"]) #["amos", "btcv", "flare22", "abdomenct1k", "totalsegmentator", "nihpancreas", "word", "urogram122"]
    parser.add_argument("--modalities_for_testing", nargs="+", type=str, default=["ct", "t1w"])
    parser.add_argument("--snapshot_dir", type=str, default='./snapshots/')
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--reload_path", type=str, default='snapshots/best_model_ckpt.pth')
    parser.add_argument("--save_path", type=str, default='test_results')
    parser.add_argument("--common_spacing", nargs="+", type=float, default=[2.0, 2.0, 2.0])
    parser.add_argument("--input_size", nargs="+", type=int, default=[112, 112, 112]) # [112, 112, 112], [96, 96, 96], [64, 64, 64]
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument("--total_classes", type=int, default=17)
    parser.add_argument("--clip_lower", type=int, default=-400)
    parser.add_argument("--clip_upper", type=int, default=400)
    
    return parser

def resample_img(itk_image, out_spacing=[1.0, 1.0, 3.0], out_size=None):
    
    imageFilter = sitk.MinimumMaximumImageFilter()
    imageFilter.Execute(itk_image)
    
    width, height, depth = itk_image.GetSize()
    original_spacing = itk_image.GetSpacing()
    
    rotation_center = itk_image.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                                               int(np.ceil(height/2)),
                                                               int(np.ceil(depth/2))))
    theta_z = 0 # random.uniform(-math.pi/6.0, math.pi/6.0)
    theta_y = 0 # random.uniform(-math.pi/6.0, math.pi/6.0)
    theta_x = 0 # random.uniform(-math.pi/6.0, math.pi/6.0)
    translation = [0, 0, 0] # [random.randint(-25, 25) for i in range(3)]
    scale_factor = [1, 1, 1] #### 1 # random.uniform(0.8, 1.2)
    similarity = sitk.Euler3DTransform(rotation_center, theta_x, theta_y, theta_z, translation)
    
    T=sitk.AffineTransform(3)
    T.SetMatrix(similarity.GetMatrix())
    T.SetCenter(similarity.GetCenter())
    T.SetTranslation(similarity.GetTranslation())
    T.Scale(scale_factor)
    
    # Resample images to out_spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    if out_size == None: # This is a work-around
        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]) * scale_factor[0])),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]) * scale_factor[1])),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2]) * scale_factor[2]))]
    
    out_spacing = [out_spacing[0] / scale_factor[0], out_spacing[1] / scale_factor[1], out_spacing[2] / scale_factor[2]]
    
    resample1 = sitk.ResampleImageFilter()
    resample1.SetOutputSpacing(out_spacing)
    resample1.SetSize(out_size)
    resample1.SetOutputDirection(itk_image.GetDirection())
    resample1.SetOutputOrigin(itk_image.GetOrigin())
    # resample1.SetTransform(sitk.Transform())
    resample1.SetTransform(T)
    
    resample1.SetDefaultPixelValue(imageFilter.GetMinimum())
    resample1.SetInterpolator(sitk.sitkLinear)
    
    resampled_image = resample1.Execute(itk_image)
    
    return resampled_image

def resample_lab(itk_image, out_spacing=[1.0, 1.0, 3.0], out_size=None):
    
    imageFilter = sitk.MinimumMaximumImageFilter()
    imageFilter.Execute(itk_image)
    
    width, height, depth = itk_image.GetSize()
    original_spacing = itk_image.GetSpacing()
    
    rotation_center = itk_image.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                                               int(np.ceil(height/2)),
                                                               int(np.ceil(depth/2))))
    theta_z = 0 # random.uniform(-math.pi/6.0, math.pi/6.0)
    theta_y = 0 # random.uniform(-math.pi/6.0, math.pi/6.0)
    theta_x = 0 # random.uniform(-math.pi/6.0, math.pi/6.0)
    translation = [0, 0, 0] # [random.randint(-25, 25) for i in range(3)]
    scale_factor = [1, 1, 1] #### 1 # random.uniform(0.8, 1.2)
    similarity = sitk.Euler3DTransform(rotation_center, theta_x, theta_y, theta_z, translation)
    
    T=sitk.AffineTransform(3)
    T.SetMatrix(similarity.GetMatrix())
    T.SetCenter(similarity.GetCenter())
    T.SetTranslation(similarity.GetTranslation())
    T.Scale(scale_factor)
    
    # Resample images to out_spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    if out_size == None: # This is a work-around
        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]) * scale_factor[0])),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]) * scale_factor[1])),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2]) * scale_factor[2]))]
    
    out_spacing = [out_spacing[0] / scale_factor[0], out_spacing[1] / scale_factor[1], out_spacing[2] / scale_factor[2]]
    
    resample1 = sitk.ResampleImageFilter()
    resample1.SetOutputSpacing(out_spacing)
    resample1.SetSize(out_size)
    resample1.SetOutputDirection(itk_image.GetDirection())
    resample1.SetOutputOrigin(itk_image.GetOrigin())
    # resample1.SetTransform(sitk.Transform())
    resample1.SetTransform(T)
    
    resample1.SetDefaultPixelValue(0)
    resample1.SetInterpolator(sitk.sitkNearestNeighbor)
    
    resampled_image = resample1.Execute(itk_image)
    
    return resampled_image

def truncate(CT, min_HU, max_HU):
    CT = np.clip(CT, min_HU, max_HU)
    CT = (CT - min_HU) / (max_HU - min_HU)
    return CT

def corrected_crop(array, image_size):
    array_ = array.copy()
    image_size_ = image_size.copy()
    
    copy_from = [0, 0, 0, 0, 0, 0]
    copy_to = [0, 0, 0, 0, 0, 0]
    ## 0 ##
    if array[0] < 0:
        copy_from[0] = 0
        copy_to[0] = int(abs(array_[0]))
    else:
        copy_from[0] = int(array_[0])
        copy_to[0] = 0
    ## 1 ##
    if array[1] > image_size_[0]:
        copy_from[1] = None
        copy_to[1] = -int(array_[1] - image_size_[0])
    else:
        copy_from[1] = int(array_[1])
        copy_to[1] = None
    ## 2 ##
    if array[2] < 0:
        copy_from[2] = 0
        copy_to[2] = int(abs(array_[2]))
    else:
        copy_from[2] = int(array_[2])
        copy_to[2] = 0
    ## 3 ##
    if array[3] > image_size_[1]:
        copy_from[3] = None
        copy_to[3] = -int(array_[3] - image_size_[1])
    else:
        copy_from[3] = int(array_[3])
        copy_to[3] = None
    ## 4 ##
    if array[4] < 0:
        copy_from[4] = 0
        copy_to[4] = int(abs(array_[4]))
    else:
        copy_from[4] = int(array_[4])
        copy_to[4] = 0
    ## 5 ##  
    if array[5] > image_size_[2]:
        copy_from[5] = None
        copy_to[5] = -int(array_[5] - image_size_[2])
    else:
        copy_from[5] = int(array_[5])
        copy_to[5] = None

    return copy_from, copy_to

def flip_array(array, direction):
    if direction[0] < 0:
        array = array[:, :, ::-1]
    if direction[4] < 0:
        array = array[:, ::-1, :]
    if direction[8] < 0:
        array = array[::-1, :, :]
    return array

def main(args):
    
    patch_size = np.array(args.input_size)
    
    patch_stride_regulator = np.array([2, 2, 2]) # this value determines the stride in each dimension when getting the patch; if value = 2, the stride in that dimension is half the value of patch_size
    stride = patch_size/patch_stride_regulator
    
    # Create network
    vit_seg = get_vit_3d_config()
    model = ViT_seg(vit_seg, img_size=args.input_size, num_classes=args.total_classes, in_channels=1)
    
    # Load checkpoint...
    model.load_state_dict(torch.load(args.reload_path)['model_state_dict'])
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    val_subjects = []
    images_for_val = os.listdir(args.data_dir + "/imagesVa")
    for val_subject in images_for_val:
        modality = val_subject.split('_')[1]
        dataset = val_subject.split('_')[0]
        
        if dataset in args.datasets_for_testing and modality in args.modalities_for_testing:
            val_subjects.append(val_subject)
    
    start = timeit.default_timer()
    
    for subject in val_subjects:
        
        print("\nProcessing ", subject)
        
        modality = subject.split('_')[1].lower()
        
        image = sitk.ReadImage(os.path.join(args.data_dir, "imagesVa", subject))
        
        size = image.GetSize()
        spacing = image.GetSpacing()
        orientation = image.GetDirection()
        origin = image.GetOrigin()
        
        image = resample_img(image, out_spacing=[args.common_spacing[0], args.common_spacing[1], args.common_spacing[2]])
        
        image = sitk.GetArrayFromImage(image)
        
        if modality == "ct":
            image = np.clip(image, args.clip_lower, args.clip_upper)
            image = (image - args.clip_lower) / (args.clip_upper - args.clip_lower)
        elif modality == "t1w" or modality == "t2w":
            mri_min = np.quantile(image[np.where(image>0)], 0.01)
            mri_max = np.quantile(image[np.where(image>0)], 0.99)
            
            image = np.clip(image, mri_min, mri_max)
            image = (image - mri_min) / (mri_max - mri_min)
        else:
            assert 0, "not defined yet"
        
        image = flip_array(image, orientation)
        
        image_size = np.shape(image)
        image = np.expand_dims(image, axis=0)
        
        expanded_image_size = np.maximum( (np.ceil(image_size/(1.0*stride)) * stride).astype(np.int32), patch_size)
        
        pad_front = (expanded_image_size - image_size) // 2
        
        expanded_image = np.zeros([1,] + list(expanded_image_size), dtype=np.float32)
        expanded_image[:, pad_front[0]:pad_front[0]+image_size[0], pad_front[1]:pad_front[1]+image_size[1], pad_front[2]:pad_front[2]+image_size[2]] = image
        
        predicted_seg = np.zeros([args.total_classes, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
        count_matrix_seg = np.zeros([args.total_classes, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)

        num_of_patch_with_overlapping = (expanded_image_size/stride - patch_stride_regulator + 1).astype(np.int16)
        
        total_num_of_patches = np.prod(num_of_patch_with_overlapping)
        
        num_patch_z = num_of_patch_with_overlapping[0] # used for get patches
        num_patch_y = num_of_patch_with_overlapping[1] # used for get patches
        num_patch_x = num_of_patch_with_overlapping[2] # used for get patches
        
        print("total number of patches in the image is {0}".format(total_num_of_patches))
        
        center = np.zeros([total_num_of_patches, 3])
        
        patch_index = 0
        
        for ii in range(0, num_patch_z):
            for jj in range(0, num_patch_y):
                for kk in range(0, num_patch_x):
                    center[patch_index] = np.array([int(ii*stride[0] + patch_size[0]//2),
                                                    int(jj*stride[1] + patch_size[1]//2),
                                                    int(kk*stride[2] + patch_size[2]//2)])
                    patch_index += 1
        
        modulo=np.mod(total_num_of_patches, args.val_batch_size)
        if modulo!=0:
            num_to_add = args.val_batch_size-modulo
            inds_to_add = np.random.randint(0, total_num_of_patches, num_to_add)
            to_add = center[inds_to_add]
            new_center = np.vstack((center, to_add))
        else:
            new_center = center
        
        np.random.shuffle(new_center)
        for i_batch in range(int(new_center.shape[0]/args.val_batch_size)):
            label = 0
            subvertex = new_center[i_batch*args.val_batch_size:(i_batch+1)*args.val_batch_size]
            for count in range(args.val_batch_size):
                
                image_one = np.zeros([1, args.input_size[0], args.input_size[1], args.input_size[2]], dtype=np.float32)
                
                z_lower_bound = int(subvertex[count][0] - patch_size[0]//2)
                z_higher_bound = int(subvertex[count][0] + patch_size[0]//2)
                y_lower_bound = int(subvertex[count][1] - patch_size[1]//2)
                y_higher_bound = int(subvertex[count][1] + patch_size[1]//2)
                x_lower_bound = int(subvertex[count][2] - patch_size[2]//2)
                x_higher_bound = int(subvertex[count][2] + patch_size[2]//2)
                
                virgin_range = np.array([z_lower_bound, z_higher_bound, y_lower_bound, y_higher_bound, x_lower_bound, x_higher_bound])
                copy_from, copy_to = corrected_crop(virgin_range, expanded_image_size)

                cf_z_lower_bound = int(copy_from[0])
                if copy_from[1] is not None:
                    cf_z_higher_bound = int(copy_from[1])
                else:
                    cf_z_higher_bound = None
                
                cf_y_lower_bound = int(copy_from[2])
                if copy_from[3] is not None:
                    cf_y_higher_bound = int(copy_from[3])
                else:
                    cf_y_higher_bound = None
                
                cf_x_lower_bound = int(copy_from[4])
                if copy_from[5] is not None:
                    cf_x_higher_bound = int(copy_from[5])
                else:
                    cf_x_higher_bound = None
                
                image_one[:,
                          int(copy_to[0]):copy_to[1],
                          int(copy_to[2]):copy_to[3],
                          int(copy_to[4]):copy_to[5]] = \
                          expanded_image[:,
                                         cf_z_lower_bound:cf_z_higher_bound,
                                         cf_y_lower_bound:cf_y_higher_bound,
                                         cf_x_lower_bound:cf_x_higher_bound]
                
                ## output batch ##
                image_one = np.expand_dims(image_one, axis=0)
                
                if label == 0:
                    Img_1 = image_one
                    label += 1
                else:
                    Img_1 = np.vstack((Img_1, image_one))
                    label += 1
            
            predicted_one = model(torch.from_numpy(Img_1).cuda())
            predicted_one = torch.softmax(predicted_one, dim=1)
            predicted_one = predicted_one.cpu().detach().numpy()
            
            for idx in range(args.val_batch_size):
                
                predicted_seg[:, 
                              np.int16(subvertex[idx][0] - patch_size[0]//2):np.int16(subvertex[idx][0] + patch_size[0]//2),
                              np.int16(subvertex[idx][1] - patch_size[1]//2):np.int16(subvertex[idx][1] + patch_size[1]//2),
                              np.int16(subvertex[idx][2] - patch_size[2]//2):np.int16(subvertex[idx][2] + patch_size[2]//2)] += predicted_one[idx]
    
                count_matrix_seg[:, 
                                 np.int16(subvertex[idx][0] - patch_size[0]//2):np.int16(subvertex[idx][0] + patch_size[0]//2),
                                 np.int16(subvertex[idx][1] - patch_size[1]//2):np.int16(subvertex[idx][1] + patch_size[1]//2),
                                 np.int16(subvertex[idx][2] - patch_size[2]//2):np.int16(subvertex[idx][2] + patch_size[2]//2)] += 1.0
        
        predicted_seg = predicted_seg/(1.0*count_matrix_seg)
        
        output_seg = predicted_seg[:, pad_front[0]:image_size[0]+pad_front[0], pad_front[1]:image_size[1]+pad_front[1], pad_front[2]:image_size[2]+pad_front[2]]
        
        output_label = np.argmax(output_seg, axis=0)
        
        output_label = flip_array(output_label, orientation)
        
        output_image_to_save = sitk.GetImageFromArray(output_label.astype(np.float32))
        output_image_to_save.SetSpacing([args.common_spacing[0], args.common_spacing[1], args.common_spacing[2]])
        output_image_to_save.SetDirection(orientation)
        
        output_image_to_save = resample_lab(output_image_to_save, out_spacing=spacing, out_size=size)
        output_image_to_save.SetSpacing(spacing)
        output_image_to_save.SetDirection(orientation)
        output_image_to_save.SetOrigin(origin)
        
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        
        sitk.WriteImage(output_image_to_save, args.save_path + '/' + '/pred_' + subject)
    
    end = timeit.default_timer()
    print(end - start, 'seconds')

if __name__ == '__main__':
    parser = get_arguments()
    args = parser.parse_args()
    
    main(args)
