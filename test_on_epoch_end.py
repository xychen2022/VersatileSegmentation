#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18, 2023

@author: xychen
"""

import os
import torch
import shutil
import numpy as np
import SimpleITK as sitk

def resample_img_lab_pair(itk_image, itk_label, out_spacing=[1.0, 1.0, 3.0], out_size=None):
    
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
    resample1.SetTransform(T)
    
    resample2 = sitk.ResampleImageFilter()
    resample2.SetOutputSpacing(out_spacing)
    resample2.SetSize(out_size)
    resample2.SetOutputDirection(itk_image.GetDirection())
    resample2.SetOutputOrigin(itk_image.GetOrigin())
    resample2.SetTransform(T)
    
    resample1.SetDefaultPixelValue(imageFilter.GetMinimum())
    resample1.SetInterpolator(sitk.sitkLinear)
    
    resample2.SetDefaultPixelValue(0)
    resample2.SetInterpolator(sitk.sitkNearestNeighbor)
    
    resampled_image = resample1.Execute(itk_image)
    resampled_label = resample2.Execute(itk_label)
    
    return resampled_image, resampled_label

def resample_img_single(itk_image, out_spacing=[1.0, 1.0, 3.0], out_size=None):
    
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

def flip_img_lab_array_pair(img_array, lab_array, direction):
    if direction[0] < 0:
        img_array = img_array[:, :, ::-1]
        lab_array = lab_array[:, :, ::-1]
    if direction[4] < 0:
        img_array = img_array[:, ::-1, :]
        lab_array = lab_array[:, ::-1, :]
    if direction[8] < 0:
        img_array = img_array[::-1, :, :]
        lab_array = lab_array[::-1, :, :]
    return img_array, lab_array

def flip_array(array, direction):
    if direction[0] < 0:
        array = array[:, :, ::-1]
    if direction[4] < 0:
        array = array[:, ::-1, :]
    if direction[8] < 0:
        array = array[::-1, :, :]
    return array

def test(args, model, subject_list, snapshot_path, epoch=0, current_best=0.):
    # the following parameters need to be assigned values before training
    batch_size = args.val_batch_size # very important
    patch_size = np.array([args.input_size[0], args.input_size[1], args.input_size[2]]) # very important
    num_of_downpooling = 4 # very important
    patch_stride_regulator = np.array([2, 2, 2]) # this value determines the stride in each dimension when getting the patch; if value = 2, the stride in that dimension is half the value of patch_size
    assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
    stride = patch_size/patch_stride_regulator
    
    tmp_path = os.path.join(os.getcwd(), 'tmp')
    best_path = os.path.join(os.getcwd(), 'best_results')
    
    dice_coef_list = []
    for j in range(len(subject_list)):
        subject_id = subject_list[j]
        
        modality = subject_id.split('_')[1].lower()
        
        image = sitk.ReadImage(os.path.join(args.data_dir, "imagesVa", subject_id))
        
        size = image.GetSize()
        spacing = image.GetSpacing()
        orientation = image.GetDirection()
        origin = image.GetOrigin()
        
        segmentation_gt = sitk.ReadImage(os.path.join(args.data_dir, "labelsVa", subject_id), sitk.sitkUInt32)
        
        image, segmentation_gt = resample_img_lab_pair(image, segmentation_gt, out_spacing=[args.common_spacing[0], args.common_spacing[1], args.common_spacing[2]])
        
        image = sitk.GetArrayFromImage(image)
        segmentation_gt = np.round(sitk.GetArrayFromImage(segmentation_gt)).astype(np.int32)
        
        image, segmentation_gt = flip_img_lab_array_pair(image, segmentation_gt, orientation)
        
        image_size = np.array(np.shape(image), dtype=np.int32)
        
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
        
        image = np.expand_dims(image, axis=0)
        
        expanded_image_size = np.maximum( (np.ceil(image_size/(1.0*stride)) * stride).astype(np.int32), patch_size)
        
        pad_front = (expanded_image_size - image_size) // 2
        
        expanded_image = np.zeros([1,] + list(expanded_image_size), dtype=np.float32)
        
        expanded_image[:,
                       pad_front[0]:(image_size[0]+pad_front[0]),
                       pad_front[1]:(image_size[1]+pad_front[1]),
                       pad_front[2]:(image_size[2]+pad_front[2])] = image
        
        expanded_seg_gt = np.zeros([expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
        expanded_seg_gt[pad_front[0]:(image_size[0]+pad_front[0]),
                        pad_front[1]:(image_size[1]+pad_front[1]),
                        pad_front[2]:(image_size[2]+pad_front[2])] = segmentation_gt
        
        predicted_seg = np.zeros([args.total_classes, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)

        count_matrix_seg = np.zeros([args.total_classes, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
        
        num_of_patch_with_overlapping = (expanded_image_size/stride - patch_stride_regulator + 1).astype(np.int32)
        
        total_num_of_patches = np.prod(num_of_patch_with_overlapping)
        
        num_patch_z = num_of_patch_with_overlapping[0] # used for get patches
        num_patch_y = num_of_patch_with_overlapping[1] # used for get patches
        num_patch_x = num_of_patch_with_overlapping[2] # used for get patches
        
        patch_index = 0
        center = np.zeros([total_num_of_patches, 3], dtype=np.float32)
        
        for ii in range(0, num_patch_z):
            for jj in range(0, num_patch_y):
                for kk in range(0, num_patch_x):
                    center[patch_index] = np.array([int(ii*stride[0] + patch_size[0]//2),
                                                    int(jj*stride[1] + patch_size[1]//2),
                                                    int(kk*stride[2] + patch_size[2]//2)])
                    patch_index += 1
        
        modulo=np.mod(total_num_of_patches, batch_size)
        if modulo!=0:
            num_to_add=batch_size-modulo
            inds_to_add=np.random.randint(0, total_num_of_patches, num_to_add) ## the return value is a ndarray
            to_add = center[inds_to_add]
            new_center = np.vstack((center, to_add))
        else:
            new_center = center
        
        np.random.shuffle(new_center)
        for i_batch in range(int(new_center.shape[0]/batch_size)):
            label = 0
            subvertex = new_center[i_batch*batch_size:(i_batch+1)*batch_size]
            for count in range(batch_size):
                
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
                       
            for idx in range(batch_size):
                
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
        
        categories_present = np.unique(segmentation_gt)
        
        DSCs = []
        for class_i in categories_present:
            
            if class_i != 0:
                label_i = np.zeros_like(output_label, dtype=np.float32)
                label_i[np.where(output_label == class_i)] = 1.
                
                gt_i = np.zeros_like(segmentation_gt, dtype=np.float32)
                gt_i[np.where(segmentation_gt == class_i)] = 1.
                
                DSCs.append((2. * (label_i * gt_i).sum() + 1.0) / (label_i.sum() + gt_i.sum() + 1.0))
        
        if len(DSCs) != 0:
            dice_coef_list.append(np.mean(DSCs))
        else:
            print(subject_id.split('.')[0], " doesn't any labeled organ, so is skipped for DSC evaluation")
        
        output_label = flip_array(output_label, orientation)
        
        output_image_to_save = sitk.GetImageFromArray(output_label.astype(np.float32))
        output_image_to_save.SetSpacing([args.common_spacing[0], args.common_spacing[1], args.common_spacing[2]])
        output_image_to_save.SetDirection(orientation)
        
        output_image_to_save = resample_img_single(output_image_to_save, out_spacing=spacing, out_size=size)
        output_image_to_save.SetSpacing(spacing)
        output_image_to_save.SetDirection(orientation)
        output_image_to_save.SetOrigin(origin)
        
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        
        sitk.WriteImage(output_image_to_save, tmp_path + '/pred_' + subject_id)
    
    print('dice_coef_list: ', dice_coef_list, '\nmean: ', np.mean(dice_coef_list), ' std: ', np.std(dice_coef_list))
    current_mean = np.mean(np.array(dice_coef_list))
    
    if current_mean > current_best:
        if os.path.exists(best_path):
            shutil.rmtree(best_path)
        
        shutil.copytree(tmp_path, best_path)
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict()
                }, snapshot_path+'/best_model.pth')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict()
                }, snapshot_path+'/model_ckpt_epoch{0}_avg{1:3f}.pth'.format(epoch, current_mean))
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
                }, snapshot_path+'/best_model.pth')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
                }, snapshot_path+'/model_ckpt_epoch{0}_avg{1:3f}.pth'.format(epoch, current_mean))
        
        return current_mean
    else:
        
        if current_mean > 0.9:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict()
                    }, snapshot_path+'/model_ckpt_epoch{0}_avg{1:3f}.pth'.format(epoch, current_mean))
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, snapshot_path+'/model_ckpt_epoch{0}_avg{1:3f}.pth'.format(epoch, current_mean))
        
        return current_best

