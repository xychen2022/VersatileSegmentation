#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18, 2023

@author: xychen
"""

import os
import pickle
import numpy as np
import SimpleITK as sitk

datasets_for_training = ["amos", "btcv", "flare22", "abdomenct1k", "totalsegmentator", "nihpancreas", "word", "urogram122"]
modalities_for_training = ["ct", "t1w"]

data_path = "/mnt/yapdata/data/xychen/awesome/MM/labelsTr"
files = os.listdir(data_path)

hierarchy = {}
for file in files:
    
    label = sitk.ReadImage(os.path.join(data_path, file))
    label = sitk.GetArrayFromImage(label)
    
    categories_present = np.sort(np.unique(np.round(label).astype(np.int32)))
    
    print("Current file = ", file, " have annotations ", categories_present)
    
    modality = file.split('_')[1]
    dataset = file.split('_')[0]
    
    if modality not in modalities_for_training or dataset not in datasets_for_training:
        continue
    
    for category in categories_present:
        
        if category not in hierarchy.keys():
            hierarchy[category] = {}
        
        if modality not in hierarchy[category].keys():
            hierarchy[category][modality] = {}
    
        if dataset not in hierarchy[category][modality].keys():
            hierarchy[category][modality][dataset] = []
            
        if file not in hierarchy[category][modality][dataset]:
            hierarchy[category][modality][dataset].append(file)


with open('tree_repr_8sets_mm.pkl', 'wb') as f:
    pickle.dump(hierarchy, f)

