import SimpleITK as sitk
import pandas as pd
import numpy as np
import os

from ID2AbdominalOrgan import ID2Organ

data_path = '/gpfs/fs001/cbica/home/chexiao/MultiOrgan/awesome/MM'

total_classes = 17

col_names = ["ID"]
for organ_id in sorted(ID2Organ.keys()):
    col_names.append(ID2Organ[organ_id])

col_names.remove("background")

print("columns: ", col_names, ", num_columns = ", len(col_names))

val_subjects = sorted(os.listdir(data_path + "/labelsVa"))

def one_hot_encoding(label, numClasses):
    prob_list = []
    for i in range(numClasses):
        temp_prob = label == i
        prob_list.append(np.expand_dims(temp_prob, axis=0))
    return np.concatenate(prob_list, axis=0)

data = []
for idx in range(len(val_subjects)):
    subject_id = val_subjects[idx].split('.')[0]
    print("Subject: ", subject_id)
    
    groundtruth = sitk.ReadImage(data_path + '/labelsVa/' + subject_id + ".nii.gz")
    groundtruth = sitk.GetArrayFromImage(groundtruth)
    
    classes_that_exist, counts = np.unique(groundtruth.astype(np.int32), return_counts=True)
    
    gt = one_hot_encoding(groundtruth, numClasses=total_classes)
    print("gt: ", gt.shape)
    
    prediction = sitk.ReadImage('./best_results/' + 'pred_' + subject_id + '.nii.gz')
    prediction = sitk.GetArrayFromImage(prediction)

    pred = one_hot_encoding(prediction, numClasses=total_classes)
    print("pred: ", pred.shape)
    
    
    dice = [subject_id]
    for idx in range(1, total_classes):
        
        if idx in classes_that_exist and counts[np.where(classes_that_exist == idx)] > 50:
            pred_i = pred[idx]
            gt_i = gt[idx]
            
            intersection = np.sum(pred_i * gt_i)
            dsc = (2. * np.sum(intersection) + 1.0) / (np.sum(pred_i) + np.sum(gt_i) + 1.0)
            
            dice.append(dsc)
        else:
            dice.append(None)
    
    data.append(dice)
    print(dice)
    
    print('\n')

# Create the pandas DataFrame
data = np.array(data)
print(data.shape)
df = pd.DataFrame(data, columns=col_names)

df.to_csv('overall_performance.csv', index=False)
