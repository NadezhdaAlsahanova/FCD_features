import os, time, logging
import numpy as np
from pathlib import Path
import nibabel as nib
from nilearn import image
import warnings
import argparse
from torch.nn import functional as F
import torch
import re
from utilities import *
warnings.filterwarnings("ignore", category=DeprecationWarning) 

    
def blurring(brain_path, GM_path, WM_path, output_dir, type_data='T1'):

    brain = nib.load(brain_path)
    brain_data = brain.get_fdata()            
    GM_data = nib.load(GM_path).get_fdata() 
    WM_data = nib.load(WM_path).get_fdata() 
    mask_e = nib.load('./masks/exclusive_mask_MNI1mm.nii.gz')
    mask_b = nib.load('./masks/MNI152_T1_1mm_brain_mask.nii.gz')

    brain_mask_data = image.resample_to_img(mask_b,brain, interpolation='nearest').get_fdata()
    mask_e_data = image.resample_to_img(mask_e,brain, interpolation='nearest').get_fdata()
    brain_mask_data = np.where((brain_mask_data>0.5)&(mask_e_data<0.5),1,0) 

    if type_data=='t1':
        coef = 0.4
        threshold1 = brain_data[np.where((GM_data > 0.5))].mean() + coef * brain_data[np.where((GM_data > 0.5))].std() 
        threshold2 = brain_data[np.where((WM_data > 0.5))].mean() - coef * brain_data[np.where((WM_data > 0.5))].std()
    elif type_data=='t2':        
        coef = 0.03 
        threshold1 = brain_data[np.where((WM_data > 0.5))].mean() + coef * brain_data[np.where((WM_data > 0.5))].std()
        threshold2 = brain_data[np.where((GM_data > 0.5))].mean() - coef * brain_data[np.where((GM_data > 0.5))].std() 
    else:
        coef = 0.04 
        threshold1 = brain_data[np.where((WM_data > 0.5))].mean() + coef * brain_data[np.where((WM_data > 0.5))].std()
        threshold2 = brain_data[np.where((GM_data > 0.5))].mean() - coef * brain_data[np.where((GM_data > 0.5))].std() 

    if threshold2<threshold1:
        print('Problem with thresholds! The feature map will be zeros.')

    brain_data = np.where((brain_data < threshold2)&(brain_data > threshold1)&(brain_mask_data>0.01), 1, 0)
    shape = brain_data.shape
    brain_data = torch.Tensor(brain_data.reshape(1,1,shape[0],shape[1],shape[2]))
    brain_data = F.conv3d(brain_data ,torch.ones(1,1,5,5,5), None, 1, 2).reshape(shape[0],shape[1],shape[2]).numpy()
    brain_data = brain_data*np.where(brain_mask_data>0.01, 1, 0)
    brain_data = nib.Nifti1Image(brain_data, brain.affine)
    nib.save(brain_data, os.path.join(output_dir,f'Blurring_{type_data}.nii.gz'))
    
    return None

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for blurring calculation')
    parser.add_argument('--mri_sequence', type=str, help='One of three MRI sequences: t1, t2, flair')
    parser.add_argument('--input_dir', type=str, help='Input directory with preprocessed MRI map, gray and white matter segmentations')
    parser.add_argument('--sub', type=str, help='Subject name')
    parser.add_argument('--output_dir', type=str, help='Output directory where blurring map will be saved')
    args = parser.parse_args()
    
    if args.mri_sequence == 't1':
        brain_path, GM_path, WM_path = create_paths_t1(args.input_dir, args.sub)
    elif args.mri_sequence == 't2':
        brain_path, GM_path, WM_path = create_paths_t2(args.input_dir, args.sub)
    elif args.mri_sequence == 'flair':
        brain_path, GM_path, WM_path = create_paths_flair(args.input_dir, args.sub)
    else:
        raise Exception("Wrong name of sequence! Should be one of follows: t1, t2, flair")
        
    
     blurring(brain_path, GM_path, WM_path, args.output_dir, type_data=args.mri_sequence)   
    