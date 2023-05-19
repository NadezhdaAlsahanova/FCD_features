import os, time, logging
import numpy as np
from pathlib import Path
import nibabel as nib
from nilearn import image
import warnings
import argparse
import re
from utilities import *
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def CR(brain_path, GM_path, output_dir, type_data='t2'):
    size,num,m = [2, 105, 15]
   
    brain = nib.load(brain_path)
    brain_data = brain.get_fdata()            
    GMfile = nib.load(GM_path).get_fdata() 
    mask_e = nib.load('./masks/exclusive_mask_MNI1mm.nii.gz')
    mask_b = nib.load('./masks/MNI152_T1_1mm_brain_mask.nii.gz')


    mask_data = image.resample_to_img(mask_e, brain, interpolation='nearest').get_fdata()
    brain_mask_data = image.resample_to_img(mask_b, brain, interpolation='nearest').get_fdata()
    new_feature = np.zeros(brain_data.shape)

    for i in np.array(np.where((GMfile>0.6)&(mask_data<0.5))).T:
        if (i>size-1).all() & (i+size<brain_data.shape).all():
            square = brain_data[i[0]-size:i[0]+size+1,i[1]-size:i[1]+size+1,i[2]-size:i[2]+size+1]  
        else:
            continue
        x = np.sort(square.reshape(-1))
        new_feature[i[0],i[1],i[2]] = x[x.shape[0]-num-m:-m].sum()
    new_feature = new_feature/new_feature.max()
    new_feature = new_feature * np.where(brain_mask_data>0.01,1,0)
    img = nib.Nifti1Image(new_feature, brain.affine)
    nib.save(img, os.path.join(output_dir,f'CR_{type_data}.nii.gz'))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for blurring calculation')
    parser.add_argument('--mri_sequence', type=str, help='One of three MRI sequences: t2, flair')
    parser.add_argument('--input_dir', type=str, help='Input directory with preprocessed MRI map, gray and white matter segmentations')
    parser.add_argument('--sub', type=str, help='Subject name')
    parser.add_argument('--output_dir', type=str, help='Output directory where blurring map will be saved')
    args = parser.parse_args()
    
    if args.mri_sequence == 't1':        
        raise Exception("Concentration rate can be calculated only for T2 and Flair!")
    elif args.mri_sequence == 't2':
        brain_path, GM_path, _ = create_paths_t2(args.input_dir, args.sub)
    elif args.mri_sequence == 'flair':
        brain_path, GM_path, _ = create_paths_flair(args.input_dir, args.sub)
    else:
        raise Exception("Wrong name of sequence! Should be one of follows: t2, flair")
        
    
     CR(brain_path, GM_path, args.output_dir, type_data=args.mri_sequence)   
    