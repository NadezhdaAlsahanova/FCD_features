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
    
    
def variance(brain_path, GM_path, output_dir):
    size=4
    
    flair_file = nib.load(brain_path)
    flair_data = brain.get_fdata()            
    GMfile = nib.load(GM_path).get_fdata() 
    mask = nib.load('./masks/exclusive_mask_MNI1mm.nii.gz')    
    mask_b = nib.load('./masks/MNI152_T1_1mm_brain_mask.nii.gz')
    mask_data = image.resample_to_img(mask,flair_file, interpolation='nearest').get_fdata()
    brain_mask_data = image.resample_to_img(mask_b,flair_file, interpolation='nearest').get_fdata()
    
    new_feature = np.zeros(flair_data.shape)
    for i in np.array(np.where((GMflfile>0.2)&(mask_data<0.5))).T:
        if (i>size-1).all() & (i+size<flair_data.shape).all():
            square = flair_data[i[0]-size:i[0]+size+1,i[1]-size:i[1]+size+1,i[2]-size:i[2]+size+1]  
        else:
            continue
        new_feature[i[0],i[1],i[2]] = square.std(ddof=1)
    new_feature = new_feature/new_feature.max()
    img = nib.Nifti1Image(new_feature*np.where(brain_mask_data>0.01, 1, 0), flair_file.affine)
    nib.save(img, os.path.join(output_dir,f'Variance.nii.gz')) 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for entropy calculation')
    parser.add_argument('--input_dir', type=str, help='Input directory with preprocessed MRI map, gray and white matter segmentations')
    parser.add_argument('--sub', type=str, help='Subject name')
    parser.add_argument('--output_dir', type=str, help='Output directory where blurring map will be saved')
    args = parser.parse_args()
    
    brain_path, GM_path, _ = create_paths_flair(args.input_dir, args.sub)
        
    
     variance(brain_path, GM_path, args.output_dir)    
    