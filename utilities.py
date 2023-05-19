import numpy as np
from pathlib import Path
import nibabel as nib
from nilearn import image
import logging
import re

#########################################
# Functions for Renyi entropy estimation
#########################################
def x_log2_x(x):
    """ Return x * log2(x) and 0 if x is 0."""
    results = x * np.log2(x)
    if np.size(x) == 1:
        if np.isclose(x, 0.0):
            results = 0.0
    else:
        results[np.isclose(x, 0.0)] = 0.0
    return results

def renyi_entropy(alpha, X):
    assert alpha >= 0, "Error: renyi_entropy only accepts values of alpha >= 0, but alpha = {}.".format(alpha)  # DEBUG
    if np.isinf(alpha):
        return - np.log2(np.max(X))
    elif np.isclose(alpha, 0):
        return np.log2(len(X))
    elif np.isclose(alpha, 1):
        return - np.sum(x_log2_x(X))
    else:
        return (1.0 / (1.0 - alpha)) * np.log2(np.sum(X ** alpha))
    
###########################################
# Functions to work with files in input_dir
###########################################   

def create_paths_t1(input_dir, sub):
    '''
    Function creates path to all needed files for feature estimation.
    All files should be in input_dir directory. There shoul be next files:
        - preprocessed T1 map, '^{sub}_t1.*brain-final.nii\.gz' ;
        - gray matter probability map for T1, '^c1{sub}_space.*_T1w.*\.nii.*' ;
        - white matter probability map for T1, '^c2{sub}_space.*_T1w.*\.nii.*' ;
    where 'sub' is subject name.
        
    The files should contain 3D maps with shape (197, 233, 189).
    '''
    series_path = os.listdir(input_dir)
    try:
        t1_file = str([i for i in series_path if re.findall(f'^{sub}_t1.*brain-final.nii\.gz', str(i))][0])   
        GM_file_t1 = str([i for i in series_path if re.findall(f'^c1{sub}_space.*_T1w.*\.nii.*', str(i))][0])
        WM_file_t1 = str([i for i in series_path if re.findall(f'^c2{sub}_space.*_T1w.*\.nii.*', str(i))][0])  
        t1f = os.path.join(input_dir, t1_file)
        GM_t1f=os.path.join(input_dir, GM_file_t1)
        WM_t1f=os.path.join(input_dir, WM_file_t1)
        return t1f, GM_t1f, WM_t1f
        
    except BaseException:
        logging.exception(f"Directory {input_dir} does not have needed files!")
        
        
def create_paths_t2(input_dir, sub):
    '''
    Function creates path to all needed files for feature estimation.
    All files should be in input_dir directory. There shoul be next files:
        - preprocessed T2 map, '^{sub}_t2.*brain-final.nii\.gz' ;
        - gray matter probability map for T2, '^c1{sub}_space.*_T2w.*\.nii.*' ;
        - white matter probability map for T2, '^c2{sub}_space.*_T2w.*\.nii.*' ;
    where 'sub' is subject name.
        
    The files should contain 3D maps with shape (197, 233, 189).
    '''
    series_path = os.listdir(input_dir)
    try:
        t2_file = str([i for i in series_path if re.findall(f'^{sub}_t2.*brain-final.nii\.gz', str(i))][0])     
        GM_file_t2 = str([i for i in series_path if re.findall(f'^c1{sub}_space.*_T2w.*\.nii.*', str(i))][0])
        WM_file_t2 = str([i for i in series_path if re.findall(f'^c2{sub}_space.*_T2w.*\.nii.*', str(i))][0]) 
        t2f = os.path.join(input_dir, t2_file)
        GM_t2f=os.path.join(input_dir, GM_file_t2)
        WM_t2f=os.path.join(input_dir, WM_file_t2)
        return t2f, GM_t2f, WM_t2f
        
    except BaseException:
        logging.exception(f"Directory {input_dir} does not have needed files!")
        
        
def create_paths_flair(input_dir, sub):
    '''
    Function creates path to all needed files for feature estimation.
    All files should be in input_dir directory. There shoul be next files:
        - preprocessed FLAIR map, '^{sub}_fl.*brain-final.nii\.gz' ;
        - gray matter probability map for FLAIR, '^c1{sub}_space.*_FLAIR.*\.nii.*' ;
        - white matter probability map for FLAIR, '^c2{sub}_space.*_FLAIR.*\.nii.*' ;
    where 'sub' is subject name.
        
    The files should contain 3D maps with shape (197, 233, 189).
    '''
    series_path = os.listdir(input_dir)
    try:
        fl_file = str([i for i in series_path if re.findall(f'^{sub}_fl.*brain-final.nii\.gz', str(i))][0])     
        GM_file_fl = str([i for i in series_path if re.findall(f'^c1{sub}_space.*_FLAIR.*\.nii.*', str(i))][0])
        WM_file_fl = str([i for i in series_path if re.findall(f'^c2{sub}_space.*_FLAIR.*\.nii.*', str(i))][0])
        flf=os.path.join(input_dir, fl_file)
        GM_flf=os.path.join(input_dir, GM_file_fl)
        WM_flf=os.path.join(input_dir, WM_file_fl)
        return flf, GM_flf, WM_flf
        
    except BaseException:
        logging.exception(f"Directory {input_dir} does not have needed files!")


    
