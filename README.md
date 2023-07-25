# FCD_features

Code for paper: Alsahanova N. et al. "Automated focal cortical dysplasia detection on brain MR images using image analysis".

Code includes .py files for calculation of features:

1. Blurring
2. Concentration Rate
3. Entropy
4. Variance

The features can be calculated for preprocessed magnetic resonance images. Preprocessing includes the following steps:
- convertion to the Neuroimaging Informatics Technology Initiative (NIfTI) format;
- registration of T1w NIfTI images to symmetric template (MNI152);
- translation of T2w and FLAIR images to MNI space with evaluated linear mapping on T1w;
- preprocessing with SPM12;
- generation of three tissue probability maps (GM, WM, Cerebrospinal fluid (CSF)); 
- extraction of brain mask and brain volume with HD-BET (deep learning skull-stripping tool).
