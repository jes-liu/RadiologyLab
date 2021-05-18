# Jesse Liu

import os
import nibabel as nib
import numpy as np
from deepbrain import Extractor
# Make sure to use tensorflow==1.14

img_dir = r'C:\Users\jesse\OneDrive\Desktop\Research\PD'
img_file = '\\raw_files\\PD'
filelist = os.listdir(img_dir + img_file)

for i in range(len(filelist)):

    # get subject id's
    name = filelist[i][0:9]  # [5:9] for id

    # skullstrip images
    img_path = img_dir + img_file + '\\' + filelist[i]

    img = nib.load(img_path).get_fdata()
    img_length, img_width, img_height = img.shape[0], img.shape[1], img.shape[2]
    img_array = np.array(img)

    ext = Extractor()
    brain = ext.run(img)
    mask = brain > 0.5

    mask_array = np.array(mask, dtype=int)
    mask_array = mask_array.reshape([img_length, img_width, img_height])
    img_array = img_array.reshape([img_length, img_width, img_height])

    skull_array = mask_array*img_array
    new_img = nib.Nifti1Image(skull_array, None)

    new_file = '\\skullstripped_files\\PD'
    location = img_dir + new_file
    output = os.path.join(location, ('%s_skullstripped.nii' % name))
    nib.save(new_img, output)
