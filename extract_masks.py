import os
import glob
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

BRATS_RAW_PATH = "D:\Datasets\BraTS2021_Training_Data" 
PROCESSED_PATH = "./processed_dataset"

def extract_masks():
    tumor_img_dir = os.path.join(PROCESSED_PATH, "test", "tumor")
    mask_out_dir = os.path.join(PROCESSED_PATH, "test", "masks")
    os.makedirs(mask_out_dir, exist_ok=True)

    tumor_files = glob.glob(os.path.join(tumor_img_dir, "*.png"))
    
    print(f"Found {len(tumor_files)} test images. Extracting corresponding masks...")

    for file_path in tqdm(tumor_files):
        filename = os.path.basename(file_path)
        
        parts = filename.split('_')
        patient_id = f"{parts[0]}_{parts[1]}" 
        slice_idx = int(parts[3].split('.')[0])

        # Find the Segmentation file for patient
        seg_path = os.path.join(BRATS_RAW_PATH, patient_id, f"{patient_id}_seg.nii.gz")
        
        if not os.path.exists(seg_path):
            print(f"Warning: Could not find seg file for {patient_id}")
            continue

        # Load volume and get specific slice
        seg_vol = nib.load(seg_path).get_fdata()
        slice_seg = seg_vol[:, :, slice_idx]

        # Convert to binary mask (0 = Healthy, 255 = Tumor)
        binary_mask = np.where(slice_seg > 0, 255, 0).astype(np.uint8)

        cv2.imwrite(os.path.join(mask_out_dir, filename), binary_mask)

    print("Success! Masks saved to:", mask_out_dir)

if __name__ == "__main__":
    extract_masks()