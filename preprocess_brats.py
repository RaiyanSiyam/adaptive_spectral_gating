import os
import glob
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm


BRATS_PATH = "D:\Datasets\BraTS2021_Training_Data" 


OUTPUT_PATH = "./processed_dataset"


MODALITY = "flair" 

# Skip empty black slices (start/end of scan)
MIN_BRAIN_PIXELS = 1000 


def normalize_slice(slice_img):
    """Normalizes MRI slice to 0-255 range for PNG saving."""
    if np.max(slice_img) == 0:
        return slice_img
    slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
    slice_img = (slice_img * 255).astype(np.uint8)
    return slice_img

def process_brats():
    # Create directories
    train_dir = os.path.join(OUTPUT_PATH, "train", "healthy")
    test_dir = os.path.join(OUTPUT_PATH, "test", "tumor")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Find all patient folders
    patient_folders = glob.glob(os.path.join(BRATS_PATH, "BraTS2021_*"))
    print(f"Found {len(patient_folders)} patients. Starting processing...")

    for patient_path in tqdm(patient_folders):
        patient_id = os.path.basename(patient_path)
        
        # Construct file paths
        image_path = os.path.join(patient_path, f"{patient_id}_{MODALITY}.nii.gz")
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")

        if not os.path.exists(image_path) or not os.path.exists(seg_path):
            continue

        # Load 3D volumes
        img_vol = nib.load(image_path).get_fdata()
        seg_vol = nib.load(seg_path).get_fdata()

        # Iterate through slices (Axis 2 is usually the axial plane)
        num_slices = img_vol.shape[2]
        
        for i in range(num_slices):
            slice_img = img_vol[:, :, i]
            slice_seg = seg_vol[:, :, i]

            # Filter 1: Ignore slices with very little brain (top/bottom of head)
            if np.count_nonzero(slice_img) < MIN_BRAIN_PIXELS:
                continue

            # Normalize image to 0-255
            processed_img = normalize_slice(slice_img)

            # Filter 2: Check for Tumor
            if np.max(slice_seg) == 0:
                # HEALTHY SLICE -> GOES TO TRAINING SET
                save_name = os.path.join(train_dir, f"{patient_id}_slice_{i}.png")
                cv2.imwrite(save_name, processed_img)
            else:
                # TUMOR SLICE -> GOES TO TEST SET
                save_name = os.path.join(test_dir, f"{patient_id}_slice_{i}.png")
                cv2.imwrite(save_name, processed_img)

    print(f"\nProcessing Complete!")
    print(f"Healthy slices saved to: {train_dir}")
    print(f"Tumor slices saved to: {test_dir}")

if __name__ == "__main__":
    process_brats()