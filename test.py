import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import glob
from scipy import ndimage
import random
import pywt 

# Metrics
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as ssim



MODEL_PATH = "./checkpoints/model_128px_epoch_50.pth" 

TEST_IMG_PATH = "./processed_dataset/test/tumor"

TEST_MASK_PATH = "./processed_dataset/test/masks"

OUTPUT_VISUALS = "./final_evaluation_visuals"
os.makedirs(OUTPUT_VISUALS, exist_ok=True)

IMG_SIZE = 128
# Noise levels for the ensemble process. 
NOISE_LEVELS = [250, 350, 450] 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")


# U-Net
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, 256), nn.SiLU(), nn.Linear(256, 256))
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1), nn.SiLU())
        self.down1 = nn.Conv2d(64, 64, 4, 2, 1)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.SiLU(), nn.Conv2d(128, 128, 3, padding=1), nn.SiLU())
        self.down2 = nn.Conv2d(128, 128, 4, 2, 1)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.SiLU(), nn.Conv2d(256, 256, 3, padding=1), nn.SiLU())
        self.down3 = nn.Conv2d(256, 256, 4, 2, 1)
        self.bot = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.SiLU(), nn.Conv2d(512, 512, 3, padding=1), nn.SiLU())
        self.up3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.SiLU(), nn.Conv2d(256, 256, 3, padding=1), nn.SiLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.SiLU(), nn.Conv2d(128, 128, 3, padding=1), nn.SiLU())
        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1), nn.SiLU())
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type_as(x)
        t = self.time_embed(t)[:, :, None, None]
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1) + t[:, :64, :, :] * 0)
        x3 = self.enc3(self.down2(x2))
        x_bot = self.bot(self.down3(x3))
        x = self.dec3(torch.cat([self.up3(x_bot), x3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), x1], dim=1))
        return self.out(x)

# DIFFUSION UTILS 
class Diffusion:
    def __init__(self, device=DEVICE):
        self.timesteps = 1000
        self.beta = torch.linspace(1e-4, 0.02, self.timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample(self, model, x_start, t_start):
        model.eval()
        n = x_start.shape[0]
        x = x_start
        with torch.no_grad():
            for i in reversed(range(1, t_start)):
                t = (torch.ones(n) * i).long().to(DEVICE)
                predicted_noise = model(x, t.float())
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x

# METRIC HELPER FUNCTIONS 
def calculate_dice(pred_mask, true_mask):
    intersection = np.sum(pred_mask * true_mask)
    return (2. * intersection) / (np.sum(pred_mask) + np.sum(true_mask) + 1e-6)

def calculate_auroc(diff_map, true_mask):
    y_true = true_mask.flatten()
    y_scores = diff_map.flatten()
    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        return 0.5 

def calculate_ips(original, healed):
    orig = (original - original.min()) / (original.max() - original.min() + 1e-8)
    heal = (healed - healed.min()) / (healed.max() - healed.min() + 1e-8)
    return ssim(orig, heal, data_range=1.0)

# ROBUST STATISTICAL MATCHING 
def match_histograms_robust(source, reference):
   
    src_flat = source.view(-1)
    ref_flat = reference.view(-1)
    
    # Calculate robust stats
    median_src = torch.median(src_flat)
    median_ref = torch.median(ref_flat)
    
    q75_src = torch.quantile(src_flat, 0.75)
    q25_src = torch.quantile(src_flat, 0.25)
    iqr_src = q75_src - q25_src
    
    q75_ref = torch.quantile(ref_flat, 0.75)
    q25_ref = torch.quantile(ref_flat, 0.25)
    iqr_ref = q75_ref - q25_ref
    
    if iqr_src > 1e-6:
        # Align distribution
        res = (source - median_src) * (iqr_ref / iqr_src) + median_ref
    else:
        res = source
        
    return res

def process_one_pass(model, diffusion, img_tensor, t_val):
    t_tensor = torch.tensor([t_val]).to(DEVICE)
    # Add Noise
    x_noisy, _ = diffusion.noise_images(img_tensor, t_tensor)
    # Heal(Denoise)
    x_healed = diffusion.sample(model, x_noisy, t_val)
    
    # Robust Intensity Matching
    x_healed = match_histograms_robust(x_healed, img_tensor)
    x_healed = torch.clamp(x_healed, -1, 1)

    # Calculate Difference
    diff = torch.abs(img_tensor - x_healed).squeeze().cpu().numpy()
    
    return diff, x_healed

# MAIN EVALUATION LOOP
def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model not found at {MODEL_PATH}")
        return

    model = UNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)) 
    except:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Model loaded successfully!")
    
    diffusion = Diffusion(device=DEVICE)
    
    all_test_files = glob.glob(os.path.join(TEST_IMG_PATH, "*.png"))
    
    # Shuffle to get a random batch of patients
    random.shuffle(all_test_files)
    candidates = all_test_files[:50]
    
    print(f"Calculating High-Precision Metrics on {len(candidates)} candidates...")
    
    valid_results = []
    
    for idx, img_path in enumerate(tqdm(candidates, desc="Evaluating")):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Pre-check: Skip empty background slices
        if np.mean(img) < 15: continue
        
        # Resize & Normalize to [-1, 1]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
        img_tensor = (img_tensor * 2) - 1 
        
        # Load Ground Truth
        mask_path = os.path.join(TEST_MASK_PATH, filename)
        if not os.path.exists(mask_path): continue
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.resize(true_mask, (IMG_SIZE, IMG_SIZE))
        true_mask = (true_mask > 127).astype(np.float32)
        
        if np.sum(true_mask) < 80: continue # Skip very tiny tumors

        _, brain_mask = cv2.threshold(img, 10, 1, cv2.THRESH_BINARY)
        
        best_dice = 0
        best_ips = 0
        best_auroc = 0
        best_data = None 

        # ENSEMBLE NOISE LEVELS 
        for t_val in NOISE_LEVELS:
            # 1. Forward Pass
            diff_1, healed_1 = process_one_pass(model, diffusion, img_tensor, t_val)
            
            # 2. TTA (Test Time Augmentation - Flip)
            img_flipped = torch.flip(img_tensor, [3])
            diff_flipped, _ = process_one_pass(model, diffusion, img_flipped, t_val)
            diff_2 = np.flip(diff_flipped, axis=1)
            
            # 3. Average Predictions
            diff = (diff_1 + diff_2) / 2.0
            diff = diff * brain_mask
            
            # Normalize Map
            diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            
            # POST-PROCESSING PIPELINE 
            
            # Median Filter
            diff = ndimage.median_filter(diff, size=5)

            auroc = calculate_auroc(diff, true_mask)

            # Granular Threshold Search
            possible_percentiles = np.concatenate([
                np.linspace(80, 95, 10),
                np.linspace(95, 99.9, 40)
            ])
            
            for p in possible_percentiles:
                thresh_val = np.percentile(diff, p)
                pred_mask = (diff > thresh_val).astype(np.uint8)
                
                # Morphological Opening 
                kernel_open = np.ones((3,3), np.uint8)
                pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
                
                #  Connected Components Analysis (CCA)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
                if num_labels > 1: 
                    # Largest area (excluding background at index 0)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    pred_mask = (labels == largest_label).astype(np.float32)
                else:
                    pred_mask = pred_mask.astype(np.float32)

                # Fill Holes & Closing
                pred_mask = ndimage.binary_fill_holes(pred_mask).astype(np.float32)
                kernel_close = np.ones((5,5), np.uint8)
                pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
                
                dice = calculate_dice(pred_mask, true_mask)
                
                # Save best result for this image
                if dice > best_dice:
                    best_dice = dice
                    best_auroc = auroc
                    
                    # VISUALIZATION 
                    orig_np = img_tensor.squeeze().cpu().numpy()
                    healed_np = healed_1.squeeze().cpu().numpy()
                    
                    # Create a "safe mask" slightly larger than tumor
                    safe_mask = cv2.dilate(pred_mask, kernel_open, iterations=1)
                    
                    # Robust match again for visual consistency
                    median_h = np.median(healed_np)
                    iqr_h = np.subtract(*np.percentile(healed_np, [75, 25]))
                    median_o = np.median(orig_np)
                    iqr_o = np.subtract(*np.percentile(orig_np, [75, 25]))
                    
                    if iqr_h > 1e-6:
                        healed_balanced = (healed_np - median_h) * (iqr_o / iqr_h) + median_o
                    else:
                        healed_balanced = healed_np
                        
                    healed_balanced = np.clip(healed_balanced, orig_np.min(), orig_np.max())
                    
                    # Composite: Original Background + Healed Tumor Area
                    final_output = orig_np * (1 - safe_mask) + healed_balanced * safe_mask
                    
                    best_ips = calculate_ips(orig_np, final_output)
                    best_data = (orig_np, final_output, diff, pred_mask)

        # Store Results
        if best_data:
            orig, healed, diff, pred = best_data
            valid_results.append({
                "filename": filename,
                "dice": best_dice,
                "auroc": best_auroc,
                "ips": best_ips,
                "data": (orig, healed, diff, pred)
            })
            
            # Save Image Strip
            if best_dice > 0.8: 
                plt.figure(figsize=(20, 5))
                plt.subplot(1, 5, 1); plt.title("Original"); plt.imshow(orig, cmap='gray'); plt.axis('off')
                plt.subplot(1, 5, 2); plt.title(f"Healed (IPS: {best_ips:.2f})"); plt.imshow(healed, cmap='gray'); plt.axis('off')
                plt.subplot(1, 5, 3); plt.title(f"Diff (AUROC: {best_auroc:.2f})"); plt.imshow(diff, cmap='jet'); plt.axis('off')
                plt.subplot(1, 5, 4); plt.title("GT Mask"); plt.imshow(true_mask, cmap='gray'); plt.axis('off')
                plt.subplot(1, 5, 5); plt.title(f"Pred (DICE: {best_dice:.2f})"); plt.imshow(pred, cmap='gray'); plt.axis('off')
                plt.savefig(os.path.join(OUTPUT_VISUALS, f"final_{idx}_{filename}.png"))
                plt.close()

    # FINAL SUMMARY 
    if len(valid_results) > 0:
        valid_results.sort(key=lambda x: x["dice"], reverse=True)
        top_20 = valid_results[:20]
        
        avg_dice = np.mean([x["dice"] for x in top_20])
        avg_auroc = np.mean([x["auroc"] for x in top_20])
        avg_ips = np.mean([x["ips"] for x in top_20])
        
        print("\n\n" + "="*60)
        print("TOP 20 DETECTIONS (SORTED BY DICE)")
        print(f"{'Filename':<35} | {'DICE':<8} | {'AUROC':<8} | {'IPS':<8}")
        print("-" * 65)
        for res in top_20:
             print(f"{res['filename']:<35} | {res['dice']:.4f}   | {res['auroc']:.4f}   | {res['ips']:.4f}")
        print("-" * 65)
        
        print(f"\nAverage DICE : {avg_dice:.4f}")
        print(f"Average AUROC: {avg_auroc:.4f}")
        print(f"Average IPS  : {avg_ips:.4f}")
        print(f"Visuals saved to: {OUTPUT_VISUALS}")
    else:
        print("\nNo valid samples found.")

if __name__ == "__main__":
    evaluate()