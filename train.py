import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import zipfile


IMG_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {DEVICE}")


# SETUP
from google.colab import drive
drive.mount('/content/drive')


if not os.path.exists("/content/processed_dataset"):
    print("Unzipping dataset...")
    
    zip_path = "/content/drive/MyDrive/MProject/brats_ready.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("/content/")
        print("Data Ready!")
    else:
        print(f"ERROR: Zip file not found at {zip_path}")
else:
    print("Data already unzipped.")

DATASET_PATH = "/content/processed_dataset"
OUTPUT_PATH = "/content/drive/MyDrive/MProject/checkpoints"
os.makedirs(OUTPUT_PATH, exist_ok=True)


# U-NET ARCHITECTURE 
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 256), nn.SiLU(), nn.Linear(256, 256),
        )
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


# UTILS
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

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,), device=DEVICE)


# TRAINING
def train():
    global OUTPUT_PATH 

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_path = os.path.join(DATASET_PATH, "train")
    dataset = datasets.ImageFolder(root=train_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    print(f"Training on {len(dataset)} images at {IMG_SIZE}x{IMG_SIZE}")

    model = UNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    mse = nn.MSELoss()
    diffusion = Diffusion(device=DEVICE)
    scaler = torch.amp.GradScaler('cuda')

    # RESUME LOGIC
    start_epoch = 0

    
    checkpoints = sorted(glob.glob(os.path.join(OUTPUT_PATH, "model_128px_epoch_*.pth")))

    
    if len(checkpoints) == 0:
        print(f"--> No checkpoints in {OUTPUT_PATH}. Searching parent directory for duplicates...")
        parent_dir = os.path.dirname(OUTPUT_PATH)
        
        checkpoints = sorted(glob.glob(os.path.join(parent_dir, "checkpoints*", "model_128px_epoch_*.pth")))

    if len(checkpoints) > 0:
        
        latest_ckpt = checkpoints[-1]

        found_dir = os.path.dirname(latest_ckpt)
        if found_dir != OUTPUT_PATH:
            OUTPUT_PATH = found_dir
            print(f"--> FOUND CHECKPOINTS IN: {OUTPUT_PATH}")
            print("--> Switching save location to this folder.")

        print(f"--> Found checkpoint: {latest_ckpt}")
        print("--> Loading weights and resuming...")

        # Load weights
        model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))

        
        try:
            fname = os.path.basename(latest_ckpt)
            epoch_str = fname.replace('.pth', '').split('_')[-1]
            start_epoch = int(epoch_str)
            print(f"--> Successfully resumed. Starting from Epoch {start_epoch + 1}")
        except:
            print("--> WARNING: Could not parse epoch number. Starting based on loop.")
    else:
        print("--> No checkpoints found anywhere. Starting from Epoch 1.")

    #  Restart Logic
    for epoch in range(start_epoch, EPOCHS):
        pbar = tqdm(dataloader)
        epoch_loss = 0
        for images, _ in pbar:
            images = images.to(DEVICE)
            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.noise_images(images, t)

            with torch.amp.autocast('cuda'):
                predicted_noise = model(x_t, t.float())
                loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader)}")

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, f"model_128px_epoch_{epoch+1}.pth"))
            print(f"Checkpoint Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    train()
