"""
Synthetic Multispectral (MS) Data Generation Script

This script trains and runs a GAN to generate synthetic multispectral images from RGB images,
aligning with the project's production-ready data and model APIs.
"""
import argparse
import logging
from pathlib import Path
# A comment is added to recommend proper package installation over sys.path modification.
# To make this script runnable in a production setup, install the project package
# using `pip install -e .` from the project root.
# import sys
# sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.dataset import PlantDiseaseDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MSGenerator(nn.Module):
    """Generator network (U-Net) for RGB to multispectral image translation."""
    def __init__(self, input_channels: int = 3, output_channels: int = 4):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512, padding=1)

        # Decoder with skip connections
        self.dec1 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256 + 256, 128) # Account for skip connection channels
        self.dec3 = self._upconv_block(128 + 128, 64)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Normalize output to [-1, 1]
        )

    def _conv_block(self, in_c, out_c, bn=True, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, padding, bias=False),
            nn.BatchNorm2d(out_c) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d1 = torch.cat([self.dec1(e4), e3], dim=1)
        d2 = torch.cat([self.dec2(d1), e2], dim=1)
        d3 = torch.cat([self.dec3(d2), e1], dim=1)
        return self.dec4(d3)


class MSDiscriminator(nn.Module):
    """Discriminator network for distinguishing real vs. synthetic MS images."""
    def __init__(self, input_channels: int = 4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 1), # Output a patch-based prediction
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


class MSGAN(pl.LightningModule):
    """PyTorch Lightning module for training the RGB-to-MS GAN."""
    def __init__(self, lr: float = 0.0002, b1: float = 0.5, b2: float = 0.999, lambda_l1: float = 100.0):
        super().__init__()
        self.save_hyperparameters()
        self.generator = MSGenerator()
        self.discriminator = MSDiscriminator()
        self.adversarial_loss = nn.BCELoss()
        self.reconstruction_loss = nn.L1Loss()

    def forward(self, rgb_image):
        return self.generator(rgb_image)

    def configure_optimizers(self):
        opt_g = optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        return [opt_g, opt_d], []

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int):
        # FIXED: Use correct keys from the refactored dataset
        rgb_images = batch['image']
        real_ms_images = batch['ms_data']

        # Train Generator
        if optimizer_idx == 0:
            fake_ms_images = self(rgb_images)
            pred_fake = self.discriminator(fake_ms_images)
            valid = torch.ones_like(pred_fake)
            g_adv_loss = self.adversarial_loss(pred_fake, valid)
            g_rec_loss = self.reconstruction_loss(fake_ms_images, real_ms_images)
            g_total_loss = g_adv_loss + self.hparams.lambda_l1 * g_rec_loss
            self.log_dict({'g_loss': g_adv_loss, 'rec_loss': g_rec_loss, 'g_total': g_total_loss}, prog_bar=True)
            return g_total_loss

        # Train Discriminator
        if optimizer_idx == 1:
            fake_ms_images = self(rgb_images).detach()
            pred_real = self.discriminator(real_ms_images)
            loss_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))
            pred_fake = self.discriminator(fake_ms_images)
            loss_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
            d_loss = (loss_real + loss_fake) / 2
            self.log('d_loss', d_loss, prog_bar=True)
            return d_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAN or generate synthetic multispectral images.")
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True)
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the dataset metadata CSV.")
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory for RGB images.")
    parser.add_argument('--ms_dir', type=str, help="Root directory for real multispectral images (required for training).")
    parser.add_argument('--output_dir', type=str, default='output/synthetic_ms', help="Directory to save generated images and logs.")
    parser.add_argument('--checkpoint', type=str, help="Path to a model checkpoint for inference or resuming training.")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=200)
    return parser.parse_args()


def train(args):
    """Sets up and runs the GAN training pipeline."""
    if not args.ms_dir:
        raise ValueError("--ms_dir is required for training mode to provide ground truth.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # The dataset MUST be initialized with both RGB and MS for pix2pix training.
    dataset = PlantDiseaseDataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        ms_dir=args.ms_dir,
        transform=transform, # Assuming same transform for both for now
        use_ms=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = MSGAN()
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='auto',
        devices=1,
        default_root_dir=output_dir / 'logs',
        callbacks=[
            pl.callbacks.ModelCheckpoint(dirpath=output_dir / 'checkpoints', monitor='rec_loss', mode='min', save_top_k=2),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        ],
    )
    trainer.fit(model, dataloader, ckpt_path=args.checkpoint)


def inference(args):
    """Runs inference to generate synthetic MS images from RGB images."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = PlantDiseaseDataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        transform=transform,
        use_ms=False, # For inference, we only need the RGB source images.
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = MSGAN.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating synthetic MS images"):
            # Use the correct key and move data to the correct device
            rgb_images = batch['image'].to(device)
            fake_ms_batch = model(rgb_images)

            # Use the image_path from the batch for robust filename mapping.
            for i, path_str in enumerate(batch['image_path']):
                img_path = Path(path_str)
                output_path = output_dir / f"{img_path.stem}_synthetic_ms.png"

                # Denormalize and convert tensor to a savable image format.
                ms_img_tensor = fake_ms_batch[i].cpu()
                ms_img_np = (ms_img_tensor.numpy() * 0.5 + 0.5) * 255
                ms_img_np = np.clip(ms_img_np, 0, 255).astype('uint8')
                
                # Convert from CHW to HWC format for saving with Pillow
                ms_img_np = np.transpose(ms_img_np, (1, 2, 0))

                # Save as a single 4-channel PNG image.
                if ms_img_np.shape[2] == 4:
                    Image.fromarray(ms_img_np, 'RGBA').save(output_path)
                else: # Fallback for 3-channel
                    Image.fromarray(ms_img_np, 'RGB').save(output_path)


def main():
    """Main entry point for the script."""
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        if not args.checkpoint:
            raise ValueError("--checkpoint path is required for inference mode.")
        inference(args)

if __name__ == '__main__':
    main()