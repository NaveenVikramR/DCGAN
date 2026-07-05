"""DCGAN training loop (Phase 3 / Phase 4)."""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from data import get_dataloader
from model import LATENT_DIM, Discriminator, Generator, weights_init

REAL_LABEL = 1.0
FAKE_LABEL = 0.0
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def train(dataloader, num_epochs, device, results_dir=DEFAULT_RESULTS_DIR, log_every=50,
          lr=0.0002, betas=(0.5, 0.999), latent_dim=LATENT_DIM):
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    netG = Generator(latent_dim=latent_dim).to(device)
    netG.apply(weights_init)
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas)

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    history = {"D_loss": [], "G_loss": [], "D_x": [], "D_G_z": []}

    step = 0
    for epoch in range(num_epochs):
        for i, (real, _) in enumerate(dataloader):
            real = real.to(device)
            b_size = real.size(0)

            # --- Discriminator step: real batch (label=1) + fake batch (label=0) ---
            netD.zero_grad()

            label_real = torch.full((b_size,), REAL_LABEL, device=device)
            output_real = netD(real)
            loss_d_real = criterion(output_real, label_real)
            loss_d_real.backward()
            d_x = output_real.mean().item()

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = netG(noise)
            label_fake = torch.full((b_size,), FAKE_LABEL, device=device)
            output_fake = netD(fake.detach())
            loss_d_fake = criterion(output_fake, label_fake)
            loss_d_fake.backward()
            d_g_z1 = output_fake.mean().item()

            loss_d = loss_d_real + loss_d_fake
            optimizerD.step()

            # --- Generator step: fool D on the same fakes (non-saturating trick, label=1) ---
            netG.zero_grad()
            label_gen = torch.full((b_size,), REAL_LABEL, device=device)
            output = netD(fake)
            loss_g = criterion(output, label_gen)
            loss_g.backward()
            d_g_z2 = output.mean().item()
            optimizerG.step()

            history["D_loss"].append(loss_d.item())
            history["G_loss"].append(loss_g.item())
            history["D_x"].append(d_x)
            history["D_G_z"].append(d_g_z2)

            if step % log_every == 0:
                print(f"[epoch {epoch}/{num_epochs}][step {i}/{len(dataloader)}] "
                      f"D_loss: {loss_d.item():.4f} G_loss: {loss_g.item():.4f} "
                      f"D(x): {d_x:.4f} D(G(z)): {d_g_z1:.4f}/{d_g_z2:.4f}")

            step += 1

        with torch.no_grad():
            fake_fixed = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake_fixed, os.path.join(samples_dir, f"epoch_{epoch:02d}.png"),
                           normalize=True, nrow=8)

        # Overwrite checkpoints/loss curve each epoch so a mid-run disconnect
        # (e.g. Colab timeout) only loses progress since the last epoch, not the whole run.
        checkpoints_dir = os.path.join(results_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        torch.save(netG.state_dict(), os.path.join(checkpoints_dir, "generator.pth"))
        torch.save(netD.state_dict(), os.path.join(checkpoints_dir, "discriminator.pth"))
        plot_losses(history, os.path.join(results_dir, "loss_curve.png"))

    return netG, netD, history


def plot_losses(history, save_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(history["G_loss"], label="G")
    plt.plot(history["D_loss"], label="D")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run a few hundred steps on a small subset to sanity-check the loop.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataloader = get_dataloader(batch_size=args.batch_size, image_size=args.image_size, data_dir=args.data_dir)

    epochs = args.epochs
    if args.smoke_test:
        from torch.utils.data import DataLoader, Subset

        subset = Subset(dataloader.dataset, range(min(2000, len(dataloader.dataset))))
        dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        epochs = 5
        args.log_every = 10

    train(dataloader, epochs, device, results_dir=args.results_dir, log_every=args.log_every)

    print("Training complete. Checkpoints, sample grids, and loss curve saved to", args.results_dir)


if __name__ == "__main__":
    main()
