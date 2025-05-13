import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os

from src.models import VAE

DATASET_PATH = "/home/mrozek/ssne-2025l/beta-vae-traffic/trafic_32"
IMAGE_SIZE = 32
BATCH_SIZE = 64
VAL_SPLIT_RATIO = 0.1
LEARNING_RATE = 1e-4
EPOCHS = 200
BETA = 2.0
MODEL_SAVE_PATH = (
    "/home/mrozek/ssne-2025l/beta-vae-traffic/models/traffic_sign_beta_vae.pth"
)
SAVE_EVERY_N_EPOCHS = 5
NUM_IMAGES_TO_LOG = 8

IMG_CHANNELS = 3
LATENT_DIM = 256
BASE_CHANNELS = 32
CHANNEL_MULTIPLIERS = [2, 4, 8]
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1
ENABLE_BATCH_NORM = False


def denormalize_images(images_tensor):
    return (images_tensor + 1) / 2


def generate_images_to_wandb(
    model: VAE,
    epoch_num,
    device,
    num_images=8,
):
    model.eval()
    with torch.no_grad():
        print(f"Generating images for wandb at epoch {epoch_num+1}...")
        generated_images_tensor = model.sample(num_images, device)
        denormalized_images = denormalize_images(generated_images_tensor)

        wandb_images = [wandb.Image(img) for img in denormalized_images]
    return wandb_images


def vae_loss_function(reconstructed_x, x, mu, log_var, beta):
    recon_loss = F.mse_loss(reconstructed_x, x, reduction="sum") / x.shape[0]

    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kld_loss = kld_loss / x.shape[0]

    total_loss = recon_loss + beta * kld_loss

    return total_loss, recon_loss, kld_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hyperparameters = {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "beta": BETA,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "channel_multipliers": CHANNEL_MULTIPLIERS,
        "kernel_size": KERNEL_SIZE,
        "stride": STRIDE,
        "padding": PADDING,
        "val_split_ratio": VAL_SPLIT_RATIO,
    }
    wandb.init(project="beta-vae-traffic-signs", config=hyperparameters)

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    loss_fn = vae_loss_function

    full_dataset = ImageFolder(DATASET_PATH, transform=train_transform)
    dataset_size = len(full_dataset)
    val_size = int(VAL_SPLIT_RATIO * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )
    print(
        f"Loaded dataset from {DATASET_PATH}. Train size: {train_size}, Val size: {val_size}"
    )

    model = VAE(
        in_channels=IMG_CHANNELS,
        model_channels=BASE_CHANNELS,
        latent_dim=LATENT_DIM,
        channel_mult=CHANNEL_MULTIPLIERS,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
        enable_bn=ENABLE_BATCH_NORM,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)  # Opcjonalnie

    start_epoch = 0
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading checkpoint from {MODEL_SAVE_PATH}...")
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch = 0

    print(
        f"Number of parameters in BetaVAE model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print("Starting training...")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss_total = 0.0
        train_loss_recon = 0.0
        train_loss_kld = 0.0

        for i, (batch_images, _) in enumerate(train_loader):
            optimizer.zero_grad()

            x_original = batch_images.to(device)

            output_dict = model(x_original)
            x_reconstructed = output_dict["reconstructed_x"]
            mu = output_dict["mu"]
            log_var = output_dict["log_var"]

            loss, recon_loss, kld_loss = loss_fn(
                x_reconstructed, x_original, mu, log_var, BETA
            )

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_loss_recon += recon_loss.item()
            train_loss_kld += kld_loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, KLD: {kld_loss.item():.4f})"
                )

        avg_train_loss_total = train_loss_total / len(train_loader)
        avg_train_loss_recon = train_loss_recon / len(train_loader)
        avg_train_loss_kld = train_loss_kld / len(train_loader)

        scheduler.step()

        model.eval()
        val_loss_total = 0.0
        val_loss_recon = 0.0
        val_loss_kld = 0.0
        original_samples_for_log = []
        reconstructed_samples_for_log = []

        with torch.no_grad():
            for i, (batch_images, _) in enumerate(val_loader):
                x_original = batch_images.to(device)

                output_dict = model(x_original)
                x_reconstructed = output_dict["reconstructed_x"]
                mu = output_dict["mu"]
                log_var = output_dict["log_var"]

                loss, recon_loss, kld_loss = loss_fn(
                    x_reconstructed, x_original, mu, log_var, BETA
                )

                val_loss_total += loss.item()
                val_loss_recon += recon_loss.item()
                val_loss_kld += kld_loss.item()

                if i == 0 and len(original_samples_for_log) < NUM_IMAGES_TO_LOG:
                    num_to_take = min(
                        NUM_IMAGES_TO_LOG - len(original_samples_for_log),
                        x_original.size(0),
                    )
                    original_samples_for_log.append(x_original[:num_to_take].cpu())
                    reconstructed_samples_for_log.append(
                        x_reconstructed[:num_to_take].cpu()
                    )

        avg_val_loss_total = val_loss_total / len(val_loader)
        avg_val_loss_recon = val_loss_recon / len(val_loader)
        avg_val_loss_kld = val_loss_kld / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(
            f"  Train Loss: {avg_train_loss_total:.4f} (Recon: {avg_train_loss_recon:.4f}, KLD: {avg_train_loss_kld:.4f})"
        )
        print(
            f"  Val Loss:   {avg_val_loss_total:.4f} (Recon: {avg_val_loss_recon:.4f}, KLD: {avg_val_loss_kld:.4f})"
        )
        images = generate_images_to_wandb(
            model,
            epoch,
            device,
            num_images=NUM_IMAGES_TO_LOG,
        )

        log_dict = {
            "train/total_loss": avg_train_loss_total,
            "train/recon_loss": avg_train_loss_recon,
            "train/kld_loss": avg_train_loss_kld,
            "val/total_loss": avg_val_loss_total,
            "val/recon_loss": avg_val_loss_recon,
            "val/kld_loss": avg_val_loss_kld,
            "generated_images": images,
        }

        wandb.log(log_dict)

        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == EPOCHS:
            print(f"Saving model checkpoint at epoch {epoch+1}...")
            save_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": hyperparameters,
                "enable_bn": ENABLE_BATCH_NORM,
            }
            try:
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save(save_data, MODEL_SAVE_PATH)
                print(f"Checkpoint saved to {MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

    wandb.finish()
    print("Training finished.")
    print(f"Final model checkpoint saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
