"""Dataset and dataloader utilities for the Anime Face Dataset (Phase 1)."""

import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

KAGGLE_DATASET = "splcher/animefacedataset"


def download_dataset() -> str:
    """Downloads the Anime Face Dataset via kagglehub and returns the local root path."""
    import kagglehub

    return kagglehub.dataset_download(KAGGLE_DATASET)


def get_dataloader(batch_size: int, image_size: int = 64, data_dir: str | None = None,
                    num_workers: int = 2) -> DataLoader:
    """Builds a DataLoader over the Anime Face Dataset.

    `data_dir` must point to a directory containing one subfolder per class
    (torchvision.datasets.ImageFolder convention) — the Kaggle dataset ships
    all images under a single `images/` subfolder, which satisfies this as
    one implicit class. If `data_dir` is None, the dataset is downloaded via
    kagglehub first.
    """
    if data_dir is None:
        data_dir = download_dataset()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    loader = get_dataloader(batch_size=64)
    real_batch = next(iter(loader))[0]
    print("Batch shape:", real_batch.shape)  # expect [64, 3, 64, 64]

    grid = vutils.make_grid(real_batch, padding=2, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Real Anime Faces — Sample Batch")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "results", "real_batch_sanity_check.png"))
    plt.show()
