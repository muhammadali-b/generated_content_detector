
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import clip

# Base paths
BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
REAL_DIR = DATA_DIR / "real"
AI_DIR = DATA_DIR / "ai"

OUTPUT_PATH = BACKEND_DIR / "ml" / "features_cifake.npz"

# Image extensions our system will accept
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def list_image_files(folder: Path) -> List[Path]:
    """
    Scans a folder and return all image files in a folder (non-recursive).

    Args:
        folder (Path): Path of the folder with images.

    Returns:
        List[Path]: a list of file paths.
    """

    files: List[Path] = []
    for entry in folder.iterdir():
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(entry)
    return sorted(files)


def load_clip_model(device: str = "cpu"):
    """Load CLIP ViT-B/32 model and preprocessing."""
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def extract_folder_features(
    folder: Path,
    label: int,
    model,
    preprocess,
    device: str = "cpu",
) -> Tuple[List[np.ndarray], List[int]]:
    """Extract CLIP features for all images in a folder with given label."""
    features: List[np.ndarray] = []
    labels: List[int] = []

    files = list_image_files(folder)
    print(f"Found {len(files)} images in {folder}")

    with torch.no_grad():
        for i, img_path in enumerate(files, start=1):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  [WARN] Could not open {img_path}: {e}")
                continue

            image_tensor = preprocess(img).unsqueeze(0).to(device)

            image_features = model.encode_image(image_tensor)
            # Convert to 1D numpy vector (optionally normalize)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu().numpy().squeeze())
            labels.append(label)

            if i % 50 == 0:
                print(f"  Processed {i} images from {folder}")

    return features, labels


def main():
    # Check that data folders exist
    if not REAL_DIR.exists() or not AI_DIR.exists():
        raise RuntimeError(
            f"Expected data folders {REAL_DIR} and {AI_DIR} to exist. "
            "Make sure you put images there first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP model (ViT-B/32)...")
    model, preprocess = load_clip_model(device=device)
    print("Model loaded.")

    # Extract features for real images (label 0)
    real_features, real_labels = extract_folder_features(
        REAL_DIR, label=0, model=model, preprocess=preprocess, device=device
    )

    # Extract features for AI images (label 1)
    ai_features, ai_labels = extract_folder_features(
        AI_DIR, label=1, model=model, preprocess=preprocess, device=device
    )

    # Combine and shuffle
    X = np.vstack(real_features + ai_features)
    y = np.array(real_labels + ai_labels, dtype=np.int64)

    # Shuffle the dataset (same permutation for X and y)
    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print(f"Final dataset shape: X = {X.shape}, y = {y.shape}")
    print(f"Saving features to {OUTPUT_PATH}")
    np.savez_compressed(OUTPUT_PATH, X=X, y=y)
    print("Done.")


if __name__ == "__main__":
    main()
