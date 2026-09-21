import argparse
import os

import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt


# Kernel builders
def box_kernel(size: int) -> np.ndarray:
    """Normalized box (mean) filter kernel of shape (size, size)."""
    return np.ones((size, size), dtype=np.float64) / (size * size)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Normalized 2D Gaussian kernel of shape (size, size)."""
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


# (A) SPATIAL DOMAIN: direct convolution
def circular_convolution_spatial(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return ndimage.convolve(image, kernel, mode="wrap")


# (B) FREQUENCY DOMAIN: multiply Fourier transforms
def circular_convolution_frequency(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    padded_kernel = np.zeros_like(image, dtype=np.float64)
    padded_kernel[:kh, :kw] = kernel
    padded_kernel = np.roll(padded_kernel, -(kh // 2), axis=0)
    padded_kernel = np.roll(padded_kernel, -(kw // 2), axis=1)

    F = np.fft.fft2(image)
    H = np.fft.fft2(padded_kernel)
    result = np.fft.ifft2(F * H)
    return np.real(result)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def synthetic_pattern() -> np.ndarray:
    n = 256
    xx, yy = np.meshgrid(np.arange(n), np.arange(n))
    img = (((xx // 16) + (yy // 16)) % 2) * 255.0
    img[100:156, 100:156] = 255.0  # a solid square for edge inspection
    return img


def load_grayscale(path: str | None, sample: str = "synthetic") -> np.ndarray:
    if path is not None:
        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=np.float64)

    if sample == "synthetic":
        return synthetic_pattern()

    from skimage import data, color, img_as_float

    if sample == "cameraman":
        arr = data.camera()  # already grayscale
    elif sample == "baboon":
        try:
            arr = data.mandrill()  # some versions do have it
        except AttributeError:
            arr = color.rgb2gray(data.astronaut()) * 255.0
    else:
        raise ValueError(f"Unknown --sample '{sample}'")

    return np.asarray(arr, dtype=np.float64)


def save_gray(arr: np.ndarray, path: str) -> None:
    clipped = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(path)

# Main
def main():
    parser = argparse.ArgumentParser(description="Spatial vs frequency-domain blurring")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument(
        "--sample",
        choices=["synthetic", "cameraman", "baboon"],
        default="synthetic",
        help="Built-in test image, used when --image is not given.",
    )
    parser.add_argument("--kernel", choices=["box", "gaussian"], default="gaussian")
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--sigma", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    image = load_grayscale(args.image, args.sample)
    kernel = (
        gaussian_kernel(args.size, args.sigma)
        if args.kernel == "gaussian"
        else box_kernel(args.size)
    )

    blur_spatial = circular_convolution_spatial(image, kernel)
    blur_frequency = circular_convolution_frequency(image, kernel)

    diff = np.abs(blur_spatial - blur_frequency)
    mse = float(np.mean(diff ** 2))
    max_err = float(np.max(diff))

    print("=" * 60)
    print(f"Kernel: {args.kernel}, size={args.size}, sigma={args.sigma}")
    print(f"MSE between spatial and frequency-domain results : {mse:.3e}")
    print(f"Max absolute pixel error                         : {max_err:.3e}")
    print("(Both should be ~0, i.e. numerical noise only, confirming")
    print(" that spatial convolution == frequency-domain multiplication.)")
    print("=" * 60)

    save_gray(image, "output/01_original.png")
    save_gray(blur_spatial, "output/02_blur_spatial.png")
    save_gray(blur_frequency, "output/03_blur_frequency.png")
    save_gray(diff / (diff.max() + 1e-9) * 255.0, "output/04_difference_map.png")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["Original", "Spatial-domain blur", "Frequency-domain blur", "|Difference| (scaled)"]
    imgs = [image, blur_spatial, blur_frequency, diff]
    for ax, title, im in zip(axes, titles, imgs):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("output/comparison_figure.png", dpi=150)
    print("Saved figures to ./output/")


if __name__ == "__main__":
    main()