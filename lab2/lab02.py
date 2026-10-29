"""
Lab Task 02 - Digital Images, Pixels, Color, and Basic Transformations
Digital Image Processing Lab - University of Azad Jammu and Kashmir
"""

import sys
import cv2 as cv
import numpy as np
import matplotlib

matplotlib.use("Agg")  # save figures to files (works with or without a display)
import matplotlib.pyplot as plt

IMAGE_PATH = "my_image.png"

# ---------------------------------------------------------------- Step 1
print("=" * 60)
print("STEP 1: ENVIRONMENT CHECK")
print("=" * 60)
print("Python executable :", sys.executable)
print("OpenCV version    :", cv.__version__)
print("NumPy version     :", np.__version__)
print("Matplotlib version:", matplotlib.__version__)

# ---------------------------------------------------------------- Step 2
print("\n" + "=" * 60)
print("STEP 2: LOAD THE IMAGE")
print("=" * 60)
image = cv.imread(IMAGE_PATH)  # OpenCV loads color images in BGR order
if image is None:
    raise FileNotFoundError(f"Image not found or unreadable: {IMAGE_PATH}")
print("Image shape (H, W, C):", image.shape)
print("Image data type      :", image.dtype)

# ---------------------------------------------------------------- Step 3
print("\n" + "=" * 60)
print("STEP 3: INSPECT THE IMAGE")
print("=" * 60)
height, width, channels = image.shape
print("Height   :", height, "pixels")
print("Width    :", width, "pixels")
print("Channels :", channels)

center_row, center_col = height // 2, width // 2
print("Top-left pixel [0, 0] (B, G, R)            :", image[0, 0])
print(f"Center pixel [{center_row}, {center_col}] (B, G, R)          :",
      image[center_row, center_col])

print("Global minimum value :", image.min())
print("Global maximum value :", image.max())
for idx, name in enumerate(["Blue", "Green", "Red"]):
    print(f"{name:5s} channel -> min: {image[:, :, idx].min():3d}, "
          f"max: {image[:, :, idx].max():3d}")

# ---------------------------------------------------------------- Step 4
print("\n" + "=" * 60)
print("STEP 4: COLOR CONVERSION AND CHANNEL SEPARATION")
print("=" * 60)
rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
print("RGB shape  :", rgb.shape)
print("Gray shape :", gray.shape)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
panels = [
    (rgb, "RGB (correct colors)", None),
    (red, "Red channel", "gray"),
    (green, "Green channel", "gray"),
    (blue, "Blue channel", "gray"),
    (gray, "Grayscale", "gray"),
    (image, "Unconverted BGR shown by Matplotlib (wrong colors)", None),
]
for ax, (data, title, cmap) in zip(axes.ravel(), panels):
    ax.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
plt.tight_layout()
plt.savefig("channels_figure.png", dpi=120)
plt.close()
print("Saved figure: channels_figure.png")

r, g, b = (int(v) for v in rgb[center_row, center_col])
hex_value = "#{:02X}{:02X}{:02X}".format(r, g, b)
print(f"Center pixel RGB : ({r}, {g}, {b})")
print("Center pixel HEX :", hex_value)

# ---------------------------------------------------------------- Step 5
print("\n" + "=" * 60)
print("STEP 5: CROP, RESIZE, AND SAVE")
print("=" * 60)
y_start, y_end = 100, 200
x_start, x_end = 250, 350
crop = rgb[y_start:y_end, x_start:x_end]
print(f"Crop region: rows {y_start}:{y_end}, cols {x_start}:{x_end}")
print("Crop shape :", crop.shape)

resized = cv.resize(crop, (200, 200), interpolation=cv.INTER_LINEAR)
print("Resized crop shape:", resized.shape)

original_marked = rgb.copy()
cv.rectangle(original_marked, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
axes[0].imshow(original_marked)
axes[0].set_title(f"Original {width}x{height} (crop region in red)")
axes[1].imshow(crop)
axes[1].set_title("Crop 100x100")
axes[2].imshow(resized)
axes[2].set_title("Resized 200x200")
for ax in axes:
    ax.axis("off")
plt.savefig("crop_resize_figure.png", dpi=120)
plt.close()
print("Saved figure: crop_resize_figure.png")

resized_bgr = cv.cvtColor(resized, cv.COLOR_RGB2BGR)
saved = cv.imwrite("resized_crop.png", resized_bgr)
print("cv.imwrite() saved resized_crop.png successfully:", saved)

# ---------------------------------------------------------------- Step 6
print("\n" + "=" * 60)
print("STEP 6: GRAYSCALE HISTOGRAM")
print("=" * 60)
flat = gray.flatten()
plt.figure(figsize=(9, 5))
plt.hist(flat, bins=256, range=(0, 256), color="gray", edgecolor="none")
plt.title("Grayscale Histogram (256 bins)")
plt.xlabel("Pixel intensity (0 = black, 255 = white)")
plt.ylabel("Number of pixels")
plt.xlim(0, 255)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("grayscale_histogram.png", dpi=120)
plt.close()
print("Saved plot: grayscale_histogram.png")
print("Total pixels      :", flat.size)
print("Mean intensity    :", round(float(flat.mean()), 2))
print("Median intensity  :", float(np.median(flat)))
print("Pixels < 85 (dark)       : {:.1f}%".format(100 * np.mean(flat < 85)))
print("Pixels 85-170 (mid)      : {:.1f}%".format(100 * np.mean((flat >= 85) & (flat <= 170))))
print("Pixels > 170 (bright)    : {:.1f}%".format(100 * np.mean(flat > 170)))

# ---------------------------------------------------------------- Step 7
print("\n" + "=" * 60)
print("STEP 7: BIT DEPTH DEMONSTRATION")
print("=" * 60)
row = np.linspace(0, 255, 256).astype(np.uint8)          # 0..255
gradient_8 = np.tile(row, (100, 1))                      # 100 x 256, 8-bit


def quantize(img, bits):
    """Reduce an 8-bit image to 2**bits gray levels, scaled back to 0-255."""
    levels = 2 ** bits
    step = 256 // levels
    q = (img // step).astype(np.uint8)                   # 0 .. levels-1
    return (q * (255 // (levels - 1))).astype(np.uint8)  # stretch to 0..255


gradient_1 = quantize(gradient_8, 1)
gradient_2 = quantize(gradient_8, 2)
gradient_4 = quantize(gradient_8, 4)

fig, axes = plt.subplots(4, 1, figsize=(9, 7))
for ax, data, title in zip(
    axes,
    [gradient_8, gradient_4, gradient_2, gradient_1],
    ["8-bit (256 levels)", "4-bit (16 levels)", "2-bit (4 levels)", "1-bit (2 levels)"],
):
    ax.imshow(data, cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])  # keep a thin frame so white bands stay visible
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("bit_depth_figure.png", dpi=120)
plt.close()
print("Saved figure: bit_depth_figure.png")

print("Unique values in 8-bit gradient:", len(np.unique(gradient_8)))
print("Unique values in 1-bit gradient:", len(np.unique(gradient_1)))
print("Unique values in 2-bit gradient:", len(np.unique(gradient_2)))
print("Unique values in 4-bit gradient:", len(np.unique(gradient_4)))

print("\n" + "=" * 60)
print("STEP 8: ALL STEPS COMPLETED WITHOUT ERRORS")
print("=" * 60)
