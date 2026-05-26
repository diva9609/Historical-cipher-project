import os

import cv2
import numpy as np


DATASETS = {"1": "Borg", "2": "Copiale", "3": "BNF", "4": "Ramanacoil"}

"""
Lower BACKGROUND_BASELINE:
    Helps faint symbols become more apparent

Higher BACKGROUND_BLUR_SIGMA:
    Smooths uneven illumination.
"""

WINDOW_SIZE = 19
R = 128
MAX_VALUE = 255
THRESHOLD_TYPE = cv2.THRESH_BINARY

BACKGROUND_BLUR_SIGMA = 60
USE_MEDIAN_PREBLUR = True
BACKGROUND_BASELINE = 90

FAINT_CONTRAST_THRESHOLD = 0.35
NOISY_THRESHOLD = 0.04

K_FAINT = 0.32
K_NORMAL_CLEAN = 0.47
K_NORMAL_NOISY = 0.50


def choose_dataset(default="2"):
    prompt = ("What is the name of the folder containing the ciphers? "
              "[Default: 2] (1:Borg 2:Copiale 3:BNF 4:Ramancoil)\n")

    choice = input(prompt).strip() or default

    if choice not in DATASETS:
        raise ValueError(f"Invalid dataset choice: {choice}")

    return DATASETS[choice]


def normalize_background(image,
                         blur_sigma=BACKGROUND_BLUR_SIGMA,
                         use_median_preblur=USE_MEDIAN_PREBLUR,
                         baseline=BACKGROUND_BASELINE):
    
    if use_median_preblur:
        image = cv2.medianBlur(image, 3)

    background = cv2.GaussianBlur(image, (0, 0), blur_sigma)

    corrected = image.astype(np.float32) - background.astype(np.float32) + baseline
    normalized = np.clip(corrected, 0, 255).astype(np.uint8)

    return normalized


def get_image_metrics(image):
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)

    contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0

    median_filtered = cv2.medianBlur(image, 3)
    noise = np.mean(np.abs(image.astype(np.float32) - median_filtered.astype(np.float32))) / 255.0

    return contrast, noise


def choose_sauvola_k(image):
    contrast, noise = get_image_metrics(image)

    if contrast < FAINT_CONTRAST_THRESHOLD:
        k = K_FAINT
        reason = f"faint text: contrast={contrast:.2f}, noise={noise:.3f}"

    elif noise > NOISY_THRESHOLD:
        k = K_NORMAL_NOISY
        reason = f"normal noisy: contrast={contrast:.2f}, noise={noise:.3f}"

    else:
        k = K_NORMAL_CLEAN
        reason = f"normal clean: contrast={contrast:.2f}, noise={noise:.3f}"

    return k, reason


def sauvola_binarize(image_path, output_path, debug_path=None, verbose=True):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    normalized = normalize_background(image)
    k, reason = choose_sauvola_k(normalized)

    if verbose:
        print(f"  {os.path.basename(image_path)}: window={WINDOW_SIZE}, k={k} ({reason})")

    binary = cv2.ximgproc.niBlackThreshold(normalized,
                                           maxValue=MAX_VALUE,
                                           type=THRESHOLD_TYPE,
                                           blockSize=WINDOW_SIZE,
                                           k=k,
                                           binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA,
                                           r=R)

    cv2.imwrite(output_path, binary)

    if debug_path is not None:
        cv2.imwrite(debug_path, normalized)


def process_images(input_folder, output_folder, debug_folder):
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(valid_extensions):
            continue

        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        debug_path = os.path.join(debug_folder, f"normalized_{filename}")

        sauvola_binarize(image_path,
                         output_path,
                         debug_path=debug_path,
                         verbose=True)


def main():
    dataset_name = choose_dataset()

    input_folder = os.path.join(f"{dataset_name}_test", "img")
    output_folder = os.path.join("1Preprocessing", "Binarized", dataset_name)
    debug_folder = os.path.join(output_folder, "debug_normalized")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)


    process_images(input_folder, output_folder, debug_folder)

    print("\nProcessing Complete")


if __name__ == "__main__":
    main()