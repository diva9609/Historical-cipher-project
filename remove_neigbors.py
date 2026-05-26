import os

import cv2
import numpy as np


DATASETS = {"1": "Borg", "2": "Copiale", "3": "BNF", "4": "Ramanacoil"}



def choose_dataset(default="2"):

    prompt = ("What is the name of the folder containing the ciphers? "
              "[Default: 2] (1:Borg 2:Copiale 3:BNF 4:Ramanacoil)\n")

    choice = input(prompt).strip() or default

    if choice not in DATASETS:
        raise ValueError(f"Invalid dataset choice: {choice}")

    return DATASETS[choice]

print("\nMain line settings")

MIN_COMPONENT_AREA = int(input("Minimum connected-component area to keep? [Default: 20]\n") or "20")

ADAPTIVE_TOP_MARGIN = int(input("Extra margin above adaptive top? [Default: 3]\n") or "3")

FIXED_BOTTOM_EXTENSION = int(input("Fixed pixels to add below selected band? [Default: 20]\n") or "20")

print("\nNoise cleanup settings")

MIDDLE_ZONE_RATIO = float(input("Middle zone ratio of cropped height (0-1)? [Default: 0.50]\n") or "0.50")

EDGE_TOUCH_TOLERANCE = int(input("Edge touch tolerance in pixels? [Default: 2]\n") or "2")


def remove_small_components(binary_fg, min_area=20):
    """
    Remove small connected components from the foreground mask.

    This is used for detection, not directly for saving.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_fg, connectivity=8)

    cleaned = np.zeros_like(binary_fg)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def smooth_projection(projection):
    kernel_size = 7
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size

    return np.convolve(projection, kernel, mode="same")


def find_line_bands(binary_fg):
    row_sum = np.sum(binary_fg > 0, axis=1).astype(np.float32)
    smoothed = smooth_projection(row_sum)

    max_value = float(np.max(smoothed))

    if max_value <= 0:
        return []

    threshold = 0.30 * max_value
    active = smoothed > threshold

    bands = []
    in_band = False
    start = 0

    for y, value in enumerate(active):
        if value and not in_band:
            start = y
            in_band = True

        elif not value and in_band:
            end = y - 1
            score = float(np.sum(row_sum[start:end + 1]))
            bands.append((start, end, score))
            in_band = False

    if in_band:
        end = len(active) - 1
        score = float(np.sum(row_sum[start:end + 1]))
        bands.append((start, end, score))

    return bands


def filter_bands_by_height(bands):
    min_height = 10
    max_height = 120
    filtered = []

    for y1, y2, score in bands:
        height = y2 - y1 + 1

        if min_height <= height <= max_height:
            filtered.append((y1, y2, score))

    return filtered


def choose_main_band(bands, image_height):

    center_y = image_height / 2.0

    selected_index = min(range(len(bands)),
                         key=lambda i: abs(((bands[i][0] + bands[i][1]) / 2.0) - center_y))

    return bands[selected_index]


def crop_to_main_line_adaptive_top_only(binary_image,
                                        binary_fg,
                                        band,
                                        top_margin=3,
                                        bottom_extension=20,
                                        min_component_area=20):

    overlap_ratio = 0.30
    height, width = binary_image.shape[:2]
    y1, y2, _ = band

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_fg, connectivity=8)

    selected_tops = []

    band_top = y1
    band_bottom = y2
    band_height = band_bottom - band_top + 1

    for i in range(1, num_labels):
        y = stats[i, cv2.CC_STAT_TOP]
        component_height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_component_area:
            continue

        component_top = y
        component_bottom = y + component_height - 1

        overlap_top = max(component_top, band_top)
        overlap_bottom = min(component_bottom, band_bottom)
        overlap_height = overlap_bottom - overlap_top + 1

        if overlap_height <= 0:
            continue

        component_overlap_ratio = overlap_height / max(component_height, 1)
        band_overlap_ratio = overlap_height / max(band_height, 1)

        if (component_overlap_ratio >= overlap_ratio
            or band_overlap_ratio >= overlap_ratio):
            selected_tops.append(component_top)


    top = max(0, min(selected_tops) - top_margin)
    bottom = min(height, y2 + bottom_extension + 1)

    left = 0
    right = width

    cropped = binary_image[top:bottom, left:right]

    return cropped, top, bottom, left, right


def remove_edge_components_not_reaching_middle(binary_crop,
                                               foreground_crop,
                                               middle_ratio=0.50,
                                               edge_tolerance=2):

    height, width = foreground_crop.shape[:2]

    if height == 0 or width == 0:
        return binary_crop

    middle_ratio = max(0.0, min(1.0, middle_ratio))

    zone_height = int(round(height * middle_ratio))
    zone_height = max(1, min(zone_height, height))

    middle_top = (height - zone_height) // 2
    middle_bottom = middle_top + zone_height - 1

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_crop, connectivity=8,)

    keep_mask = np.zeros_like(foreground_crop)

    for i in range(1, num_labels):
        component_top = stats[i, cv2.CC_STAT_TOP]
        component_height = stats[i, cv2.CC_STAT_HEIGHT]
        component_bottom = component_top + component_height - 1

        touches_top = component_top <= edge_tolerance
        touches_bottom = component_bottom >= height - 1 - edge_tolerance

        if not (touches_top or touches_bottom):
            keep_mask[labels == i] = 255
            continue

        overlaps_middle = not (component_bottom < middle_top
                               or component_top > middle_bottom)

        if overlaps_middle:
            keep_mask[labels == i] = 255

    filtered_binary = binary_crop.copy()
    filtered_binary[keep_mask == 0] = 255

    return filtered_binary


def save_debug_image(binary_image, selected_band, top, bottom, left, right, debug_path):
    visualization = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    if selected_band is not None:
        y1, y2, score = selected_band

        cv2.rectangle(visualization,
                      (0, y1),
                      (binary_image.shape[1] - 1, y2),
                      (0, 0, 255),
                      2)

        cv2.putText(visualization,
                    f"selected y=({y1},{y2}) h={y2 - y1 + 1} score={score:.0f}",
                    (10, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA)

    if None not in (left, top, right, bottom):
        cv2.rectangle(visualization,
                      (left, top),
                      (right - 1, bottom - 1),
                      (0, 255, 0),
                      2)

    cv2.imwrite(debug_path, visualization)


def process_images(input_folder, output_folder, debug_folder):

    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(valid_extensions):
            continue

        image_path = os.path.join(input_folder, filename)

        binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        binary_fg = 255 - binary_image

        cleaned_fg = remove_small_components(binary_fg, min_area=MIN_COMPONENT_AREA)

        all_bands = find_line_bands(cleaned_fg)
        valid_bands = filter_bands_by_height(all_bands)

        selected_band = choose_main_band(valid_bands, binary_image.shape[0])


        cropped, top, bottom, left, right = crop_to_main_line_adaptive_top_only(binary_image,
                                                                                cleaned_fg,
                                                                                selected_band,
                                                                                top_margin=ADAPTIVE_TOP_MARGIN,
                                                                                bottom_extension=FIXED_BOTTOM_EXTENSION,
                                                                                min_component_area=MIN_COMPONENT_AREA)


        cropped_fg = cleaned_fg[top:bottom, left:right]

        cropped = remove_edge_components_not_reaching_middle(cropped,
                                                             cropped_fg,
                                                             middle_ratio=MIDDLE_ZONE_RATIO,
                                                             edge_tolerance=EDGE_TOUCH_TOLERANCE)

        output_path = os.path.join(output_folder, filename)
        debug_path = os.path.join(debug_folder, f"debug_{filename}")

        cv2.imwrite(output_path, cropped)

        save_debug_image(binary_image,
                         selected_band,
                         top,
                         bottom,
                         left,
                         right,
                         debug_path)

def main():
    dataset_name = choose_dataset()

    input_folder = os.path.join("1Preprocessing", "Deskewed", dataset_name)
    output_folder = os.path.join("1Preprocessing", "LineCropped", dataset_name)
    debug_folder = os.path.join(output_folder, "debug")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)

    process_images(input_folder, output_folder, debug_folder)

    print("\nDone.")


if __name__ == "__main__":
    main()