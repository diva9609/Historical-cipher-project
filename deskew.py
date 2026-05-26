import os

import cv2
import numpy as np




DATASETS = {"1": "Borg", "2": "Copiale", "3": "BNF", "4": "Ramanacoil"}



MAX_ANGLE = float(input("Maximum absolute skew angle to search? [Default: 15]\n") or "15")
COARSE_STEP = float(input("Coarse search step in degrees? [Default: 0.5]\n") or "0.5")
FINE_STEP = float(input("Fine search step in degrees? [Default: 0.1]\n") or "0.1")
MIN_COMPONENT_AREA = int(
    input("Minimum connected-component area to keep for scoring? [Default: 20]\n") or "20")




def choose_dataset(default="2"):
    prompt = ("What is the name of the folder containing the binarized ciphers? "
              "[Default: 2] (1:Borg 2:Copiale 3:BNF 4:Ramanacoil)\n")

    choice = input(prompt).strip() or default

    if choice not in DATASETS:
        raise ValueError(f"Invalid dataset choice: {choice}")

    return DATASETS[choice]




def remove_small_components(binary_foreground, min_area=20):
    """
    Remove tiny connected components from the foreground image.

    This is only used temporarily for deskew scoring.
    It does not remove noise from the final output image.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_foreground,
                                                                    connectivity=8)

    cleaned = np.zeros_like(binary_foreground)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label_id] = 255

    return cleaned


def prepare_for_scoring(binary_foreground, min_area=20):
    cleaned = remove_small_components(binary_foreground, min_area=min_area)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    return cleaned


def rotate_image(image, angle, border_value=0):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated = cv2.warpAffine(image,
                             rotation_matrix,
                             (new_width, new_height),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=border_value)

    return rotated


def projection_score(rotated_foreground):
    projection = np.sum(rotated_foreground > 0, axis=1).astype(np.float32)

    if len(projection) < 2:
        return -1.0

    difference = np.diff(projection)
    score = np.sum(difference * difference)

    return float(score)


def search_best_angle(binary_foreground,
                      max_angle=15.0,
                      coarse_step=0.5,
                      fine_step=0.1):
    best_angle = 0.0
    best_score = -1.0

    coarse_angles = np.arange(-max_angle, max_angle + coarse_step, coarse_step)

    for angle in coarse_angles:
        rotated = rotate_image(binary_foreground, angle, border_value=0)
        score = projection_score(rotated)

        if score > best_score:
            best_score = score
            best_angle = angle

    fine_start = best_angle - coarse_step
    fine_end = best_angle + coarse_step
    fine_angles = np.arange(fine_start, fine_end + fine_step, fine_step)

    for angle in fine_angles:
        rotated = rotate_image(binary_foreground, angle, border_value=0)
        score = projection_score(rotated)

        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def save_debug_overlay(binary_image, angle, debug_path):
    visualization = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    cv2.putText(visualization,
                f"Estimated skew: {angle:.2f} deg",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA)

    cv2.imwrite(debug_path, visualization)



def process_images(input_folder, output_folder, debug_folder):
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(valid_extensions):
            continue

        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        debug_path = os.path.join(debug_folder, f"debug_{filename}")

        binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


        binary_foreground = 255 - binary_image

        scoring_foreground = prepare_for_scoring(binary_foreground, min_area=MIN_COMPONENT_AREA)

        best_angle = search_best_angle(scoring_foreground,
                                       max_angle=MAX_ANGLE,
                                       coarse_step=COARSE_STEP,
                                       fine_step=FINE_STEP)

        corrected = rotate_image(binary_image, best_angle, border_value=255)

        cv2.imwrite(output_path, corrected)
        save_debug_overlay(corrected, best_angle, debug_path)


def main():
    dataset_name = choose_dataset()

    input_folder = os.path.join("1Preprocessing", "Binarized", dataset_name)
    output_folder = os.path.join("1Preprocessing", "Deskewed", dataset_name)
    debug_folder = os.path.join(output_folder, "debug")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)

    process_images(input_folder, output_folder, debug_folder)

    print("\nDeskewing Complete")


if __name__ == "__main__":
    main()