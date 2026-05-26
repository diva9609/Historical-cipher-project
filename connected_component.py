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


CENTER_OF_MASS_DISTANCE = float(
    input("Enter max center-of-mass distance(in pixels) for grouping [Default: 20]\n") or "20")


def centers_of_mass_are_close(first_center, second_center, threshold):
    """
    Return True if the Euclidean distance between two centers of mass
    is less than or equal to the threshold.
    """
    distance = cv2.norm(first_center, second_center, cv2.NORM_L2)

    return distance <= threshold


def component_is_above_below_or_small_inside_box(box, group_area, centroid, component_area):
    """
    Return True if the component should be grouped with the current box.
    """
    x, y, width, height = box
    cx, cy = centroid

    horizontally_inside = x <= cx <= x + width

    above_box = cy < y
    below_box = cy > y + height
    inside_box = y <= cy <= y + height

    component_4_smaller = component_area * 4 <= group_area

    return horizontally_inside and (above_box or below_box or (inside_box and component_4_smaller))


def merge_indices_to_box(indices, stats):
    """Build one merged bounding box from a list of component indices."""
    x_min = min(stats[index][0] for index in indices)
    y_min = min(stats[index][1] for index in indices)

    x_max = max(stats[index][0] + stats[index][2] for index in indices)
    y_max = max(stats[index][1] + stats[index][3] for index in indices)

    width = x_max - x_min
    height = y_max - y_min

    return x_min, y_min, width, height



def group_components(components, stats, centroids, distance_threshold):
    """
    Group components from left to right using center distance and box rules.
    """
    grouped = []
    index = 0

    while index < len(components):
        current_index = components[index]
        current_group = [current_index]

        anchor_centroid = centroids[current_index]

        group_x, group_y, group_width, group_height, group_area = stats[current_index]

        rule1_used = False
        next_position = index + 1

        while next_position < len(components):
            next_index = components[next_position]
            next_centroid = centroids[next_index]
            next_area = stats[next_index][4]

            if (not rule1_used and centers_of_mass_are_close(anchor_centroid, next_centroid, distance_threshold)):
                current_group.append(next_index)
                rule1_used = True

                group_x, group_y, group_width, group_height = merge_indices_to_box(current_group, stats)
                group_area = sum(stats[com_idx][4] for com_idx in current_group)

                next_position += 1
                continue

            if component_is_above_below_or_small_inside_box((group_x, group_y, group_width, group_height), group_area, next_centroid, next_area):
                current_group.append(next_index)

                group_x, group_y, group_width, group_height = merge_indices_to_box(current_group, stats)
                group_area = sum(stats[com_idx][4] for com_idx in current_group)

                next_position += 1
                continue

            break

        grouped.append((current_group, group_x, group_y, group_width, group_height))
        index = next_position

    return grouped

def process_images(input_folder, output_folder, sentence_output_folder, boxes_folder, dataset_name):
    global_counter = 0
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(valid_extensions):
            continue

        image_path = os.path.join(input_folder, filename)

        binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if binary_image is None:
            print(f"Could not read image: {image_path}")
            continue

        boxes_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        sentence_name = os.path.splitext(filename)[0]
        current_sentence_folder = os.path.join(sentence_output_folder, sentence_name)
        os.makedirs(current_sentence_folder, exist_ok=True)

        foreground_image = cv2.bitwise_not(binary_image)

        _, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground_image, connectivity=8)

        components = list(range(1, len(stats)))
        components = sorted(components, key=lambda component_index: centroids[component_index][0])

        merged_symbols = group_components(components, stats, centroids, CENTER_OF_MASS_DISTANCE)

        for indices, x, y, width, height in merged_symbols:
            current_id = global_counter

            merged_mask = np.zeros_like(binary_image, dtype=np.uint8)

            for component_index in indices:
                merged_mask[labels == component_index] = 255

            cropped_mask = merged_mask[y:y + height, x:x + width]
            cropped_original = binary_image[y:y + height, x:x + width]

            cleaned_symbol = cropped_original.copy()
            cleaned_symbol[cropped_mask == 0] = 255

            flat_symbol_filename = os.path.join(output_folder, f"{dataset_name}_{current_id}.jpg")
            cv2.imwrite(flat_symbol_filename, cleaned_symbol)

            sentence_symbol_filename = os.path.join(current_sentence_folder, f"{dataset_name}_{current_id}.jpg")
            cv2.imwrite(sentence_symbol_filename, cleaned_symbol)

            cv2.rectangle(boxes_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

            cv2.putText(boxes_image, str(current_id), (x, max(y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            global_counter += 1

        boxes_filename = os.path.join(boxes_folder, f"boxes_{filename}")
        cv2.imwrite(boxes_filename, boxes_image)

    print("\nConnected Component Segmentation Complete")


def main():
    dataset_name = choose_dataset()

    input_folder = os.path.join("1Preprocessing", "LineCropped", dataset_name)
    output_folder = os.path.join("2Segmentation", dataset_name, "symbols")
    sentence_output_folder = os.path.join("2Segmentation", dataset_name, "symbols_by_sentence")
    boxes_folder = os.path.join("2Segmentation", dataset_name, "bounding_boxes")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(sentence_output_folder, exist_ok=True)
    os.makedirs(boxes_folder, exist_ok=True)

    process_images(input_folder, output_folder, sentence_output_folder, boxes_folder, dataset_name)


if __name__ == "__main__":
    main()