"""
Feature Extractor: Dense-grid SIFT Descriptors with Fisher Vector Aggregation

Also saves sentence-level context features in parallel.

Outputs:
 features.h5
    Plain per-symbol Fisher vectors
    Only images that successfully produce descriptors

 features_trigrams.h5
    Sentence-level context features
    Keeps symbol positions found in sentence folders
    Missing descriptors are replaced with zero vectors
    Stores prev / center / next separately
    Stores concatenated [prev | center | next]
    Stores prev_names / center_names / next_names for clustering refinement
"""

import os
import re

import cv2 as cv
import h5py
import numpy as np
from skimage.feature import fisher_vector, learn_gmm
from tqdm import tqdm


DATASETS = {"1": "Borg", "2": "Copiale", "3": "BNF", "4": "Ramanacoil"}



VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

GRID_STEP_X = 8
GRID_STEP_Y = 6
GMM_MODES = 8


def choose_dataset(default="2"):
    prompt = ("What is the name of the folder containing the ciphers? "
              "[Default: 2] (1:Borg 2:Copiale 3:BNF 4:Ramanacoil)\n")

    choice = input(prompt).strip() or default

    if choice not in DATASETS:
        raise ValueError(f"Invalid dataset choice: {choice}")

    return DATASETS[choice]


def natural_sort_key(filename):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", filename)]


def build_grid_keypoints(image, num_step_x=GRID_STEP_X, num_step_y=GRID_STEP_Y):
    height, width = image.shape

    step_size_x = width / float(num_step_x)
    step_size_y = height / float(num_step_y)

    start_x = step_size_x / 2.0
    start_y = step_size_y / 2.0

    keypoint_size = min(step_size_x / 2.0, step_size_y / 2.0)

    keypoints = []

    for x_index in range(num_step_x):
        x = start_x + step_size_x * x_index

        for y_index in range(num_step_y):
            y = start_y + step_size_y * y_index
            keypoints.append(cv.KeyPoint(x, y, keypoint_size))

    return keypoints


def extract_descriptors(image_path, sift):

    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    keypoints = build_grid_keypoints(image)
    _, descriptors = sift.compute(image, keypoints)


    return descriptors


def get_image_files(folder):

    image_files = [filename for filename in os.listdir(folder)
                   if filename.lower().endswith(VALID_EXTENSIONS)]

    image_files.sort(key=natural_sort_key)

    return image_files


def extract_all_descriptors(input_folder, all_files, sift):
    descriptor_map = {}

    for filename in tqdm(all_files, desc="Extracting descriptors"):
        image_path = os.path.join(input_folder, filename)
        descriptors = extract_descriptors(image_path, sift)


        descriptor_map[filename] = descriptors

    return descriptor_map


def train_gmm(descriptor_map):

    all_descriptors = np.vstack(list(descriptor_map.values()))


    gmm = learn_gmm(all_descriptors, n_modes=GMM_MODES)

    return gmm


def extract_fisher_features(all_files, descriptor_map, gmm):

    image_features = []
    image_names = []
    feature_map = {}

    for filename in tqdm(all_files, desc="Extracting features"):
        descriptors = descriptor_map.get(filename)

        feature = fisher_vector(descriptors, gmm, improved=True).astype(np.float32)

        image_features.append(feature)
        image_names.append(filename)
        feature_map[filename] = feature

    features_array = np.vstack(image_features).astype(np.float32)

    return features_array, image_names, feature_map


def save_features_file(features_file, features_array, image_names, dataset_name):
    print("\nSaving features to HDF5...")

    feature_dim = features_array.shape[1]
    string_dtype = h5py.special_dtype(vlen=str)

    with h5py.File(features_file, "w") as h5_file:
        h5_file.create_dataset("features",
                               data=features_array,
                               compression="gzip",
                               chunks=True,
                               shuffle=True)

        h5_file.create_dataset("image_names",
                               data=np.array(image_names, dtype=object),
                               dtype=string_dtype,
                               compression="gzip")

        h5_file.attrs["total_images"] = len(image_names)
        h5_file.attrs["feature_dim"] = feature_dim
        h5_file.attrs["dataset"] = dataset_name
        h5_file.attrs["binarization"] = "sauvola"
        h5_file.attrs["aggregation"] = "fisher_vector"
        h5_file.attrs["feature_type"] = "SIFT"
        h5_file.attrs["keypoint_strategy"] = "dense_grid"
        h5_file.attrs["grid_step_x"] = GRID_STEP_X
        h5_file.attrs["grid_step_y"] = GRID_STEP_Y
        h5_file.attrs["gmm_modes"] = GMM_MODES


def get_sentence_folders(sentence_input_folder):

    sentence_folders = [folder for folder in os.listdir(sentence_input_folder)
                        if os.path.isdir(os.path.join(sentence_input_folder, folder))]

    sentence_folders.sort(key=natural_sort_key)

    return sentence_folders


def build_folder_to_images(sentence_input_folder, sentence_folders):

    folder_to_images = {}

    for sentence_folder in sentence_folders:
        sentence_path = os.path.join(sentence_input_folder, sentence_folder)
        folder_to_images[sentence_folder] = get_image_files(sentence_path)

    return folder_to_images


def get_feature_or_zero(filename, feature_map, zero_vector):
    """Return a feature if it exists, otherwise return a zero vector."""
    if filename and filename in feature_map:
        return feature_map[filename], 1

    return zero_vector, 0


def build_trigram_features(sentence_folders, folder_to_images, feature_map, zero_vector):
    """Build sentence-level trigram context features."""
    center_features = []
    prev_features = []
    next_features = []
    context_features = []

    context_image_names = []
    context_sentence_folders = []

    center_names = []
    prev_names = []
    next_names = []

    has_center = []
    has_prev = []
    has_next = []

    for sentence_folder in sentence_folders:
        files = folder_to_images[sentence_folder]

        for index, filename in enumerate(files):
            center_feature, center_flag = get_feature_or_zero(filename,
                                                              feature_map,
                                                              zero_vector)

            if index == 0:
                prev_name = ""
            else:
                prev_name = files[index - 1]

            if index == len(files) - 1:
                next_name = ""
            else:
                next_name = files[index + 1]

            prev_feature, prev_flag = get_feature_or_zero(prev_name,
                                                          feature_map,
                                                          zero_vector)

            next_feature, next_flag = get_feature_or_zero(next_name,
                                                          feature_map,
                                                          zero_vector)

            concatenated_feature = np.hstack([prev_feature, center_feature, next_feature]).astype(np.float32)

            prev_features.append(prev_feature)
            center_features.append(center_feature)
            next_features.append(next_feature)
            context_features.append(concatenated_feature)

            context_image_names.append(filename)
            context_sentence_folders.append(sentence_folder)

            center_names.append(filename)
            prev_names.append(prev_name)
            next_names.append(next_name)

            has_center.append(center_flag)
            has_prev.append(prev_flag)
            has_next.append(next_flag)


    return {"center_features": np.vstack(center_features).astype(np.float32),
            "prev_features": np.vstack(prev_features).astype(np.float32),
            "next_features": np.vstack(next_features).astype(np.float32),
            "trigram_features": np.vstack(context_features).astype(np.float32),
            "image_names": context_image_names,
            "sentence_folders": context_sentence_folders,
            "center_names": center_names,
            "prev_names": prev_names,
            "next_names": next_names,
            "has_center": has_center,
            "has_prev": has_prev,
            "has_next": has_next}


def save_trigram_features_file(trigrams_file, trigram_data, dataset_name, feature_dim):
    """Save sentence-level trigram context features to HDF5."""
    print(f"Built context features for {trigram_data['center_features'].shape[0]} symbol positions")

    string_dtype = h5py.special_dtype(vlen=str)

    with h5py.File(trigrams_file, "w") as h5_file:
        h5_file.create_dataset("center_features",
                               data=trigram_data["center_features"],
                               compression="gzip",
                               chunks=True,
                               shuffle=True)

        h5_file.create_dataset("prev_features",
                               data=trigram_data["prev_features"],
                               compression="gzip",
                               chunks=True,
                               shuffle=True)

        h5_file.create_dataset("next_features",
                               data=trigram_data["next_features"],
                               compression="gzip",
                               chunks=True,
                               shuffle=True)

        h5_file.create_dataset("trigram_features",
                               data=trigram_data["trigram_features"],
                               compression="gzip",
                               chunks=True,
                               shuffle=True)

        h5_file.create_dataset("image_names",
                               data=np.array(trigram_data["image_names"], dtype=object),
                               dtype=string_dtype,
                               compression="gzip",)

        h5_file.create_dataset("sentence_folders",
                               data=np.array(trigram_data["sentence_folders"], dtype=object),
                               dtype=string_dtype,
                               compression="gzip")

        h5_file.create_dataset("center_names",
                               data=np.array(trigram_data["center_names"], dtype=object),
                               dtype=string_dtype,
                               compression="gzip")

        h5_file.create_dataset("prev_names",
                               data=np.array(trigram_data["prev_names"], dtype=object),
                               dtype=string_dtype,
                               compression="gzip")

        h5_file.create_dataset("next_names",
                               data=np.array(trigram_data["next_names"], dtype=object),
                               dtype=string_dtype,
                               compression="gzip")

        h5_file.create_dataset("has_center", data=np.array(trigram_data["has_center"], dtype=np.uint8))

        h5_file.create_dataset("has_prev", data=np.array(trigram_data["has_prev"], dtype=np.uint8))

        h5_file.create_dataset("has_next", data=np.array(trigram_data["has_next"], dtype=np.uint8))

        h5_file.attrs["total_images"] = trigram_data["center_features"].shape[0]
        h5_file.attrs["feature_dim"] = feature_dim
        h5_file.attrs["trigram_feature_dim"] = trigram_data["trigram_features"].shape[1]
        h5_file.attrs["context_type"] = "sentence_trigram"
        h5_file.attrs["blank_representation"] = "zero_vector"
        h5_file.attrs["dataset"] = dataset_name
        h5_file.attrs["binarization"] = "sauvola"
        h5_file.attrs["aggregation"] = "fisher_vector"
        h5_file.attrs["feature_type"] = "SIFT"
        h5_file.attrs["keypoint_strategy"] = "dense_grid"
        h5_file.attrs["grid_step_x"] = GRID_STEP_X
        h5_file.attrs["grid_step_y"] = GRID_STEP_Y
        h5_file.attrs["gmm_modes"] = GMM_MODES
        h5_file.attrs["input_structure"] = "sentence_folders"
        h5_file.attrs["stores_neighbor_names"] = True
        h5_file.attrs["name_key"] = "filename_with_extension"

    print(f" Saved {trigram_data['center_features'].shape[0]} context entries")


def build_and_save_trigram_features(sentence_input_folder,
                                    trigrams_file,
                                    feature_map,
                                    zero_vector,
                                    dataset_name,
                                    feature_dim):
    
    sentence_folders = get_sentence_folders(sentence_input_folder)


    folder_to_images = build_folder_to_images(sentence_input_folder, sentence_folders)

    trigram_data = build_trigram_features(sentence_folders, folder_to_images, feature_map, zero_vector)


    save_trigram_features_file(trigrams_file, trigram_data, dataset_name, feature_dim)


def main():
    dataset_name = choose_dataset()

    input_folder = os.path.join("2Segmentation", dataset_name, "symbols")

    sentence_input_folder = os.path.join("2Segmentation", dataset_name, "symbols_by_sentence")

    output_dir = os.path.join("features", "ciphers", dataset_name)

    os.makedirs(output_dir, exist_ok=True)

    features_file = os.path.join(output_dir, "features.h5")
    trigrams_file = os.path.join(output_dir, "features_trigrams.h5")


    sift = cv.SIFT_create()


    all_files = get_image_files(input_folder)
    total_images = len(all_files)



    for index in range(min(5, total_images)):
        print(f"  {index}: {all_files[index]}")

    descriptor_map = extract_all_descriptors(input_folder, all_files, sift)

    gmm = train_gmm(descriptor_map)

    features_array, image_names, feature_map = extract_fisher_features(all_files, descriptor_map, gmm)

    feature_dim = features_array.shape[1]
    zero_vector = np.zeros(feature_dim, dtype=np.float32)

    save_features_file(features_file, features_array, image_names, dataset_name)


    build_and_save_trigram_features(sentence_input_folder,
                                    trigrams_file,
                                    feature_map,
                                    zero_vector,
                                    dataset_name,
                                    feature_dim)

    print(f"Dataset: {dataset_name}")
    print(f"Images with features: {len(image_names)}")
    print(f"Features shape: {features_array.shape}")
    print(f"Features file: {features_file}")


    for index, name in enumerate(image_names[:5]):
        print(f"  {index}: {name}")


if __name__ == "__main__":
    main()