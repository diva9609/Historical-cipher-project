import os
import re

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm


DATASETS = {"1": "Borg", "2": "Copiale", "3": "BNF", "4": "Ramanacoil"}

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

ENABLE_CONTEXT_REFINEMENT = True

USE_THRESHOLD_SEARCH = True
DEFAULT_THRESHOLD = 1.071

MIN_CLUSTER_SIZE = 5
MIN_DESTINATION_CLUSTER_SIZE = 2

REFINE_ONLY_ABOVE_PERCENTILE = 0
CONTEXT_MATCH_THRESHOLD = 0.20

MAX_SUSPECTS_PER_CLUSTER = 1
MAX_REFINEMENT_PASSES = 15

START_TOKEN = "__START__"
END_TOKEN = "__END__"


def choose_dataset(default="2"):
    """Ask the user which dataset to process."""
    prompt = ("Dataset [Default: 2] "
              "(1:Borg 2:Copiale 3:BNF 4:Ramanacoil)\n")

    choice = input(prompt).strip() or default

    if choice not in DATASETS:
        raise ValueError(f"Invalid dataset choice: {choice}")

    return DATASETS[choice]

def clean_name(name):

    if isinstance(name, bytes):
        name = name.decode("utf-8")

    name = str(name)

    if name == "":
        return ""

    return os.path.splitext(name)[0]


def natural_sort_key(text):

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def get_image_files(folder):
    image_files = [filename for filename in os.listdir(folder) if filename.lower().endswith(VALID_EXTENSIONS)]

    image_files.sort(key=natural_sort_key)

    return image_files


def get_context_cluster_ids(base_name, trigram_context, filename_to_cluster):

    if base_name not in trigram_context:
        return []

    context_cluster_ids = []

    prev_name = trigram_context[base_name]["prev"]
    next_name = trigram_context[base_name]["next"]

    if prev_name is None:
        context_cluster_ids.append(START_TOKEN)
    elif prev_name in filename_to_cluster:
        context_cluster_ids.append(filename_to_cluster[prev_name])

    if next_name is None:
        context_cluster_ids.append(END_TOKEN)
    elif next_name in filename_to_cluster:
        context_cluster_ids.append(filename_to_cluster[next_name])

    return context_cluster_ids


def remove_one_occurrence_each(target_list, values_to_remove):

    for value in values_to_remove:
        if value in target_list:
            target_list.remove(value)


def context_match_score(suspect_context_clusters, cluster_context_clusters):

    if len(suspect_context_clusters) == 0 or len(cluster_context_clusters) == 0:
        return 0, 0.0

    suspect_context_set = set(suspect_context_clusters)

    matches = sum(1 for cluster_id in cluster_context_clusters if cluster_id in suspect_context_set)

    ratio = matches / len(cluster_context_clusters)

    return matches, ratio


def compute_cluster_scores(features, effective_labels, n_clusters, min_cluster_size):

    cluster_scores = {}
    valid_scores = {}

    for label in range(n_clusters):
        indices = np.where(effective_labels == label)[0]
        size = len(indices)

        if size < min_cluster_size:
            cluster_scores[label] = 999.0
            continue

        cluster_features = features[indices]
        centroid = np.mean(cluster_features, axis=0)

        average_distance = np.mean(np.linalg.norm(cluster_features - centroid, axis=1))

        cluster_scores[label] = average_distance
        valid_scores[label] = average_distance

    return cluster_scores, valid_scores


def get_accepted_candidates(features, effective_labels, n_clusters, cluster_scores, min_cluster_size, var_threshold):
    
    accepted_candidates = []

    for label in range(n_clusters):
        indices = np.where(effective_labels == label)[0]
        size = len(indices)

        if size < min_cluster_size:
            continue

        average_distance = cluster_scores[label]

        if average_distance <= var_threshold:
            accepted_candidates.append((label, size, average_distance))

    return accepted_candidates


def load_plain_features(h5_path):

    with h5py.File(h5_path, "r") as h5_file:
        all_features = h5_file["features"][:]
        all_image_names = h5_file["image_names"][:]

    all_image_names_clean = [clean_name(name) for name in all_image_names]

    return all_features, all_image_names_clean


def load_trigram_context(trigram_h5_path, enable_context_refinement):
    trigram_context = {}

    if not enable_context_refinement:
        return trigram_context, False

    with h5py.File(trigram_h5_path, "r") as h5_file:
        center_names = [clean_name(name) for name in h5_file["center_names"][:]]
        prev_names_raw = [clean_name(name) for name in h5_file["prev_names"][:]]
        next_names_raw = [clean_name(name) for name in h5_file["next_names"][:]]

        has_prev = h5_file["has_prev"][:]
        has_next = h5_file["has_next"][:]

    for index, center_name in enumerate(center_names):
        if has_prev[index] and prev_names_raw[index] != "":
            prev_name = prev_names_raw[index]
        else:
            prev_name = None

        if has_next[index] and next_names_raw[index] != "":
            next_name = next_names_raw[index]
        else:
            next_name = None

        trigram_context[center_name] = {"prev": prev_name, "next": next_name,}

    print(f"Loaded context for {len(trigram_context)} symbol positions")

    return trigram_context, True


def match_images_to_features(image_folder, all_features, all_image_names_clean):

    print("Syncing images with features...")

    all_images = get_image_files(image_folder)

    name_to_index = {name: index for index, name in enumerate(all_image_names_clean)}

    filenames = []
    features = []
    images = []

    for image_name in tqdm(all_images):
        base_name = os.path.splitext(image_name)[0]

        feature_index = name_to_index[base_name]
        feature = all_features[feature_index]

        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        features.append(feature)
        images.append(image)
        filenames.append(image_name)

    features = np.array(features)


    features = normalize(features, norm="l2")

    print(f"Matched {len(features)} images to feature vectors")

    return filenames, features, images



def find_best_threshold(features, results_folder):

    print("\nSearching for optimal threshold...")

    test_thresholds = np.linspace(0.5, 1.5, 15)

    best_score = -1
    best_threshold = DEFAULT_THRESHOLD
    scores = []

    for threshold in test_thresholds:
        cluster_test = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage="average", metric="euclidean")

        labels_test = cluster_test.fit_predict(features)
        clusters_found = len(set(labels_test))

        if 1 < clusters_found < len(features):
            score = silhouette_score(features, labels_test)
            scores.append(score)

            if score > best_score:
                best_score = score
                best_threshold = threshold

            print(f"Threshold: {threshold:.3f} | " f"Clusters: {clusters_found} | " f"Silhouette: {score:.4f}")
        else:
            scores.append(0)

    print(f"\nBest threshold: {best_threshold:.3f}")

    plt.figure()
    plt.plot(test_thresholds, scores, marker="o")
    plt.title("Threshold Optimization")
    plt.xlabel("Distance Threshold")
    plt.ylabel("Silhouette Score")
    plt.savefig(os.path.join(results_folder, "optimization_curve.png"))
    plt.close()

    return best_threshold


def run_final_clustering(features, best_threshold):
    
    print("\nRunning final clustering...")

    final_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=best_threshold, linkage="average", metric="euclidean")

    labels = final_clustering.fit_predict(features)
    effective_labels = labels.copy()

    n_clusters = len(set(labels))

    print(f"Final clusters found: {n_clusters}")

    return labels, effective_labels, n_clusters


def build_filename_to_cluster(filenames, effective_labels):
    return {os.path.splitext(filenames[index])[0]: int(effective_labels[index])
            for index in range(len(filenames))}


def build_cluster_context_dict(filenames, effective_labels, n_clusters, trigram_context, filename_to_cluster, enable_context_refinement):
    cluster_context_dict = {label: [] for label in range(n_clusters)}

    if enable_context_refinement and trigram_context:
        print("\nBuilding cluster context dictionary...")

        for index in range(len(filenames)):
            label = int(effective_labels[index])
            base_name = os.path.splitext(filenames[index])[0]

            context_cluster_ids = get_context_cluster_ids(base_name, trigram_context, filename_to_cluster)

            cluster_context_dict[label].extend(context_cluster_ids)

        print(f"Built context dictionary for {len(cluster_context_dict)} clusters")

    return cluster_context_dict



def run_iterative_context_refinement(features, filenames, effective_labels, n_clusters, cluster_context_dict, filename_to_cluster, trigram_context, enable_context_refinement):
    reassigned_by_context = {}
    total_reassignment_events = 0
    refinement_passes_completed = 0

    if not (enable_context_refinement and trigram_context):
        print("\nContext refinement skipped.")
        return reassigned_by_context, total_reassignment_events, refinement_passes_completed

    print("\nRunning iterative context-based reassignment refinement...")

    refinement_pass = 0

    while refinement_pass < MAX_REFINEMENT_PASSES:
        refinement_pass += 1
        refinement_passes_completed = refinement_pass
        moves_this_pass = 0

        print(f"\nContext refinement pass {refinement_pass}...")

        cluster_scores, valid_scores = compute_cluster_scores(features, effective_labels, n_clusters, MIN_CLUSTER_SIZE)

        if valid_scores:
            refine_score_threshold = np.percentile(list(valid_scores.values()), REFINE_ONLY_ABOVE_PERCENTILE)
        else:
            refine_score_threshold = 0

        for label in range(n_clusters):
            indices = np.where(effective_labels == label)[0]
            size = len(indices)

            if size < MIN_CLUSTER_SIZE:
                continue

            cluster_score = cluster_scores[label]

            if cluster_score <= refine_score_threshold:
                continue

            cluster_features = features[indices]

            pairwise_distance = cdist(cluster_features, cluster_features, metric="euclidean")

            np.fill_diagonal(pairwise_distance, np.nan)

            average_pairwise_distance = np.nanmean(pairwise_distance, axis=1)

            suspect_order = np.argsort(average_pairwise_distance)[::-1]
            suspect_order = suspect_order[:MAX_SUSPECTS_PER_CLUSTER]

            for suspect_local_index in suspect_order:
                suspect_global_index = indices[suspect_local_index]

                suspect_base = os.path.splitext(filenames[suspect_global_index])[0]
                current_label = int(effective_labels[suspect_global_index])

                suspect_context_clusters = get_context_cluster_ids(suspect_base, trigram_context, filename_to_cluster)

                if len(suspect_context_clusters) == 0:
                    continue

                current_cluster_context_without_suspect = list(cluster_context_dict[current_label])

                remove_one_occurrence_each(current_cluster_context_without_suspect, suspect_context_clusters,)

                current_matches, current_match_ratio = context_match_score(suspect_context_clusters, current_cluster_context_without_suspect)

                if current_match_ratio >= CONTEXT_MATCH_THRESHOLD:
                    continue

                best_label = None
                best_matches = 0
                best_match_ratio = 0.0

                for candidate_label, candidate_contexts in cluster_context_dict.items():
                    if candidate_label == current_label:
                        continue

                    if len(candidate_contexts) == 0:
                        continue

                    candidate_size = np.sum(effective_labels == candidate_label)

                    if candidate_size < MIN_DESTINATION_CLUSTER_SIZE:
                        continue

                    candidate_matches, candidate_match_ratio = context_match_score(suspect_context_clusters, candidate_contexts)

                    if (best_label is None or candidate_match_ratio > best_match_ratio
                        or (candidate_match_ratio == best_match_ratio and candidate_matches > best_matches)):
                        best_label = candidate_label
                        best_matches = candidate_matches
                        best_match_ratio = candidate_match_ratio

                if (best_label is not None and best_matches > 0 and best_match_ratio > current_match_ratio):
                    old_label = current_label

                    reassigned_by_context[suspect_global_index] = {"from": old_label, "to": best_label, "pass": refinement_pass}

                    effective_labels[suspect_global_index] = best_label
                    filename_to_cluster[suspect_base] = best_label

                    remove_one_occurrence_each(cluster_context_dict[old_label], suspect_context_clusters)

                    cluster_context_dict[best_label].extend(suspect_context_clusters)

                    total_reassignment_events += 1
                    moves_this_pass += 1

                    print(f"Pass {refinement_pass}: moved suspect "
                          f"{filenames[suspect_global_index]} "
                          f"from cluster {old_label} to cluster {best_label} | "
                          f"current ratio={current_match_ratio:.3f}, "
                          f"best ratio={best_match_ratio:.3f}, "
                          f"best matches={best_matches}")

        print(f"Pass {refinement_pass} reassigned {moves_this_pass} symbols")

        if moves_this_pass == 0:
            break

    print(f"Context refinement finished after {refinement_passes_completed} passes. "
          f"Total unique symbols reassigned: {len(reassigned_by_context)}. "
          f"Total reassignment events: {total_reassignment_events}.")

    return reassigned_by_context, total_reassignment_events, refinement_passes_completed


def choose_final_accepted_labels(accepted_candidates):
    accepted_candidates_sorted = sorted(accepted_candidates, key=lambda item: item[1], reverse=True)

    selected = accepted_candidates_sorted[:150]
    final_accepted_labels = {label for label, _, _ in selected}

    print(f"Keeping top {len(final_accepted_labels)} clusters")

    return final_accepted_labels


def get_cluster_representatives(features, effective_labels, n_clusters):
    cluster_representatives = {}

    for label in range(n_clusters):
        indices = np.where(effective_labels == label)[0]

        if len(indices) == 0:
            continue

        cluster_features = features[indices]
        centroid = np.mean(cluster_features, axis=0)

        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        closest_local_index = np.argmin(distances)
        closest_global_index = indices[closest_local_index]

        cluster_representatives[label] = closest_global_index

    return cluster_representatives


def save_clusters(results_folder,
                  filenames,
                  images,
                  effective_labels,
                  n_clusters,
                  cluster_scores,
                  var_threshold,
                  final_accepted_labels,
                  cluster_representatives):

    for label in range(n_clusters):
        indices = np.where(effective_labels == label)[0]

        size = len(indices)
        score = cluster_scores.get(label, 999.0)

        if size < MIN_CLUSTER_SIZE:
            cluster_path = os.path.join(results_folder, "Rejected_TooSmall", f"Cluster_{label}")

        elif score > var_threshold:
            cluster_path = os.path.join(results_folder, "Rejected_HighVar", f"Cluster_{label}_Size{size}")

        else:
            if label in final_accepted_labels:
                cluster_path = os.path.join(results_folder, "Accepted", f"Cluster_{label}_Size{size}")
            else:
                cluster_path = os.path.join(results_folder, "Rejected_NotTop150", f"Cluster_{label}_Size{size}")

        os.makedirs(cluster_path, exist_ok=True)

        representative_index = cluster_representatives.get(label, None)

        for index in indices:
            output_name = filenames[index]

            if representative_index is not None and index == representative_index:
                if not output_name.startswith("REP_"):
                    output_name = "REP_" + output_name

            cv2.imwrite(os.path.join(cluster_path, output_name), images[index])



def save_summary(results_folder,
                 dataset_name,
                 trigram_h5_path,
                 features,
                 n_clusters,
                 var_threshold,
                 best_threshold,
                 accepted_candidates,
                 final_accepted_labels,
                 trigram_context,
                 reassigned_by_context,
                 total_reassignment_events,
                 refinement_passes_completed,
                 enable_context_refinement):
    
    summary_path = os.path.join(results_folder, "clustering_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(f"Dataset: {dataset_name}\n")
        file.write("Binarization: sauvola\n")
        file.write(f"Total images: {len(features)}\n")
        file.write(f"Clusters found: {n_clusters}\n")
        file.write(f"Min cluster size: {MIN_CLUSTER_SIZE}\n")
        file.write(f"Variance threshold: {var_threshold:.6f}\n")
        file.write(f"Best threshold: {best_threshold:.3f}\n")
        file.write(f"Accepted candidates after refinement: {len(accepted_candidates)}\n")
        file.write(f"Final accepted: {len(final_accepted_labels)}\n")
        file.write("Rule: Top 150 clusters\n")
        file.write(f"Threshold search used: {USE_THRESHOLD_SEARCH}\n")

        file.write("\nContext refinement:\n")
        file.write(f"Enabled: {enable_context_refinement}\n")
        file.write(f"trigram file: {trigram_h5_path}\n")
        file.write(f"Context entries loaded: {len(trigram_context)}\n")
        file.write(f"Unique symbols reassigned: {len(reassigned_by_context)}\n")
        file.write(f"Total reassignment events: {total_reassignment_events}\n")
        file.write(f"Refinement passes completed: {refinement_passes_completed}\n")
        file.write(f"Context match threshold: {CONTEXT_MATCH_THRESHOLD}\n")
        file.write(f"Refine above percentile: {REFINE_ONLY_ABOVE_PERCENTILE}\n")
        file.write(f"Max suspects per cluster: {MAX_SUSPECTS_PER_CLUSTER}\n")
        file.write(f"Min destination cluster size: {MIN_DESTINATION_CLUSTER_SIZE}\n")
        file.write(f"Max refinement passes: {MAX_REFINEMENT_PASSES}\n")
        file.write("Move rule: best candidate ratio must be greater than " "current cluster ratio\n")

    return summary_path


def main():
    dataset_name = choose_dataset()

    image_folder = os.path.join("2Segmentation", dataset_name,"symbols")

    results_folder = os.path.join("3Clustering", "Hierarchical", dataset_name)

    h5_path = os.path.join("features", "ciphers", dataset_name, "features.h5")

    trigram_h5_path = os.path.join("features", "ciphers", dataset_name, "features_trigrams.h5")

    os.makedirs(results_folder, exist_ok=True)

    all_features, all_image_names_clean = load_plain_features(h5_path)

    trigram_context, enable_context_refinement = load_trigram_context(trigram_h5_path, ENABLE_CONTEXT_REFINEMENT)

    filenames, features, images = match_images_to_features(image_folder, all_features, all_image_names_clean)

    if USE_THRESHOLD_SEARCH:
        best_threshold = find_best_threshold(features, results_folder)
    else:
        best_threshold = DEFAULT_THRESHOLD
        print(f"\nUsing default threshold: {best_threshold:.3f}")

    _, effective_labels, n_clusters = run_final_clustering(features, best_threshold)

    filename_to_cluster = build_filename_to_cluster(filenames, effective_labels)

    cluster_context_dict = build_cluster_context_dict(filenames, effective_labels, n_clusters, trigram_context, filename_to_cluster, enable_context_refinement)

    reassigned_by_context, total_reassignment_events, refinement_passes_completed = (run_iterative_context_refinement(features,
                                                                                                                   filenames,
                                                                                                                   effective_labels,
                                                                                                                   n_clusters,
                                                                                                                   cluster_context_dict,
                                                                                                                   filename_to_cluster,
                                                                                                                   trigram_context,
                                                                                                                   enable_context_refinement))

    cluster_scores, valid_scores_after_refinement = compute_cluster_scores(features,
                                                                           effective_labels,
                                                                           n_clusters,
                                                                           MIN_CLUSTER_SIZE)

    if valid_scores_after_refinement:
        var_threshold = np.percentile(list(valid_scores_after_refinement.values()), 90)
    else:
        var_threshold = 0

    accepted_candidates = get_accepted_candidates(features,
                                                  effective_labels,
                                                  n_clusters,
                                                  cluster_scores,
                                                  MIN_CLUSTER_SIZE,
                                                  var_threshold)

    final_accepted_labels = choose_final_accepted_labels(accepted_candidates)

    cluster_representatives = get_cluster_representatives(features, effective_labels, n_clusters)

    save_clusters(results_folder,
                  filenames,
                  images,
                  effective_labels,
                  n_clusters,
                  cluster_scores,
                  var_threshold,
                  final_accepted_labels,
                  cluster_representatives)

    summary_path = save_summary(results_folder,
                                dataset_name,
                                trigram_h5_path,
                                features,
                                n_clusters,
                                var_threshold,
                                best_threshold,
                                accepted_candidates,
                                final_accepted_labels,
                                trigram_context,
                                reassigned_by_context,
                                total_reassignment_events,
                                refinement_passes_completed,
                                enable_context_refinement)

    print(f"\nResults saved in {results_folder}")
    print(f"Summary saved in {summary_path}")


if __name__ == "__main__":
    main()