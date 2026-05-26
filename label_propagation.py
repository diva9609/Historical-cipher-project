import os
import h5py
import numpy as np
from natsort import natsorted
from sklearn.preprocessing import normalize
from sklearn.semi_supervised import LabelSpreading


def start_label_propagation_h5(imgsPath,
                               dstPath,
                               labeled_clusters_path,
                               h5_path,
                               alpha=0.1,
                               parallel=-1,
                               decision_boundary=0.4,
                               normalize_features=True):


    def extract_symbol(cluster_label):
        if cluster_label == "?":
            return "?"
        parts = cluster_label.split("_", 2)
        return parts[2] if len(parts) > 2 else cluster_label

    os.makedirs(dstPath, exist_ok=True)

    if alpha <= 0:
        alpha = 0.01
    if alpha >= 1:
        alpha = 0.99


    images = natsorted(os.listdir(imgsPath), key=lambda x: x.lower())
    images = [img for img in images if os.path.isfile(os.path.join(imgsPath, img))]

    print(f"Found {len(images)} images in imgsPath")


    train_label_images = {}
    label_to_user_label = {}

    label_folders = [folder for folder in natsorted(os.listdir(labeled_clusters_path), key=lambda x: x.lower())
                     if os.path.isdir(os.path.join(labeled_clusters_path, folder))]

    if len(label_folders) == 0:
        raise ValueError(f"No label folders found in: {labeled_clusters_path}")

    for num, labeled_folder in enumerate(label_folders):
        auxpath = os.path.join(labeled_clusters_path, labeled_folder)
        seed_images = natsorted(os.listdir(auxpath), key=lambda x: x.lower())

        for imagename in seed_images:
            full_seed_path = os.path.join(auxpath, imagename)
            if os.path.isfile(full_seed_path):
                train_label_images[imagename] = num

        label_to_user_label[num] = labeled_folder

    with h5py.File(h5_path, "r") as h5_file:
        all_features = h5_file["features"][:]
        all_image_names = h5_file["image_names"][:]

    all_image_names_clean = [(name.decode("utf-8")
                              if isinstance(name, bytes)
                              else str(name)) for name in all_image_names]
    
    all_image_names_clean = [os.path.basename(name) for name in all_image_names_clean]

    name_to_idx = {}
    for idx, name in enumerate(all_image_names_clean):
        name_to_idx[name] = idx

        base, ext = os.path.splitext(name)
        if ext == "":
            name_to_idx[base + ".png"] = idx
            name_to_idx[base + ".jpg"] = idx
            name_to_idx[base + ".jpeg"] = idx
            name_to_idx[base + ".tif"] = idx
            name_to_idx[base + ".tiff"] = idx
        else:
            name_to_idx[base] = idx


    matched_images = []
    X_train = []
    Y_train = []

    for image in images:
        idx = name_to_idx[image]

        matched_images.append(image)
        X_train.append(all_features[idx])

        if image in train_label_images:
            Y_train.append(train_label_images[image])
        else:
            Y_train.append(-1)

    X_train = np.array(X_train, dtype=np.float64)
    Y_train = np.array(Y_train, dtype=np.int32)

    if normalize_features:
        X_train = normalize(X_train, norm="l2")

    labeled_count = int(np.sum(Y_train != -1))
    unlabeled_count = int(np.sum(Y_train == -1))

    print(f"Labeled: {labeled_count} | Unlabeled: {unlabeled_count}")



    lp_model = LabelSpreading(kernel="knn",
                              n_neighbors=11,
                              max_iter=30000,
                              tol=0.0001,
                              alpha=alpha,
                              n_jobs=parallel)

    lp_model.fit(X_train, Y_train)


    final_symbols = []

    prob_file = os.path.join(dstPath, "label_propagation_output_prob.txt")

    with open(prob_file, "w", encoding="utf-8") as f:

        for label, image, distribution in zip(lp_model.transduction_,
                                              matched_images,
                                              lp_model.label_distributions_):
            prob = float(np.max(distribution))

            if np.isnan(prob) or prob < decision_boundary:
                raw_label = "?"
                final_symbol = "?"
                f.write(image + "\t?\t" + str(prob) + "\n")

            else:
                raw_label = label_to_user_label[label]
                final_symbol = extract_symbol(raw_label)
                f.write(image + "\t" + raw_label + "\t" + str(prob) + "\n")

            final_symbols.append(final_symbol)



    final_string_file = os.path.join(dstPath, "label_propagation_final_string.txt")

    with open(final_string_file, "w", encoding="utf-8") as f:
        f.write(" ".join(final_symbols))



    summary_file = os.path.join(dstPath, "label_propagation_summary.txt")

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Total images: {len(images)}\n")
        f.write(f"Matched images: {len(matched_images)}\n")
        f.write(f"Labeled samples: {labeled_count}\n")
        f.write(f"Unlabeled samples: {unlabeled_count}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Decision boundary: {decision_boundary}\n")
        f.write(f"Normalize features: {normalize_features}\n")
        f.write(f"Prob file: {prob_file}\n")
        f.write(f"Final string file: {final_string_file}\n")

    print(f"Saved summary to: {summary_file}")


if __name__ == "__main__":
    start_label_propagation_h5(imgsPath="2Segmentation/Ramanacoil/symbols",
                               dstPath="4LabelPropagation/Ramanacoil_x_15",
                               labeled_clusters_path="3Clustering/Hierarchical/Ramanacoil_x_15/Accepted",
                               h5_path="features/ciphers/Ramanacoil/features.h5",
                               alpha=0.1,
                               parallel=-1,
                               decision_boundary=0.4,
                               normalize_features=True)