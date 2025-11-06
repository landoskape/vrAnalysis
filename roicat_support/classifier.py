from pathlib import Path
import tempfile
import joblib
import json
import pandas as pd
import numpy as np
from scipy.sparse import coo_array
from scipy.special import softmax
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from umap import UMAP
import matplotlib.pyplot as plt
from syd import make_viewer

from _old_vrAnalysis import database
from _old_vrAnalysis.helpers import get_confirmation
from .files import get_classifier_files, get_results_path

import importlib.util

roicat_available = importlib.util.find_spec("roicat") is not None
if roicat_available:
    # ROICaT is only installed in the special ROICaT environment
    # See this modules init file docstring for more details.
    # Note that if it isn't available and you try to run code that depends
    # on ROICaT, it'll raise an error!
    import roicat


def choose_sessions(sessions, num_train_sessions, num_test_sessions, num_planes):
    train_sessions = list(np.random.choice(sessions, size=num_train_sessions, replace=False))
    remaining_sessions = [ses for ses in sessions if ses not in train_sessions]
    if len(remaining_sessions) < num_test_sessions:
        test_sessions = list(
            np.append(np.random.choice(sessions, size=num_test_sessions - len(remaining_sessions), replace=False), remaining_sessions)
        )
    else:
        test_sessions = list(np.random.choice(remaining_sessions, size=num_test_sessions, replace=False))

    train_planes = []
    test_planes = []
    for session in train_sessions:
        planeIDs = session.planeIDs
        if len(planeIDs) < num_planes:
            raise ValueError(f"Session {session.sessionid} has only {len(planeIDs)} planes")
        planes_to_use = list(map(int, np.random.choice(planeIDs, size=num_planes, replace=False)))
        train_planes.append(planes_to_use)

    for session in test_sessions:
        planeIDs = session.planeIDs
        if session in train_sessions:
            planeIDs = [p for p in planeIDs if p not in train_planes[train_sessions.index(session)]]
        if len(planeIDs) < num_planes:
            raise ValueError(f"Session {session.sessionid} has only {len(planeIDs)} planes")
        planes_to_use = list(map(int, np.random.choice(planeIDs, size=num_planes, replace=False)))
        test_planes.append(planes_to_use)

    session_choices = dict(
        train_sessions=train_sessions,
        test_sessions=test_sessions,
        train_planes=train_planes,
        test_planes=test_planes,
    )
    return session_choices


def define_classification_set(run_again: bool = False, overwrite: bool = False):
    """Define the training and testing sets for the classification task.

    This function is wrapped in fail-safes because it really shouldn't be used more than once,
    but of course it's useful to see how it worked and to run it again if necessary.

    The idea is to pick sessions and planes in a way that is representative of all the data
    across mice and planes as randomly as possible.

    Parameters
    ----------
    run_again : bool
        Whether to run the function again. Default is False to avoid overwriting.
    overwrite : bool
        Whether to overwrite the existing training and testing sets.
        You have to manually set this to True if you want to save the sets after running it.
        (Can set to false when overwrite is True to observe the function in action without overwriting.)
    """
    files = get_classifier_files()
    sessiondb = database.vrDatabase("vrSessions")

    if run_again:
        confirmation = get_confirmation("This will overwrite the existing training and testing sets.")
        if not confirmation:
            print("Operation cancelled.")
            return

        # Make training set for labeling (won't actually label them all but will use them to make the model and train the UMAP)
        sessions = sessiondb.iterSessions(imaging=True)
        mice_names = list(set([ses.mouseName for ses in sessions]))

        # Set metadata for training & testing set
        train_sessions_per_mouse = 2
        test_sessions_per_mouse = 1
        num_planes_per_session = 2

        # Get sessions for each mouse
        train_data = []
        test_data = []
        for mouse in mice_names:
            mouse_sessions = [ses for ses in sessions if ses.mouseName == mouse]
            session_choices = choose_sessions(mouse_sessions, train_sessions_per_mouse, test_sessions_per_mouse, num_planes_per_session)

            for train_session, train_plane in zip(session_choices["train_sessions"], session_choices["train_planes"]):
                for plane in train_plane:
                    train_data.append(
                        {
                            "mouseName": train_session.mouseName,
                            "dateString": train_session.dateString,
                            "sessionid": train_session.sessionid,
                            "plane_id": plane,
                            "num_rois": train_session.roiPerPlane[train_session.planeIDs.index(plane)],
                        }
                    )

            for test_session, test_plane in zip(session_choices["test_sessions"], session_choices["test_planes"]):
                for plane in test_plane:
                    test_data.append(
                        {
                            "mouseName": test_session.mouseName,
                            "dateString": test_session.dateString,
                            "sessionid": test_session.sessionid,
                            "plane_id": plane,
                            "num_rois": test_session.roiPerPlane[test_session.planeIDs.index(plane)],
                        }
                    )

        if overwrite:
            # Save the training and testing sets
            with open(files["train_sessions"], "w") as f:
                json.dump(train_data, f, indent=2)
            with open(files["test_sessions"], "w") as f:
                json.dump(test_data, f, indent=2)
            return train_data, test_data

    else:
        print("Without setting the run_again and overwrite flags to True, this function will not run. Be sure you want to overwrite existing sets.")


def load_classification_set(use_training_data):
    files = get_classifier_files()
    session_path = files["train_sessions"] if use_training_data else files["test_sessions"]
    print("Loading training data from:", session_path)
    with open(session_path, "r") as f:
        data = json.load(f)
    return data


def prepare_suite2p_paths(use_training_data):
    data = load_classification_set(use_training_data)

    pathSuffixToStat = "stat.npy"
    pathSuffixToOps = "ops.npy"

    paths_stat = []
    paths_ops = []
    for session in data:
        ses = session.vrExperiment(session["mouseName"], session["dateString"], session["sessionid"])
        paths_stat.append(ses.suite2pPath() / f"plane{session['plane_id']}" / pathSuffixToStat)
        paths_ops.append(ses.suite2pPath() / f"plane{session['plane_id']}" / pathSuffixToOps)

    for pstat, pops in zip(paths_stat, paths_ops):
        pstat = Path(pstat)
        pops = Path(pops)
        parent_dir = pstat.resolve().parent
        check_parent_dir = pops.resolve().parent == parent_dir
        if not check_parent_dir:
            raise ValueError("Parent dirs don't match!", pstat, pops)

    return paths_stat, paths_ops


def roi_should_be_ignored(roi_image: np.ndarray):
    csr = coo_array(roi_image)
    rows, cols = csr.row, csr.col

    # can't use scipy.interpolate.griddata with 1d values
    is_horz = np.unique(rows).size == 1
    is_vert = np.unique(cols).size == 1

    # check for diagonal pixels
    # slope = rise / run --- don't need to check if run==0
    rdiff = np.diff(rows)
    cdiff = np.diff(cols)
    is_diag = np.unique(cdiff / rdiff).size == 1 if not np.any(rdiff == 0) else False

    # best practice to just convolve instead of interpolating if too few pixels
    is_smol = rows.size < 3

    return is_horz or is_vert or is_smol or is_diag


def generate_latents_and_embeddings(use_training_data, run_again: bool = False, overwrite: bool = False):
    """
    This part of the pipeline generates the latents and embeddings for the training and testing data.

    Like above, it's important to do this once and then leave the data forever so we have failsafes to prevent
    overwriting unless you really want to redo it. (There's going to be a lot of downstream analyses that rely
    on this data being consistent at least with regards to the labels).

    Parameters
    ----------
    use_training_data : bool
        Whether to use the training data or the testing data.
    run_again : bool
        Whether to run the function again. Default is False to avoid overwriting.
    overwrite : bool
        Whether to overwrite the existing latents and embeddings.
    """
    files = get_classifier_files()

    if run_again:
        confirmation = get_confirmation("This will overwrite the existing latents, embeddings, and images.")
        if not confirmation:
            print("Operation cancelled.")
            return

        paths_statFiles, paths_opsFiles = prepare_suite2p_paths(use_training_data)

        data = roicat.data_importing.Data_suite2p(
            paths_statFiles=paths_statFiles,
            paths_opsFiles=paths_opsFiles,
            new_or_old_suite2p="new",
            verbose=True,
        )

        for idx_session, roi_image in enumerate(data.ROI_images):
            keep_rois = np.array([not roi_should_be_ignored(roi) for roi in roi_image])
            data.ROI_images[idx_session] = roi_image[keep_rois]

        assert data.check_completeness(verbose=False)["classification_inference"], f"Data object is missing attributes necessary for tracking."

        DEVICE = roicat.helpers.set_device(use_GPU=True, verbose=True)
        dir_temp = tempfile.gettempdir()

        roinet = roicat.ROInet.ROInet_embedder(
            device=DEVICE,  ## Which torch device to use ('cpu', 'cuda', etc.)
            dir_networkFiles=dir_temp,  ## Directory to download the pretrained network to
            download_method="check_local_first",  ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
            download_url="https://osf.io/c8m3b/download",  ## URL of the model
            download_hash="357a8d9b630ec79f3e015d0056a4c2d5",  ## Hash of the model file
            forward_pass_version="head",  ## How the data is passed through the network
            verbose=False,  ## Whether to print updates
        )

        roinet.generate_dataloader(
            ROI_images=data.ROI_images,  ## Input images of ROIs
            um_per_pixel=data.um_per_pixel,  ## Resolution of FOV
            pref_plot=False,  ## Whether or not to plot the ROI sizes
        )

        roinet.generate_latents()

        model_umap = UMAP(
            n_neighbors=25,
            n_components=2,
            n_epochs=400,
            verbose=False,
            densmap=False,
        )
        emb = model_umap.fit_transform(roinet.latents)
        images = np.concatenate(data.ROI_images, axis=0)

        if overwrite:
            latents_path = files["train_latents"] if use_training_data else files["test_latents"]
            embeddings_path = files["train_embeddings"] if use_training_data else files["test_embeddings"]
            images_path = files["train_images"] if use_training_data else files["test_images"]
            np.save(latents_path, roinet.latents)
            np.save(embeddings_path, emb)
            np.save(images_path, images)

            if use_training_data:
                joblib.dump(model_umap, files["train_umap"])

    else:
        print("Without setting the run_again and overwrite flags to True, this function will not run. Be sure you want to overwrite existing sets.")


def load_latents_and_embeddings(use_training_data):
    files = get_classifier_files()
    data = dict(
        latents=np.load(files["train_latents"] if use_training_data else files["test_latents"]),
        embeddings=np.load(files["train_embeddings"] if use_training_data else files["test_embeddings"]),
        images=np.load(files["train_images"] if use_training_data else files["test_images"]),
        model_umap=joblib.load(files["train_umap"]) if use_training_data else None,
    )
    return data


def read_labels(file_path: str):
    """Get the results of classifying using the integrated labeler."""
    df = pd.read_csv(file_path, usecols=["image_index", "label"])
    return df


def labels_to_df(labels: dict):
    new_df = pd.DataFrame.from_dict(labels, orient="index", columns=["label"])
    new_df.index.name = "image_index"
    new_df.reset_index(inplace=True)
    return new_df


def save_labels(file_path: str, labels: dict, overwrite: bool = False):
    ## make directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Make two-column dataframe with image_index and label
    new_df = labels_to_df(labels)

    ## check if the file exists already
    if Path(file_path).exists():
        if overwrite:
            confirmation = get_confirmation(
                "A previous labels file exists. Are you sure you want to overwrite it? (To extend instead, change the kwargs...)"
            )
            if not confirmation:
                print("Operation cancelled -- call this method again with overwrite=True to append to the existing file.")
                return
        else:
            # If it does, read the existing file and combine the labels
            saved_df = read_labels(file_path)
            new_df = pd.concat([saved_df, new_df]).drop_duplicates(subset="image_index", keep="last")

    ## then save the results
    new_df.to_csv(file_path, index=False)


def labels_df_to_dict(df: pd.DataFrame):
    """In case I need to convert the loaded DF back to a dict (like labeler.labels_)"""
    return df.set_index("image_index")["label"].to_dict()


def save_classifier(classification_results: dict, overwrite: bool = False):
    files = get_classifier_files()
    if overwrite:
        confirmation = get_confirmation("This will overwrite the existing logistic regression model of the training data.")
        if not confirmation:
            print("Operation cancelled.")
            return
        else:
            joblib.dump(classification_results, files["train_classifier"])


def load_classifier():
    files = get_classifier_files()
    return joblib.load(files["train_classifier"])


def detect_local_concavities(mask, iterations=3):
    binary = (255 * (mask > 0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Elliptical kernel
    dilated = cv2.dilate(binary, kernel, iterations=iterations)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Reverse dilation
    filled_mask = cv2.erode(filled_mask, kernel, iterations=iterations)

    # Measure how much of the local convex hull is filled in
    filled_mask = 1.0 * (filled_mask > 0)
    fraction_filled = np.sum((mask > 0) * filled_mask) / np.sum(filled_mask)
    return fraction_filled, filled_mask


def run_integrated_labeler(embeddings, images, label_path, overwrite: bool = False):
    # The idx to the overlay images is designed to pick images to show that
    # are well spread out in the embedding space from UMAP
    idx_images_overlay = roicat.visualization.get_spread_out_points(
        embeddings,
        n_ims=min(embeddings.shape[0], 1500),  ## Select number of overlayed images here
        dist_im_to_point=0.8,
    )

    with roicat.helpers.IntegratedLabeler(
        images,
        embeddings=embeddings,
        idx_images_overlay=idx_images_overlay,
        size_images_overlay=0.25,
        frac_overlap_allowed=0.25,
        crop_images_overlay=0.25,
        alpha_points=1.0,
        size_points=3.0,
    ) as labeler:
        labeler.run()

    # Note that this function will extend previous labels in the file rather than overwriting the file!!!
    save_labels(label_path, labeler.labels_, overwrite=overwrite)

    return labeler


def update_labels(embeddings, images, label_path):
    labels = labels_df_to_dict(read_labels(label_path))
    labels_index = list(labels.keys())
    labels_label = list(labels.values())

    def get_idx_selected(target):
        if target == "all":
            return labels_index
        else:
            return [i for i, l in zip(labels_index, labels_label) if l == target]

    def plot(state):
        idx = state["idx"]
        label = labels[idx]
        idx_to_selected = get_idx_selected(state["label"])
        fig, ax = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
        ax[0].imshow(images[idx])
        ax[1].scatter(embeddings[:, 0], embeddings[:, 1], c="black", s=1, alpha=0.01)
        ax[1].scatter(embeddings[idx_to_selected, 0], embeddings[idx_to_selected, 1], c="blue", s=3, alpha=0.25)
        ax[1].scatter(embeddings[idx, 0], embeddings[idx, 1], c="red", s=20)
        fig.suptitle(f"Label: {label}")
        return fig

    def update_index(state):
        idx_to_selected = get_idx_selected(state["label"])
        viewer.update_selection("idx", options=idx_to_selected)

    viewer = make_viewer(plot)
    viewer.add_selection("idx", value=labels_index[0], options=labels_index)
    viewer.add_selection("label", value="all", options=list(np.unique(labels_label)) + ["all"])
    update_index(viewer.state)

    labels_to_change = {}
    labels_to_clear = []

    def clear_label(state):
        labels_to_clear.append(state["idx"])

    def change_label(state):
        if state["new_label"] not in ["c", "b", "d", "l"]:
            print("No change --- new label not permitted")
        else:
            labels_to_change[state["idx"]] = state["new_label"]

    def update_change_message(state):
        viewer.update_button("change_label", label=f"Change to {state['new_label']}")

    viewer.add_button("clear_label", label="Clear!", callback=clear_label)
    viewer.add_text("new_label", value="")
    viewer.add_button("change_label", label="Change!", callback=change_label)
    viewer.on_change("new_label", update_change_message)
    viewer.on_change("label", update_index)
    viewer = viewer.deploy(env="notebook")

    return labels_to_change, labels_to_clear


def execute_label_updates(label_path, labels_to_change, labels_to_clear, show_updates: bool = True, execute_updates: bool = False):
    if show_updates:
        labels = labels_df_to_dict(read_labels(label_path))
        print("Changing these labels:", labels_to_change)
        print("Clearing these labels:", labels_to_clear)
        if execute_updates:
            for l in labels_to_clear:
                if l in labels:
                    labels.pop(l)
            for l in labels_to_change:
                labels[l] = labels_to_change[l]
            save_labels(label_path, labels, overwrite=True)


def visualize_counts(label_path):
    # Visualize the counts
    labels_df = read_labels(label_path)
    u, c = np.unique(labels_df["label"], return_counts=True)

    show_counts = True
    if show_counts:
        plt.close("all")
        plt.figure()
        plt.bar(u, c)
        plt.xlabel("label")
        plt.ylabel("counts")
        plt.show()


def visualize_examples(images, label_path, max_images_per_label=10, shuffle=True):
    labels_df = read_labels(label_path)
    roicat.visualization.display_labeled_ROIs(
        images=images,
        labels=labels_df.set_index("image_index"),
        max_images_per_label=max_images_per_label,
        figsize=(12, 3),
        fontsize=12,
        shuffle=shuffle,
    )


def train_classifier():
    files = get_classifier_files()
    train_data = load_latents_and_embeddings(use_training_data=True)
    train_latents = train_data["latents"]
    train_label_path = files["train_labels"]

    train_labels = labels_df_to_dict(read_labels(train_label_path))
    unique_labels = set(train_labels.values())
    unique_ids = range(len(unique_labels))
    label_to_id = {label: i for label, i in zip(unique_labels, unique_ids)}
    id_to_label = {i: label for label, i in zip(unique_labels, unique_ids)}

    label_to_description = {
        "c": "cell",
        "d": "dendrite",
        "l": "long",
        "b": "bad",
    }

    X = np.stack([train_latents[i] for i in train_labels])
    y = np.stack([label_to_id[train_labels[i]] for i in train_labels])

    model = LogisticRegression(max_iter=1000).fit(X, y)
    print(f"Accuracy: {model.score(X, y)}")

    classification_results = dict(
        model=model,
        label_to_description=label_to_description,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )

    save_classifier(classification_results, overwrite=True)


def evaluate_classifier(convert_to_goodvsbad: bool = True, show_confusion_matrix: bool = False, checkout_bad_to_good: bool = False):
    files = get_classifier_files()
    classifier = load_classifier()
    label_to_description = classifier["label_to_description"]
    model = classifier["model"]
    label_to_id = classifier["label_to_id"]
    id_to_label = classifier["id_to_label"]
    unique_labels = set(label_to_id.keys())

    # Load test labeling data
    test_data = load_latents_and_embeddings(use_training_data=False)
    test_images = test_data["images"]
    test_latents = test_data["latents"]
    test_label_path = files["test_labels"]

    test_labels = labels_df_to_dict(read_labels(test_label_path))
    if unique_labels != set(test_labels.values()):
        raise ValueError("Unique labels in training and testing sets do not match.")

    Xtest = np.stack([test_latents[i] for i in test_labels])
    ytest = np.stack([label_to_id[test_labels[i]] for i in test_labels])
    print(f"Test Accuracy: {model.score(Xtest, ytest)}")

    model_predictions = model.predict(Xtest)

    if convert_to_goodvsbad:
        good_ids = [label_to_id[label] for label in ["c", "d"]]

        confuse_predictions = np.where(np.isin(model_predictions, good_ids), 1, 0)
        confuse_ytest = np.where(np.isin(ytest, good_ids), 1, 0)
        display_labels = ["bad", "good"]
        idx_labeled_bad_to_good = np.where((confuse_predictions == 1) & (confuse_ytest == 0))[0]
        labeled_index = list(test_labels.keys())
        idx_bad_to_good = [labeled_index[i] for i in idx_labeled_bad_to_good]

    else:
        confuse_predictions = model_predictions
        confuse_ytest = ytest
        display_labels = [label_to_description[id_to_label[i]] for i in range(len(unique_labels))]

    if show_confusion_matrix:
        plt.close("all")
        fig = plt.figure(figsize=(7, 7))
        disp = ConfusionMatrixDisplay.from_predictions(
            confuse_ytest, confuse_predictions, display_labels=display_labels, ax=plt.gca(), colorbar=False, normalize=None
        )
        plt.xticks(range(len(display_labels)), display_labels, rotation=0)
        plt.yticks(range(len(display_labels)), display_labels)
        plt.show()

    if checkout_bad_to_good and convert_to_goodvsbad:
        viewer = make_viewer()
        viewer.add_selection("idx", value=idx_bad_to_good[0], options=idx_bad_to_good)

        def plot(state):
            idx_to_label = labeled_index.index(state["idx"])
            pred_prob = model.predict_proba(test_latents[idx_to_label].reshape(1, -1))[0]
            pred_prob = softmax(pred_prob)[model_predictions[idx_to_label]]
            title = f"Image {state['idx']}\n"
            title += f"Label: {id_to_label[ytest[idx_to_label]]}, Pred: {id_to_label[model_predictions[idx_to_label]]}\n"
            title += f"Prediction Probability: {pred_prob}"
            fig = plt.figure(figsize=(4, 4))
            ax = plt.gca()
            ax.imshow(test_images[state["idx"]])
            ax.set_title(title)
            return fig

        viewer.set_plot(plot)
        viewer.deploy(env="notebook")


def visualize_predictions(model, latents, embeddings, id_to_label):
    prediction = model.predict(latents)
    predicted_label = [id_to_label[i] for i in prediction]
    proba = model.predict_proba(latents)
    maxprob = np.max(proba, axis=1)
    isort = np.argsort(-maxprob)

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")
    ax[0].scatter(embeddings[:, 0], embeddings[:, 1], c=prediction, s=1)
    ax[1].scatter(embeddings[isort, 0], embeddings[isort, 1], c=maxprob[isort], s=1)
    print("Use predicted_label to make a colorbar eventually...")
    plt.show()


def classifier_description():
    description = """
This is a result of classification performed via a logistic regression classifier on roinet latents.
The notebook that produced it is in ./notebooks/roicat_mask_classification.ipynb

The model predicts the class based on the roinet latents. Those are saved in sessionPath() / roicat / roinet_latents.npy

class_predictions: the label output (index, not string label) of the logistic regression model
fill_fraction: the fraction of pixels used in the ROI mask contained in the local convex hull of the mask
            (this uses the detect_local_concavities algorithm)
footprint_size: the number of pixels in the mask
label_to_description: dictionary of string (label) to longer description
label_to_id: dictionary of the label (string, single letter) to the ID used for it (integer)
id_to_label: reverse of label_to_id
    """
    return description


def classify_and_save(session, overwrite: bool = False):
    classifier = load_classifier()
    model = classifier["model"]
    label_to_description = classifier["label_to_description"]
    label_to_id = classifier["label_to_id"]
    id_to_label = classifier["id_to_label"]

    results_path = get_results_path(session)
    if not overwrite:
        if results_path.exists():
            print(f"Skipping {session.sessionPrint()} because results file already exists! (Use overwrite to redo)")
            return

    paths_stat = [session.suite2pPath() / plane_name / "stat.npy" for plane_name in session.planeNames]
    paths_ops = [session.suite2pPath() / plane_name / "ops.npy" for plane_name in session.planeNames]

    data = roicat.data_importing.Data_suite2p(
        paths_statFiles=paths_stat,
        paths_opsFiles=paths_ops,
        new_or_old_suite2p="new",
        verbose=False,
    )

    assert data.check_completeness(verbose=False)["classification_inference"], f"Data object is missing attributes necessary for tracking."

    latents_path = session.roicatPath() / "roinet_latents.npy"
    if latents_path.exists():
        latents = np.load(latents_path)
    else:
        DEVICE = roicat.helpers.set_device(use_GPU=True, verbose=True)
        dir_temp = tempfile.gettempdir()

        roinet = roicat.ROInet.ROInet_embedder(
            device=DEVICE,  ## Which torch device to use ('cpu', 'cuda', etc.)
            dir_networkFiles=dir_temp,  ## Directory to download the pretrained network to
            download_method="check_local_first",  ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
            download_url="https://osf.io/c8m3b/download",  ## URL of the model
            download_hash="357a8d9b630ec79f3e015d0056a4c2d5",  ## Hash of the model file
            forward_pass_version="head",  ## How the data is passed through the network
            verbose=False,  ## Whether to print updates
        )

        roinet.generate_dataloader(
            ROI_images=data.ROI_images,  ## Input images of ROIs
            um_per_pixel=data.um_per_pixel,  ## Resolution of FOV
            pref_plot=False,  ## Whether or not to plot the ROI sizes
        )

        roinet.generate_latents()

        np.save(latents_path, roinet.latents)

        latents = roinet.latents

    images = np.concatenate(data.ROI_images, axis=0)
    model_predictions = model.predict(latents)
    fill_fraction = np.array([detect_local_concavities(image)[0] for image in images])
    footprint_size = np.sum(images > 0, axis=(1, 2))

    classification_results = dict(
        label_to_description=label_to_description,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        class_predictions=model_predictions,
        fill_fraction=fill_fraction,
        footprint_size=footprint_size,
        description=classifier_description(),
    )

    joblib.dump(classification_results, results_path)


def process_sessions():
    sessiondb = database.vrDatabase("vrSessions")
    sessions = sessiondb.iterSessions(imaging=True)
    for isession, session in enumerate(sessions):
        print(f"Processing and classifying session {session.sessionPrint()}, ({isession+1}/{len(sessions)})")
        classify_and_save(session)
