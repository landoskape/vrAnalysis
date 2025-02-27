from vrAnalysis import fileManagement as fm


def get_classification_dir():
    """A central directory for storing ROICaT classification data and results."""
    roicat_classification_dir = fm.analysisPath() / "roicat_classification"
    if not roicat_classification_dir.exists():
        roicat_classification_dir.mkdir(parents=True)
    return roicat_classification_dir


def get_classifier_files():
    """A dictionary of files for storing ROICaT classification data and results."""
    roicat_classification_dir = get_classification_dir()
    files = {
        # For training data
        "train_sessions": roicat_classification_dir / "train_sessions.json",
        "train_latents": roicat_classification_dir / "train_latents.npy",
        "train_embeddings": roicat_classification_dir / "train_embeddings.npy",
        "train_images": roicat_classification_dir / "train_images.npy",
        "train_umap": roicat_classification_dir / "train_umap.joblib",
        "train_labels": roicat_classification_dir / "train_labels.csv",
        "train_classifier": roicat_classification_dir / "train_classifier.joblib",
        # Now for testing data
        "test_sessions": roicat_classification_dir / "test_sessions.json",
        "test_latents": roicat_classification_dir / "test_latents.npy",
        "test_embeddings": roicat_classification_dir / "test_embeddings.npy",
        "test_images": roicat_classification_dir / "test_images.npy",
        "test_labels": roicat_classification_dir / "test_labels.csv",
    }
    return files


# Run data through all sessions
def get_results_path(session):
    """The path to store the results of a ROICaT classification for a given session."""
    if hasattr(session, "roicat_path"):
        roicat_path = session.roicat_path
    elif hasattr(session, "roicatPath"):
        roicat_path = session.roicatPath()
    else:
        raise ValueError("Session does not have a roicat path! Session type: ", type(session))
    results_path = roicat_path / "classification_results.joblib"
    if not results_path.parent.exists():
        results_path.parent.mkdir(parents=True, exist_ok=True)
    return results_path
