from pathlib import Path

from TPTBox import NII

from src.calc_features import compute_features
from src.ethmoid_split import split_ethmoid
from src.inference_function import segment_nose
from src.load_model import load_model


def run_example(
    input_file: str,
    model_path_or_version: str | int,
    output_dir: str,
):
    """
    Example function to run the segmentation and feature extraction.

    Args:
        input_file (str): Path to the input NIfTI file.
        model_path (str): Path to the segmentation model.
        output_file (str): Path to save the segmented NIfTI file.
    """
    input_filename = Path(input_file).name if isinstance(input_file, str) else input_file.name
    input_filename = input_filename.split(".")[0]  # remove file extension
    # load the image you want to process
    input_nii: NII = NII.load(input_file, seg=False)
    print(f"Loaded nifti from {input_file}: {input_nii}")

    # load the segmentation model
    paraside_segmentation_model = load_model(model_path_or_version)

    # create the segmentation mask
    segmentation: NII = segment_nose(
        image_nii=input_nii,
        model=paraside_segmentation_model,
    )
    print(f"Segmentation completed: {segmentation}")

    # save the nifti back to disk
    segmentation.save(output_dir + f"{input_filename}_segmentation.nii.gz")

    # split the ethmoid segmentation into anterior and posterior part
    segmentation_ethmoid_split = split_ethmoid(segmentation)
    print(f"Ethmoid segmentation split completed: {segmentation_ethmoid_split}")
    # save the nifti back to disk
    segmentation_ethmoid_split.save(output_dir + f"{input_filename}_segmentation_ethmoid_split.nii.gz")

    # take measurements

    features: dict[str, float] = compute_features(
        input_nii,
        segmentation_ethmoid_split,
    )
    return segmentation_ethmoid_split, features


if __name__ == "__main__":
    import numpy as np

    # Example usage
    segmentation_nii, features = run_example(
        input_file="test_MPR.nii.gz",
        model_path_or_version=1,
        output_dir="",
    )
    print("Segmentation and feature extraction completed.")
    print()
    print("Extracted features:")
    for fk, fv in features.items():
        if isinstance(fv, list):
            fvv = f"{round(np.mean(fv), 3)} +- {round(np.std(fv), 3)}"
        elif isinstance(fv, float):
            fvv = round(fv, 3)
        else:
            fvv = fv
        print(f"{fk}: {fvv}")
