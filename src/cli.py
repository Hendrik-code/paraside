import argparse
import os
from argparse import Namespace
from pathlib import Path
import sys

from TPTBox import NII

from calc_features import compute_features
from ethmoid_split import split_ethmoid
from inference_function import segment_nose
from load_model import load_model
from save_json import save_json


def entry_point():
    main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    main_parser.add_argument("--input", "--i", type=str, required=True, help="Path to the input file or folder")
    main_parser.add_argument("--model", "--m", type=str, required=True, help="Path to the trained model file or model version")
    main_parser.add_argument(
        "--cpu",
        type=bool,
        default=False,
        help="Whether to use CPU for model inference",
        action=argparse.BooleanOptionalAction,
    )

    opt = main_parser.parse_args()
    print(opt)
    print()

    input_path = Path(opt.input)
    assert input_path.exists(), f"Input path {input_path} does not exist."
    use_cpu = opt.cpu

    # try to load model
    try:
        paraside_segmentation_model = load_model(opt.model, use_cpu=use_cpu)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {opt.model}: {e}")

    # if input_path is directory, process all files in the directory
    if input_path.is_dir():
        input_files = list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.nii"))
        output_dir = input_path / "paraside_output"
        os.makedirs(output_dir, exist_ok=True)
    else:
        input_files = [input_path]
        output_dir = input_path.parent / "paraside_output"
        os.makedirs(output_dir, exist_ok=True)

    # Loop over files
    for input_file in input_files:
        # Make output paths and check if they already exist
        out_seg = Path(output_dir / f"{input_file.name.split('.')[0]}_segmentation_ethmoid_split.nii.gz")
        out_json = Path(output_dir / f"{input_file.name.split('.')[0]}_features.json")
        if out_seg.exists() and out_json.exists():
            print(f"Outputs for {input_file} already exists. Skipping.")
            continue

        # load the image you want to process
        try:
            input_nii: NII = NII.load(input_file, seg=False)
        except Exception as e:
            print(f"Failed to load NIfTI file {input_file}: {e}")
            continue

        if not out_seg.exists():
            # create the segmentation mask
            segmentation: NII = segment_nose(
                image_nii=input_nii,
                model=paraside_segmentation_model,
            )

            # split the ethmoid segmentation into anterior and posterior part
            segmentation_ethmoid_split = split_ethmoid(segmentation)
            segmentation_ethmoid_split.save(out_seg)
        else:
            segmentation_ethmoid_split = NII.load(out_seg, seg=True)

        # take measurements
        features: dict[str, float] = compute_features(
            input_nii,
            segmentation_ethmoid_split,
        )
        # save features as json
        save_json(features, out_json)

    print(f"Processed {len(input_files)} files. Outputs saved to {output_dir}")


if __name__ == "__main__":
    entry_point()
