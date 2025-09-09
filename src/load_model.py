####################
# Saved models and their direct paths for development
####################
import shutil
import urllib.request
import zipfile
from pathlib import Path

from spineps.get_models import get_actual_model
from spineps.seg_model import Segmentation_Model
from TPTBox import Print_Logger
from tqdm import tqdm


def load_model(
    model: str | Path | int,
    folds: list[str | int] = ["all"],
) -> Segmentation_Model:
    """
    Load a model by its name, path, or version number.
    """
    if isinstance(model, int):
        return load_model_by_version(model, folds=folds)
    elif isinstance(model, (str, Path)):
        if Path(model).exists():
            return load_model_by_path(model, folds=folds)
        else:
            raise ValueError(f"Model path {model} does not exist.")
    else:
        raise TypeError("Model must be a string, Path, or integer.")


def load_model_by_path(
    path_dir: str | Path,
    folds: list[str | int] = ["all"],
) -> Segmentation_Model:
    """Load a model from a specified directory."""
    return get_actual_model(in_config=path_dir).load(folds=folds)


def load_model_by_version(
    version: int,
    folds: list[str | int] = ["all"],
) -> Segmentation_Model:
    """
    Load a model by its version number.
    """
    modelname = f"Paraside_model_weights_v{version}"
    path = f"model_weights/{modelname}"

    if Path(path).exists():
        return load_model_by_path(path)
    else:
        print(f"Model version {version} not found locally. Downloading...")
        weights_url = f"https://github.com/Hendrik-code/paraside/releases/download/v1.0.0/{modelname}.zip"

        download_weights(weights_url, path)
        return load_model_by_path(path, folds=folds)


def download_weights(weights_url, out_path) -> None:
    out_path = Path(out_path)
    logger = Print_Logger()
    try:
        # Retrieve file size
        with urllib.request.urlopen(str(weights_url)) as response:
            file_size = int(response.info().get("Content-Length", -1))
    except Exception:
        logger.on_fail("Download attempt failed:", weights_url)
        return
    logger.print("Downloading pretrained weights...")

    with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc=Path(weights_url).name) as pbar:

        def update_progress(block_num: int, block_size: int, total_size: int) -> None:
            if pbar.total != total_size:
                pbar.total = total_size
            pbar.update(block_num * block_size - pbar.n)

        zip_path = Path(str(out_path) + ".zip")
        # Download the file
        urllib.request.urlretrieve(str(weights_url), zip_path, reporthook=update_progress)

    logger.print("Extracting pretrained weights...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_path)
    # Test if there is an additional folder and move the content on up.
    if not Path(out_path, "inference_config.json").exists():
        source = next(out_path.iterdir())
        assert source.is_dir()
        for i in source.iterdir():
            shutil.move(i, out_path)

    zip_path.unlink()
