[![Python Versions](https://img.shields.io/pypi/pyversions/spineps)](https://pypi.org/project/spineps/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


# PARASIDE: **PARA**nasal **S**egmentation for **I**maging-based **D**isease **E**valuation

<div align="center">
<a href="https://github.com/Hendrik-code/paraside/blob/main/images/logo.jpeg"><img src="https://github.com/Hendrik-code/paraside/blob/main/images/logo.jpeg" width="512" ></a>
</div>


This is the official repository of the paper "PARASIDE: An Automatic Paranasal Sinus Segmentation and Structure Analysis Tool for Magnetic Resonance Imaging"

PARASIDE enables the automated whole-paranasal sinus segmentation for T1-weighted MRI, extracting quantitative features.


## Citation

If you are using PARASIDE, please cite the following:

```
@article{moller2025paraside,
  title={PARASIDE: An Automatic Paranasal Sinus Segmentation and Structure Analysis Tool for MRI},
  author={M{\"o}ller, Hendrik and Krautschick, Lukas and Atad, Matan and Graf, Robert and Busch, Chia-Jung and Beule, Achim and Scharf, Christian and Kaderali, Lars and Menze, Bjoern and Rueckert, Daniel and Paperlein, Fabian},
  journal={arXiv preprint arXiv:2501.14514},
  year={2025}
}
```

## Installation

The order of the following instructions is important!

1. Use Conda or Pip to create a venv for python 3.11, we are using conda for this example:
```bash
conda create --name paraside python=3.11
conda activate paraside
conda install pip
```
2. Go to <a href="https://pytorch.org/get-started/locally/">https://pytorch.org/get-started/locally/</a> and install a correct pytorch version for your machine in your venv
3. Confirm that your pytorch package is working! Try calling these commands:
```bash
nvidia-smi
```
This should show your GPU and it's usage.
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
This should throw no errors and return True

In order to use this code, you need to install the TPTBox (https://github.com/Hendrik-code/TPTBox) among other packages. See pyproject.toml for the required packages.
```python
pip install TPTBox
```

You can also just install all requirements by calling while on the directory level of the pyproject.toml
```python
pip install -e .
```

(Optionally) Download the latest model weights from the release section (https://github.com/Hendrik-code/paraside/releases).

If you have done so, you can use PARASIDE like this:

```python
from TPTBox import NII
from src.load_model import load_model
from src.inference_function import segment_nose
from src.ethmoid_split import split_ethmoid
from src.calc_features import compute_features

# load the image you want to process
input_file = "<path-to-your-file>.nii.gz"
input_nii: NII = NII.load(input_file, seg=False)

# load the segmentation model
paraside_segmentation_model = load_model(model_path_or_version)

# create the segmentation mask
segmentation: NII = segment_nose(
    image_nii=input_nii,
    model=paraside_segmentation_model,
)
# save the nifti back to disk
segmentation.save(output_dir + f"{input_filename}_segmentation.nii.gz")

# split the ethmoid segmentation into anterior and posterior part
segmentation_ethmoid_split = split_ethmoid(segmentation)
# save the nifti back to disk
segmentation_ethmoid_split.save(output_dir + f"{input_filename}_segmentation_ethmoid_split.nii.gz")

# take measurements
features: dict[str, float] = compute_features(
    input_nii,
    segmentation_ethmoid_split,
)
```


## Segmentation Label

| Label | Structure |
| :---: | --------- |
| 1  | Air Maxillaris right |
| 2  | Air Maxillaris left |
| 3  | Air Frontalis right |
| 4  | Air Frontalis left |
| 5  | Air Sphenoid right |
| 6  | Air Sphenoid left |
| 7  | Air Ethmoid right |
| 8  | Air Ethmoid left |

If the ethmoid got split into multiple subparts, the labels are:
| Label | Structure |
| :---: | --------- |
| 7     | Air Ethmoid Anterior right |
| 8     | Air Ethmoid Anterior left |
| 107  | Air Ethmoid Posterior right |
| 108  | Air Ethmoid Posterior right |
| 207  | Air Osteometeal Complex right |
| 208  | Air Osteometeal Complex left |

The labels above are all the AIR labels. The corresponding SOFT TISSUE labels can be gathered by adding 8 on the airlabel.

e.G.:
| Label | Structure |
| :---: | --------- |
| 9  | Soft tissue Maxillaris right |
| 10  | Soft tissue Maxillaris left |
| 11  | Soft tissue Frontalis right |
| 12  | Soft tissue Frontalis left |
| 13  | Soft tissue Sphenoid right |
| 14  | Soft tissue Sphenoid left |
| 15  | Soft tissue Ethmoid right |
| 16  | Soft tissue Ethmoid left |

## Authorship

This pipeline was created in a collaboration by:


| Hendrik Möller (he/him) | Dr. Fabian Paperlein (he/him) |
| :--------- | :--------- |
| PhD Researcher | Medical Doctor |
| Department for Interventional and Diagnostic Neuroradiology  | Department of Otorhinolaryngology, Head and Neck Surgery |
| University Hospital rechts der Isar at Technical University of Munich <br> Ismaninger Street 22, 81675 Munich | University Medicine Greifswald <br> Fleischmann Straße 10, Greifswald, 17475, Mecklenburg-Western |
| - https://deep-spine.de/ <br>- https://aim-lab.io/author/hendrik-moller/  | - https://www.researchgate.net/profile/Fabian-Paperlein <br> - https://www.medizin.uni-greifswald.de/hno/klinik/mitarbeitende/schwitzing-fabian/ <br> - https://www.medizin.uni-greifswald.de/hno/forschung/ki-basierte-diagnostik/|
| hendrik.moeller[at]tum.de | fabian.paperlein[at]med.uni-greifswald.de |

We thank all co-authors and contributors!



## License

Copyright 2025 Hendrik Möller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
