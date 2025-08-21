from enum import Enum, auto
from pathlib import Path
from warnings import simplefilter

import numpy as np
from radiomics import featureextractor
from sklearn.exceptions import ConvergenceWarning
from TPTBox import NII
from TPTBox.core import np_utils
from TPTBox.core.sitk_utils import nii_to_sitk

from src.calc_label_thickness import calc_label_thickness

simplefilter("ignore", category=ConvergenceWarning)

AIRLABELS = [1, 2, 3, 4, 5, 6, 7, 8, 107, 108, 207, 208]
SOFTLABELS = [9, 10, 11, 12, 13, 14, 15, 16, 115, 116, 215, 216]
ALLLABELS = AIRLABELS + SOFTLABELS


class FEATURES(Enum):
    Volume = auto()
    VolumeRelationWhole = auto()
    #
    SizeHeight = auto()  # I
    SizeWidth = auto()  # R
    SizeLength = auto()  # P
    #
    IntensityMin = auto()
    IntensityMax = auto()
    IntensityAvg = auto()
    IntensityStd = auto()
    IntensityLIST = auto()
    #
    SurfaceThicknessLIST = auto()  # list of thicknesses for each voxel in the label
    SurfaceCoverage = auto()  # percentage of surface voxels in the label
    #
    SpatialRelation = auto()
    SpatialRelationMerged = auto()  # wenn label + label+8 merged
    SpatialRelationWhole = auto()  # label against binary whole center
    #
    NumberInstances = auto()


def featurename(label: str | int | tuple[int, ...], f: FEATURES):
    if not isinstance(label, tuple):
        return str(label) + f"_{f.name}"
    else:
        labelstr = (str(i) for i in label)
        return "-".join(labelstr) + f"_{f.name}"


def compute_features(
    image_nii: NII,
    segmentation_nii: NII,
) -> dict[str, float]:
    extractor = featureextractor.RadiomicsFeatureExtractor("src/Params.yaml")

    image_nii = image_nii.reorient()
    segmentation_nii = segmentation_nii.reorient()

    crop = segmentation_nii.compute_crop(dist=4)
    segmentation_nii_c = segmentation_nii.apply_crop(crop)
    image_nii_c = image_nii.apply_crop(crop)

    image_arr = image_nii_c.get_array()

    # creature feature dict
    features: dict[str, float] = {}

    # Hand-made features
    seg_labels = segmentation_nii_c.unique()
    features["label_existence"] = seg_labels
    zoom = [round(i, 3) for i in image_nii_c.zoom]
    features["resolution"] = zoom
    voxel_resolution_factor = np.prod(zoom)

    binary_seg_nii = segmentation_nii_c.copy()
    binary_seg_nii[binary_seg_nii != 0] = 1

    coms = segmentation_nii_c.center_of_masses()
    coms["whole"] = binary_seg_nii.center_of_masses()[1]

    def feature_from_single_label(label: int | str, seg_label: NII):
        labelpresent = 1 in seg_label.unique()

        # VOLUME
        features[featurename(label, FEATURES.Volume)] = volume_if_present_otherwise_zero(seg_label) * voxel_resolution_factor
        if isinstance(label, int):
            features[featurename(label, FEATURES.VolumeRelationWhole)] = (
                features[featurename(label, FEATURES.Volume)] / features[featurename("whole", FEATURES.Volume)]
            )
        seg_arr = seg_label.get_seg_array()

        # CONNECTED COMPONENTS
        _, cc_n = np_utils.np_connected_components(seg_arr, label_ref=1)
        features[featurename(label, FEATURES.NumberInstances)] = cc_n

        # INTENSITIES
        intensities = image_arr[seg_arr == 1].flatten()
        if np.sum(intensities) != 0:
            features[featurename(label, FEATURES.IntensityMin)] = np.min(intensities)
            features[featurename(label, FEATURES.IntensityMax)] = np.max(intensities)
            features[featurename(label, FEATURES.IntensityAvg)] = np.average(intensities)
            features[featurename(label, FEATURES.IntensityStd)] = np.std(intensities)
            features[featurename(label, FEATURES.IntensityLIST)] = intensities.tolist()
            # features[featurename(label, FEATURES.SurfaceCoverage)] = seg_label.compute_surface_mask().volumes()[1]
        else:
            features[featurename(label, FEATURES.IntensityMin)] = np.nan
            features[featurename(label, FEATURES.IntensityMax)] = np.nan
            features[featurename(label, FEATURES.IntensityAvg)] = np.nan
            features[featurename(label, FEATURES.IntensityStd)] = np.nan
            features[featurename(label, FEATURES.IntensityLIST)] = []
            # features[featurename(label, FEATURES.SurfaceCoverage)] = 0

        # BOUNDEING BOX
        if labelpresent:
            bbox = np_utils.np_bbox_binary(seg_arr)
            features[featurename(label, FEATURES.SizeLength)] = (bbox[0].stop - bbox[0].start) * zoom[0]
            features[featurename(label, FEATURES.SizeHeight)] = (bbox[1].stop - bbox[1].start) * zoom[1]
            features[featurename(label, FEATURES.SizeWidth)] = (bbox[2].stop - bbox[2].start) * zoom[2]
        else:
            features[featurename(label, FEATURES.SizeLength)] = np.nan
            features[featurename(label, FEATURES.SizeHeight)] = np.nan
            features[featurename(label, FEATURES.SizeWidth)] = np.nan

    def features_from_two_labels(label_s: int, label_v: int, seg_label_s: NII, seg_label_v: NII):
        if label_v in coms and label_s in coms:
            seg_splusv = seg_label_s + seg_label_v
            seg_splusv[seg_splusv != 0] = 1
            com_splusv = seg_splusv.center_of_masses()[1]
            features[featurename((label_s, label_v), FEATURES.SpatialRelation)] = coms[label_v] - coms[label_s]
            features[featurename((label_s, label_v), FEATURES.SpatialRelationMerged)] = coms[label_v] - com_splusv
            #
            seg_thickness = seg_label_v.copy()
            seg_thickness[seg_thickness != 0] = 1
            seg_thickness[seg_label_s != 0] = 2
            features[featurename((label_s, label_v), FEATURES.SurfaceThicknessLIST)] = calc_label_thickness(
                seg_thickness,
                labelforthickness=1,
                labelforboundary=2,
            )
        else:
            features[featurename((label_s, label_v), FEATURES.SpatialRelation)] = np.nan
            features[featurename((label_s, label_v), FEATURES.SpatialRelationMerged)] = np.nan
        # features[featurename((label_s, label_v), FEATURES.SpatialRelationWhole)] = coms[label_v] - coms["whole"]
        if label_s in coms:
            # only label_s needs to be present for this feature
            s_surface = seg_label_s.compute_surface_mask(dilated_surface=True, connectivity=3)
            # s_surface.save("/DATA/NAS/ongoing_projects/hendrik/greifswald-segmentation/test_surface2.nii.gz")
            s_surface_n = s_surface.volumes()[1]
            # how many are soft tissue
            if label_v not in coms:
                features[featurename(label_v, FEATURES.SurfaceCoverage)] = 0
            else:
                # seg_label_v.save("/DATA/NAS/ongoing_projects/hendrik/greifswald-segmentation/test_seg_label_v.nii.gz")
                intersection = s_surface + seg_label_v

                # intersection.save("/DATA/NAS/ongoing_projects/hendrik/greifswald-segmentation/test_intersection2.nii.gz")
                intersection_volume = intersection.volumes()  # how many are both
                # how many are air
                intersection_volume = intersection_volume.get(2, 0.0)
                sc = intersection_volume / s_surface_n
                features[featurename(label_v, FEATURES.SurfaceCoverage)] = sc
                # print(f"Surface Coverage {label_s}-{label_v}: {s_surface_n:.4f}, {intersection_volume:.4f}, {sc:.4f}")
        elif label_v in coms:
            features[featurename(label_v, FEATURES.SurfaceCoverage)] = 1
        else:
            features[featurename(label_v, FEATURES.SurfaceCoverage)] = 0

    feature_from_single_label("whole", binary_seg_nii)
    feature_from_single_label("unionair", segmentation_nii_c.extract_label(AIRLABELS))
    feature_from_single_label("unionsoft", segmentation_nii_c.extract_label(SOFTLABELS))

    # loop over normal labels
    for rl in AIRLABELS:
        vrl = rl + 8

        rl_msk = segmentation_nii_c.extract_label(rl)
        vrl_msk = segmentation_nii_c.extract_label(vrl)
        feature_from_single_label(rl, rl_msk)
        feature_from_single_label(vrl, vrl_msk)
        features_from_two_labels(rl, vrl, rl_msk, vrl_msk)

        # pyradiomics features
        # air
        try:
            result_1 = extractor.execute(nii_to_sitk(image_nii_c), nii_to_sitk(rl_msk))
            radiomic_features = {key: float(v) for key, v in result_1.items() if key.startswith("original_")}
        except ValueError as e:
            radiomic_features = {}
            print(f"Error computing pyradiomics features for structure {vrl}: {e}")
            # raise e
        for k, v in radiomic_features.items():
            features[f"{rl}_{k}"] = v
        # soft tissue
        try:
            result_1 = extractor.execute(nii_to_sitk(image_nii_c), nii_to_sitk(vrl_msk))
            radiomic_features = {key: float(v) for key, v in result_1.items() if key.startswith("original_")}
        except ValueError as e:
            radiomic_features = {}
            print(f"Error computing pyradiomics features for structure {vrl}: {e}")
            # raise e
        for k, v in radiomic_features.items():
            features[f"{vrl}_{k}"] = v

    return features


def volume_if_present_otherwise_zero(msk: NII):
    v = msk.volumes()
    return v.get(1, 0.0)
