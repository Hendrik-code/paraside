from pathlib import Path

import numpy as np
from spineps.seg_enums import OutputType
from spineps.seg_model import Segmentation_Model
from TPTBox import NII


def assign_ccs_based_on_mask_neighbors(cc_msk: NII, label_msk: NII) -> NII:
    # delete voxesl that are already labeled
    cc_msk[label_msk != 0] = 0
    # for each cc in cc_msk
    for l in cc_msk.unique():
        cc_l = cc_msk.extract_label(l)
        cc_l_dil = cc_l.dilate_msk(
            n_pixel=1,
            labels=1,
            connectivity=3,
            verbose=False,
        )

        mult = (label_msk * cc_l_dil).get_seg_array()
        labels, count = np.unique(mult, return_counts=True)
        if 0 in labels:
            labels = labels[1:]
            count = count[1:]
        newlabel = labels[np.argmax(count)]
        label_msk[cc_l != 0] = newlabel
    return label_msk


def segment_nose(
    image_nii: NII,
    model: Segmentation_Model,
    step_size: float = 0.5,
    seg_out: Path | None = None,
    proc_fillholes: bool = True,
    proc_remove_label_below_volume_threshold: int = 0,
    proc_remove_but_largest_cc: bool = True,
) -> NII:
    ori = image_nii.orientation
    image_nii = image_nii.reorient()

    seg_nii: NII = model.segment_scan(image_nii, pad_size=0, step_size=step_size)[OutputType.seg]  # type: ignore

    if proc_fillholes:
        # Fill holes per label
        seg_nii.fill_holes_()
        # Fill holes by neighbor
        seg_nii_bin = seg_nii.copy()
        seg_nii_bin[seg_nii_bin != 0] = 1
        seg_nii_bin_fh = seg_nii_bin.fill_holes()
        if seg_nii_bin_fh.volumes()[1] > seg_nii_bin.volumes()[1]:
            # go for each fill holed CC
            seg_nii_bin_fh[seg_nii_bin == 1] = 0
            seg_nii = assign_ccs_based_on_mask_neighbors(
                seg_nii_bin_fh.get_connected_components(connectivity=1),
                label_msk=seg_nii,
            )

    if proc_remove_label_below_volume_threshold > 0:
        for l, lv in seg_nii.volumes().items():
            if lv <= proc_remove_label_below_volume_threshold:
                seg_nii[seg_nii == l] = 0

    if proc_remove_but_largest_cc:
        seg_nii_bin = seg_nii.copy()
        seg_nii_bin[seg_nii_bin != 0] = 1
        seg_nii_bin_filtered = seg_nii_bin.filter_connected_components(labels=1, min_volume=50, connectivity=3)
        seg_nii[seg_nii_bin_filtered == 0] = 0

    image_nii.assert_affine(other=seg_nii)

    seg_nii.reorient_(ori)
    if seg_out is not None:
        seg_nii.save(seg_out)
    return seg_nii
