from pathlib import Path

import numpy as np
from TPTBox import NII
from tqdm import tqdm

# (7/8, 15/16)
ETHMOIDLABELS = [7, 8, 15, 16]

ETHMOIDBACK_OFFSET = 100
ETHMOIDSIDES_OFFSET = 200


def split_ethmoid(segmentation_nii: NII):
    """Split the ethmoid segmentation into posterior and anterior parts, and left/right sides.

    Args:
        segmentation_nii (NII): NII object containing the segmentation with a split ethmoid

    Returns:
        NII: NII object with the split ethmoid segmentation
    """
    seg_nii_new = segmentation_nii.copy()
    ori = seg_nii_new.orientation
    seg_nii_new.reorient_()

    seg_nii_eth: NII = seg_nii_new.extract_label(ETHMOIDLABELS, keep_label=True)
    crop = seg_nii_eth.compute_crop()
    total_volume = sum(seg_nii_eth.volumes().values())
    cumulative_volume = 0
    cutoff_slice: int | None = None

    # iterate over posterior direction
    for post in range(crop[0].stop, crop[0].start, -1):
        cumulative_volume += np.count_nonzero(seg_nii_eth[post, :, :])
        if cumulative_volume >= total_volume / 3:
            cutoff_slice = int(post)
            break
    if cutoff_slice is not None:
        # split ethmoid into posterior and anterior
        seg_nii_eth_back: NII = seg_nii_eth.copy()
        seg_nii_eth_back[cutoff_slice:, :, :] += ETHMOIDBACK_OFFSET
        seg_nii_eth_back[seg_nii_eth_back == ETHMOIDBACK_OFFSET] = 0
    # left/right sides
    seg_nii_front = seg_nii_eth_back.extract_label(ETHMOIDLABELS, keep_label=True)
    for lp in [(7, 15), (8, 16)]:
        seg_nii_fs = seg_nii_front.extract_label(lp, keep_label=True)
        com = (seg_nii_fs > 0).center_of_masses()[1]
        # left
        com_l = int(np.rint(com[2]))
        seg_nii_fs[:, :, com_l:] += ETHMOIDSIDES_OFFSET
        seg_nii_fs[seg_nii_fs == ETHMOIDSIDES_OFFSET] = 0
        seg_nii_front[seg_nii_fs > ETHMOIDSIDES_OFFSET] = seg_nii_fs[seg_nii_fs > ETHMOIDSIDES_OFFSET]

    seg_nii_eth_back[seg_nii_front > ETHMOIDSIDES_OFFSET] = seg_nii_front[seg_nii_front > ETHMOIDSIDES_OFFSET]
    # create new segmentation
    seg_nii_new[seg_nii_eth_back.get_seg_array() > 0] = seg_nii_eth_back[seg_nii_eth_back.get_seg_array() > 0]
    seg_nii_new.map_labels_({208: 8, 8: 208, 216: 16, 16: 216}, verbose=False)

    seg_nii_new.reorient_(ori)
    return seg_nii_new
