import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve, gaussian_filter, sobel
from TPTBox import NII


def calc_label_thickness(
    seg: NII,
    labelforthickness: int = 1,
    labelforboundary: int = 2,
    use_parallelization: bool = False,
) -> list[float]:
    surface_distances = []

    seg2 = seg.extract_label(
        (labelforthickness, labelforboundary),
        keep_label=True,
    )
    crop = seg2.compute_crop(dist=2)
    seg2.apply_crop_(crop)

    seg_largest = seg2.filter_connected_components(labels=[labelforthickness], max_count_component=2, keep_label=True, connectivity=1)
    boundary_coords, normals = compute_boundary_normals(seg_largest.get_seg_array(), labelforboundary, labelforthickness)

    # st_nii = seg_largest == labelforthickness
    # for coord in boundary_coords:
    #    st_nii[tuple(coord)] = 1
    st_nii = (seg_largest == labelforthickness).copy()
    z, y, x = boundary_coords.T  # unpack columns
    st_nii[z, y, x] = 1

    if not use_parallelization:
        for coord, normal in zip(boundary_coords, normals):
            dist = compute_distance_for_point(coord, normal, st_nii)
            surface_distances.append(dist)
    else:
        surface_distances = Parallel(n_jobs=-1)(  # or "processes"
            delayed(compute_distance_for_point)(coord, normal, st_nii) for coord, normal in zip(boundary_coords, normals)
        )
    return list(surface_distances)


def compute_distance_for_point(coord, normal, st_nii):
    normal = -1 * normal  # Invert normal to point outward
    coord = tuple(coord)
    end_point = raymarch_until_background(st_nii, start_coord=coord, direction_vector=normal)
    if end_point is None:
        return np.nan
    diff = np.subtract(end_point, np.asarray(coord))
    dist_n = np.linalg.norm(diff)
    return dist_n


def compute_boundary_normals(seg, label1, label2):
    """
    seg: 3D numpy array, where 0 = background, 1 = class 1, 2 = class 2
    Returns:
        boundary_coords: (N, 3) array of voxel coordinates (z, y, x)
        normals: (N, 3) array of normal vectors at those coordinates
    """
    class1 = seg == label1
    class2 = seg == label2

    # Define 3D 6-connectivity kernel
    kernel = np.zeros((3, 3, 3), dtype=int)
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1

    # Convolve class 1 with kernel to count neighbors
    neighbors_of_class1 = convolve(class1.astype(int), kernel, mode="constant")

    # Find class 1 voxels that are adjacent to class 2
    touching_mask = np.logical_and(class1, convolve(class2.astype(int), kernel, mode="constant") > 0)

    # Optional: smooth class 1 mask to compute normals
    smoothed = gaussian_filter(class1.astype(float), sigma=1)

    # Compute gradients
    gx = sobel(smoothed, axis=2)  # x
    gy = sobel(smoothed, axis=1)  # y
    gz = sobel(smoothed, axis=0)  # z

    # Stack gradients into vector field
    grad_field = np.stack([gz, gy, gx], axis=-1)  # shape (Z, Y, X, 3)

    # Normalize
    norm = np.linalg.norm(grad_field, axis=-1, keepdims=True) + 1e-8
    normal_field = grad_field / norm

    # Get boundary coordinates
    boundary_coords = np.argwhere(touching_mask)
    normals = normal_field[touching_mask]

    return boundary_coords, normals


def raymarch_until_background(
    seg: NII,
    start_coord: np.ndarray,
    direction_vector: np.ndarray,
    step_size: float | None = None,
    max_steps: int | None = 1000,
    max_distance: float | None = None,
):
    """
    Raymarches from start_coord in the direction of direction_vector until it hits the background (0).
    Returns the coordinate where it hits the background or None if it doesn't hit within max_steps.
    """
    assert max_steps is not None or max_distance is not None, "At least one of max_steps or max_distance must be specified."

    pos = np.array(start_coord, dtype=np.float32)
    direction_vector = direction_vector / (np.linalg.norm(direction_vector) + 1e-10)
    if step_size is None:
        step_size = min(seg.zoom) / 16

    step_vector = direction_vector * step_size

    shape = seg.shape
    seg_arr = seg.get_array().astype(np.float32)
    interpolator = RegularGridInterpolator(
        [np.arange(s, dtype=np.float32) for s in shape],
        seg_arr,
        bounds_error=False,
        fill_value=0.0,
    )

    max_steps = max_steps if max_steps is not None else 1e8

    # def is_inside(coords):
    #    if any(i < 0 for i in coords):
    #        return 0
    #    if any(coords[i] > seg.shape[i] - 1 for i in range(len(coords))):
    #        return 0
    #    # Evaluate the mask value at the interpolated coordinates
    #    mask_value = interpolator(coords)
    #    return mask_value > 0.5

    for step in range(max_steps):
        if max_distance is not None:
            dist = np.linalg.norm(pos - start_coord)
            if dist >= max_distance:
                return None

        # Stop if we're outside volume
        if np.any(pos < 0) or np.any(pos >= shape):
            return pos

        val = interpolator(pos)
        if val < 0.5:
            return pos

        pos += step_vector

    return None
