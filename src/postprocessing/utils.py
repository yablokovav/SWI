from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.fftpack import dct, idct
from scipy.optimize import fminbound


def define_projection(coord):
    """
    Automatically determine the dominant projection direction ('xz' or 'yz')
    based on the spatial extent of coordinates.

    Compares the range (peak-to-peak) of X and Y coordinates and selects the
    direction with the greater length as the main axis of interpolation.

    Args:
        coord (np.ndarray): 2D array of shape (N, 2) containing X and Y coordinates.

    Returns:
        str: 'xz' if X direction is dominant, otherwise 'yz'.
    """
    length_line_x, length_line_y = np.ptp(coord, axis=0)
    return "xz" if length_line_x >= length_line_y else "yz"


def group_files_by_basename(data_dir: Path, suffix: str = ".npz") -> Dict[str, List[Path]]:
    """
    Groups files in a directory by their basename (filename without extension or numerical index).

    This function iterates through files in the specified directory, extracts the basename
    from each filename (removing the extension and any trailing numerical indices), and
    groups the files based on these basenames.

    Args:
        data_dir (Path): The directory to search for files.
        suffix (str, optional): The file extension to filter for (e.g., ".npz"). Defaults to ".npz".

    Returns:
        Dict[str, List[Path]]: A dictionary where:
            - Keys are the unique basenames (without extension or numerical index).
            - Values are lists of `Path` objects representing the full paths to files
              with that basename.
    """

    files_by_basename: Dict[str, List[Path]] = {}  # Initialize the dictionary

    for file_path in data_dir.glob(f"*{suffix}"):  # Use Path.glob for efficient file finding

        basename = file_path.stem  # Get filename without extension


        # Remove trailing digits from the basename to group files with indices together
        # This assumes indices are at the end of the filename before the extension
        base_name_without_index = basename.split(".")[0]
        files_by_basename.setdefault(base_name_without_index, []).append(file_path)


    return files_by_basename

def read_curves(models, h_max, mape_thr):
    """
    Prepare depth, velocity, and metadata from npz fiiles into arrays from a list of ModelVCR objects.

    This function performs the following operations:
        1. Determines the maximum number of layers among all models.
        2. Pads models with fewer layers to match the maximum layer count using
           dummy layers (thickness = 0.1 m, velocity = last valid velocity).
        3. Constructs consistent depth and velocity arrays.
        4. Extracts coordinates, elevation (relief), and model errors.
        5. Filters out velocity models with MAPE (mean absolute percentage error)
           above the threshold, replacing them with NaNs.

    Args:
        models (List[ModelVCR]): List of velocity models to process.
        h_max (float): Maximum depth limit to pad the depth profile.
        mape_thr (float): Error threshold (MAPE); models with higher error will be skipped.

    Returns:
        Tuple[np.ndarray]: Tuple of:
            - depth: (N_models, N_layers+2) array with depth values.
            - velocity: (N_models, N_layers+1) array with shear velocity values (padded).
            - coord: (N_models, 2) array with CMP X and Y coordinates.
            - relief: (N_models,) array with surface elevation for each model.
            - error: (N_models,) array with model MAPE error values.
    """

    # определение максимального числа слоев для всех моделей
    n_bounders = 0
    for model in models:
        if len(model.thickness) > n_bounders:
            n_bounders = len(model.thickness)

    # для моделей с числом слоев меньше максимального добавляем слои с мощностью 0.1 м и со скоростью предыдущего слоя
    for model in models:
        if len(model.thickness) < n_bounders:
            model.thickness = np.pad(model.thickness, (0, n_bounders - len(model.thickness)), mode='constant',
                                     constant_values=0.1)
            model.velocity_shear = np.pad(model.velocity_shear, (0, n_bounders + 1 - len(model.velocity_shear)),
                                          mode='edge')
    # подготовка матриц глубин и скоростей, координат привязки моделей
    depth, velocity, coord, relief = [], [], [], []
    error = []
    for model in models:
        depth.append(np.hstack(([0], np.cumsum(model.thickness), [h_max])))
        coord.append([model.cmp_x, model.cmp_y])
        relief.append(model.relief)
        error.append(model.error_dc)
        if model.error_dc <= mape_thr:
            velocity.append(model.velocity_shear)
        else:
            velocity.append([np.nan] * len(model.velocity_shear))
    velocity = np.array(velocity)
    velocity = np.hstack((velocity, velocity[:, -1].reshape(-1, 1)))
    return np.array(depth), velocity, np.array(coord), np.array(relief), np.array(error)

def robust_smooth_2d(y, s: Optional[float], robust: True):
    """
    Smooth a 2D numpy array with optional robust outlier rejection and
    interpolation of missing values (NaNs).

    This function performs smoothing of 2D data using Discrete Cosine Transform (DCT)
    and penalized the least squares, optionally applying robust weights to reduce
    the influence of outliers. Missing values in the input array (`np.nan`)
    are automatically interpolated during smoothing.

    Args:
        y : np.ndarray
            A 2D NumPy array to be smoothed. Missing values should be marked as `np.nan`.
        s : float or None
            Optional smoothing factor. If `None`, it will be automatically computed using
            Generalized Cross Validation (GCV). A larger `s` value results in more smoothing.
        robust : bool
            If `True`, the algorithm performs robust smoothing by iteratively reducing
            the influence of outliers. If `False`, standard smoothing is applied.
    Returns:
        z : np.ndarray
            The smoothed 2D array, with interpolated values replacing `np.nan` and outliers reduced
            (if `robust=True`).

    Notes
    -----
    - Uses DCT-based fast smoothing for performance and accuracy.
    - Up to 3 robust iterations are performed to stabilize the result if `robust=True`.
    - The smoothing factor `s` is optimized via GCV when not provided.
    """

    if s is None:
        auto_s = True
    else:
        auto_s = False


    size_y = np.asarray(y.shape)
    num_elements = np.prod(size_y)
    not_finite = np.isnan(y)
    is_finite = np.logical_not(not_finite)
    num_finite = np.sum(is_finite)

    # Create the Lambda tensor, which contains the eingenvalues of the
    # difference matrix used in the penalized the least squares process. We assume
    # equal spacing in horizontal and vertical here.
    lmbda = np.zeros(size_y)
    for i in range(y.ndim):
        size_0 = np.ones((y.ndim,), dtype=int)
        size_0[i] = size_y[i]
        lmbda += 2 - 2 * np.cos(np.pi * (np.reshape(np.arange(0, size_y[i]), size_0)) / size_y[i])

    # Upper and lower bound for the smoothness parameter
    # The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    # if h is close to 1, while over-smoothing appears when h is near 0. Upper
    # and lower bounds for h are given to avoid under- or over-smoothing.
    tensor_rank = sum(size_y != 1)  # tensor rank of the y-array
    h_min = 1e-6
    h_max = 0.99
    s_min_bound = (((1 + np.sqrt(1 + 8 * h_max ** (2 / tensor_rank))) / 4 / h_max ** (2 / tensor_rank)) ** 2 - 1) / 16
    s_max_bound = (((1 + np.sqrt(1 + 8 * h_min ** (2 / tensor_rank))) / 4 / h_min ** (2 / tensor_rank)) ** 2 - 1) / 16

    # initialize stuff before iterating
    weights = np.ones(size_y)
    weights[not_finite] = 0
    weights_total = weights
    z = initial_guess(y, not_finite)
    z0 = z
    y[not_finite] = 0
    tolerance = 1
    num_robust_iterations = 1
    num_iterations = 0
    relaxation_factor = 1.75
    robust_iterate = True

    # iterative process
    while robust_iterate:
        while tolerance > 1e-3 and num_iterations < 100:
            num_iterations += 1
            dct_y = dct(dct(weights_total * (y - z) + z, norm='ortho', type=2, axis=0), norm='ortho', type=2, axis=1)

            # The generalized cross-validation (GCV) method is used to compute
            # the smoothing parameter S. Because this process is time-consuming,
            # it is performed from time to time (when the number of iterations
            # is a power of 2).
            if auto_s and not np.log2(num_iterations) % 1:
                p = fminbound(
                    gcv,
                    np.log10(s_min_bound),
                    np.log10(s_max_bound),
                    args=(lmbda, dct_y, weights_total, is_finite, y, num_finite, num_elements),
                    xtol=0.1,
                    full_output=False)
                s = 10 ** p

            Gamma = 1 / (1 + s * lmbda ** 2)
            z = relaxation_factor * idct(idct(Gamma * dct_y, norm='ortho', type=2, axis=1), norm='ortho', type=2,
                                         axis=0) + (1 - relaxation_factor) * z
            tolerance = np.linalg.norm(z0 - z) / np.linalg.norm(z)
            z0 = z  # re-initialize

        if robust:
            # average levereage
            h = 1
            for k in range(tensor_rank):
                h0 = np.sqrt(1 + 16 * s)
                h0 = np.sqrt(1 + h0) / np.sqrt(2) / h0
            h = h * h0
            # take robust weights into account
            weights_total = weights * robust_weights(y, z, is_finite, h)
            # re-initialize for another iterative weighted process
            tolerance = 1
            num_iterations = 0
            num_robust_iterations += 1
            robust_iterate = num_robust_iterations < 4  # 3 robust iterations are enough
        else:
            robust_iterate = False

    return z


def robust_weights(y, z, is_finite, h):
    """Generate bi-square weights for robust smoothing (outlier rejection)."""
    residuals = y - z
    median_abs_deviation = np.median(np.fabs(residuals[is_finite] - np.median(residuals[is_finite])))
    studentized_residuals = np.abs(residuals / (1.4826 * median_abs_deviation) / np.sqrt(1 - h))
    # the weighting can be tuned by modifying the 4.685 value (make it smaller
    # for more aggressive outlier detection)
    bisquare_weights = ((1 - (studentized_residuals / 4.685) ** 2) ** 2) * ((studentized_residuals / 4.685) < 1)
    bisquare_weights[np.isnan(bisquare_weights)] = 0
    return bisquare_weights


def gcv(p, lmbda, dct_y, weights_total, is_finite, y, num_finite, num_elements):
    """Generalized Cross Validation for determining the smoothing factor."""
    s = 10 ** p
    gamma = 1 / (1 + s * lmbda ** 2)
    y_hat = idct(idct(gamma * dct_y, norm='ortho', type=2, axis=1), norm='ortho', type=2, axis=0)
    rss = np.linalg.norm(np.sqrt(weights_total[is_finite]) * (y[is_finite] - y_hat[is_finite])) ** 2
    trace_h = np.sum(gamma)
    gcv_score = rss / num_finite / (1 - trace_h / num_elements) ** 2
    return gcv_score


def initial_guess(y, not_finite):
    """Generate an initial estimate of the smooth surface with missing values
    interpolated.
    """
    # Nearest neighbor interpolation of missing values. This can leave visible
    # artifacts resulting from the nearest neighbor interpolation that is used.
    if not_finite.any():
        indices = ndimage.distance_transform_edt(not_finite, return_indices=True)[1]
        z = y[indices[0], indices[1]]
    else:
        z = y
    # coarse smoothing using a fraction of the DCT coefficients
    z = dct(dct(z, norm='ortho', type=2, axis=0), norm='ortho', type=2, axis=1)
    zero_start = np.ceil(np.array(z.shape) / 10).astype(int)
    z[zero_start[0]:, :] = 0
    z[:, zero_start[1]:] = 0
    z = idct(idct(z, norm='ortho', type=2, axis=1), norm='ortho', type=2, axis=0)
    return z


def average_models_in_bin(
    coordinates: np.ndarray,
    error: np.ndarray,
    vel_model: np.ndarray,
    elevation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Averages velocity models within bins, weighting models based on their error.

    This function identifies unique coordinate locations, and for each location,
    averages the corresponding velocity models, weighting them linearly inversely proportional to error.
    NaN values in velocity models are handled gracefully.

    Args:
        coordinates: (N, 2) array of (x, y) coordinates for each velocity model.
        error: (N), array of error values associated with each velocity model.
        vel_model: (M, N) array of velocity models, where M is the depth dimension and N is the number of models.
        elevation: (N), array of elevation values associated with each velocity model.

    Returns:
        A tuple containing:
            - unique_coordinates: (K, 2) array of unique (x, y) coordinates.
            - averaged_velocities: (M, K) array of averaged velocity models for each unique coordinate.
            - averaged_elevations: (K), array of averaged elevation values for each unique coordinate.
    """

    unique_coords, unique_indices, counts = np.unique(
        coordinates, axis=0, return_index=True, return_counts=True
    )

    averaged_velocities = []
    averaged_elevations = []

    for index, count in zip(unique_indices, counts):
        if count > 1:  # Multiple models in this bin
            # Sort models by error within the bin
            indices_in_bin = np.argsort(error[index : index + count])
            velocities_in_bin = vel_model[:, index + indices_in_bin]

            # Create weights: linear from 1 to 0 based on error rank
            valid_mask = ~np.isnan(velocities_in_bin[0])  # Mask out NaNs
            velocities_valid = velocities_in_bin[:, valid_mask]
            num_valid = valid_mask.sum()

            if num_valid > 0:
                weights = np.linspace(1, 0, num_valid)
                # Weighted average, handling NaNs
                averaged_velocity = np.average(
                    velocities_valid, weights=weights, axis=1
                )
            else:
                # All models are NaN, use a first model
                averaged_velocity = vel_model[:, index].copy()
        else:  # Only one model in this bin
            averaged_velocity = vel_model[:, index].copy()  # Make a copy to avoid potential modification of the original

        averaged_velocities.append(averaged_velocity)
        averaged_elevations.append(np.mean(elevation[index : index + count]))

    return (
        unique_coords,
        np.array(averaged_velocities).T,
        np.array(averaged_elevations),
    )