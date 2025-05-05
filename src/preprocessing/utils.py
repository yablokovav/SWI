from collections.abc import Generator
from typing import Optional, Any

import numpy as np
from  numba import  njit
from scipy.signal import detrend
from sklearn.cluster import KMeans

from src.files_processor.readers import read_segy
from src.files_processor.savers import save_segy
from src.spectral_analysis.utils import spectral_processing
from src.preprocessing.config import name_headers
from src import *
from src.preprocessing.regularization import global_spitz_interpolation

RANDOM_STATE = 42  # Define a constant for random state

def define_spatial_step(headers: np.ndarray) -> float:
    """
    Defines the spatial step (distance between traces) based on the header data.

    This function calculates the average spatial step by computing the mean of the
    differences between consecutive offset values in the header data.

    Args:
        headers (np.ndarray): A NumPy array containing header data, where one of
            the rows represents the offset for each trace.

    Returns:
        float: The average spatial step as a float (converted to int32).
    """
    return np.int32(np.round(np.abs(np.diff(headers[HEADER_OFFSET_IND])).mean()))


def define_direction(headers: np.ndarray, header_ind_1: int, header_ind_2: int) -> tuple[int, int]:
    """
    Determines the primary and secondary coordinate directions based on the range of values in header data.

    This function compares the peak-to-peak (ptp) range of values in two header
    fields (specified by `header_ind_1` and `header_ind_2`) and assigns primary
    and secondary coordinate direction indices based on which field has a larger range.

    Args:
        headers (np.ndarray): A NumPy array containing header data.
        header_ind_1 (int): Index of the first header field.
        header_ind_2 (int): Index of the second header field.

    Returns:
        Tuple[int, int]: A tuple containing the primary and secondary coordinate direction indices.
                         The values are chosen from HEADER_SOU_X_IND, HEADER_REC_X_IND,
                         HEADER_SOU_Y_IND, and HEADER_REC_Y_IND. Specifically, if
                         `np.ptp(headers[header_ind_1]) > np.ptp(headers[header_ind_2])`,
                         then the return is (HEADER_SOU_X_IND, HEADER_REC_X_IND);
                         otherwise, the return is (HEADER_SOU_Y_IND, HEADER_REC_Y_IND).
    """
    primary_key_idx, secondary_key_idx = (
        (HEADER_SOU_X_IND, HEADER_REC_X_IND)
        if np.ptp(headers[header_ind_1]) > np.ptp(headers[header_ind_2])
        else (HEADER_SOU_Y_IND, HEADER_REC_Y_IND)
    )
    return primary_key_idx, secondary_key_idx

def _calculate_cmp(primary_key_idx: int, curr_headers: np.ndarray, base: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates CMP (Common MidPoint) coordinates based on source coordinates and a base offset.

    This function computes the CMP coordinates (x_cmp, y_cmp) from the source coordinates
    (SOU_X, SOU_Y) stored in the `curr_headers` array. It applies a `base` offset to either
    the X or Y coordinate, depending on the `primary_key_idx`.

    Args:
        primary_key_idx (int): The index of the primary coordinate (either SOU_X or SOU_Y)
                                that defines the direction of the CMP line.  If
                                equal to `HEADER_SOU_X_IND`, the CMP line is assumed to
                                run primarily along the X-axis, and the `base` offset is
                                added to the X coordinate. Otherwise (if equal to
                                `HEADER_SOU_Y_IND`), the CMP line is assumed to run
                                primarily along the Y-axis, and the `base` offset is added
                                to the Y coordinate.
        curr_headers (np.ndarray): A 2D NumPy array representing the trace headers for
                                the current segment.  Shape: (number of header fields, number of traces).
                                Assumes that the header fields are organized column-wise, with each
                                column representing a trace.
        base (float): The base offset value to add to either the X or Y coordinate.
                      This value represents the distance along the CMP line. It is cast to int before use.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - x_cmp (np.ndarray): The calculated X CMP coordinates.
                                  Shape: (number of traces in curr_headers).
            - y_cmp (np.ndarray): The calculated Y CMP coordinates.
                                  Shape: (number of traces in curr_headers).
    """

    # Calculate X CMP coordinate.  If the primary key is SOU_X, add the base offset.
    x_cmp = curr_headers[HEADER_SOU_X_IND] + int(base) if primary_key_idx == HEADER_SOU_X_IND else curr_headers[HEADER_SOU_X_IND]

    # Calculate Y CMP coordinate. If the primary key is *not* SOU_X (implying it's SOU_Y), add the base offset.
    y_cmp = curr_headers[HEADER_SOU_Y_IND] if primary_key_idx == HEADER_SOU_X_IND else curr_headers[HEADER_SOU_Y_IND] + int(base)

    # Return the calculated CMP coordinates
    return x_cmp, y_cmp


def get_part_data(
    primary_key_idx: int,
    seism: np.ndarray,
    headers: np.ndarray,
    offset: np.ndarray,
    offset_min: float,
    offset_max: float
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Extracts a segment of seismic data based on offset values, applies detrending,
    and potentially calculates new CMP coordinates.

    This function filters seismic data based on an offset range (`offset_min` to `offset_max`).
    If the offset values are considered "large" (exceeding `offset_max` in absolute value),
    it identifies the valid traces within the specified offset range, calculates new CMP
    coordinates (using `_calculate_cmp`), and applies a detrending operation to the seismic
    traces before returning the selected data segment.

    Args:
        primary_key_idx (int): The index of the primary coordinate (either SOU_X or SOU_Y)
                                used for calculating CMP coordinates. This is passed to `_calculate_cmp`.
        seism (np.ndarray): A 2D NumPy array representing the seismic data (seismogram).
                              Shape: (number of samples, number of traces).
        headers (np.ndarray): A 2D NumPy array representing the trace headers.
                              Shape: (number of header fields, number of traces).  Assumed to
                              contain SOU_X, SOU_Y, and potentially CDP_X, CDP_Y coordinates.
        offset (np.ndarray): A 1D NumPy array representing the offset values for each trace.
                              Shape: (number of traces).
        offset_min (float): The minimum acceptable offset value.
        offset_max (float): The maximum acceptable offset value.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: A tuple containing the detrended seismic
        segment and corresponding headers, only if valid traces are found within the
        specified offset range and offset values are considered "large". Returns None
        if the offset values are not considered large (and no processing is required).
        If no valid traces are found, raises a ValueError.

    Raises:
        ValueError: If the offset values are considered "large" but no valid traces
        are found within the specified `offset_min` and `offset_max`. This suggests
        a problem with the data or sorting.
    """

    # Check if any offset values are considered "large"
    has_large_values = np.any(offset >= np.abs(offset_max))

    # If there are large offset values, proceed with filtering and processing
    if has_large_values:
        # Find the indexes of traces with offsets within the valid range
        valid_indexes = np.where((offset >= np.abs(offset_min)) & (offset <= np.abs(offset_max)))[0]

        # Calculate a CMP value for placing the source at zero offset.  This assumes that the
        # CMP position will vary depending on the offset range. The value `cmp_for_source_in_zero`
        # is the midpoint of the range.
        cmp_for_source_in_zero = (offset_max-offset_min) / 2 + offset_min

        # Calculate new CDP_X and CDP_Y coordinates based on the calculated CMP value
        # Applying new CDP coordinates to headers.
        headers[HEADER_CDP_X_IND, :], headers[HEADER_CDP_Y_IND, :] = (
            _calculate_cmp(primary_key_idx, headers, cmp_for_source_in_zero)
        )

        # Ensure header contains information about offset
        headers[HEADER_OFFSET_IND, :] = offset

        # Check if any valid traces were found within the offset range
        if len(valid_indexes):
            # Detrend the seismic data for the valid traces and return the detrended data and headers

            # Call detrend function and cut seism and headers by valid indexes
            return detrend(seism[:, valid_indexes], axis=0), headers[:, valid_indexes]

        # If no valid traces were found, raise a ValueError indicating a potential problem
        # with the data or sorting
        raise ValueError("Check the sorting, maybe you need it.")

    # If there are no large offset values, return None indicating no processing is needed
    return None, None



def data_partition(
    traces: np.ndarray, headers: np.ndarray, sort_3d_order: str = "",
) -> Generator[tuple[Any, Any], None, None]:

    """
        Partitions seismic data (traces and headers) into segments based on coordinate equality.

        This function takes seismic traces and their corresponding headers and splits them
        into segments where the coordinates (either CDP or source) are the same.  It yields
        a tuple of (seismogram segment, header segment) for each unique coordinate.

        Args:
            traces (np.ndarray): A 2D NumPy array representing the seismic traces.
                                   Shape: (number of samples, number of traces).
            headers (np.ndarray): A 2D NumPy array representing the trace headers.
                                   Shape: (number of header fields, number of traces).
            sort_3d_order (str, optional): Specifies the coordinate system to use for partitioning.
                                           If "cdp", uses CDP coordinates (CDP_X, CDP_Y).
                                           Otherwise, uses source coordinates (SOU_X, SOU_Y).
                                           Defaults to "".

        Yields:
            Generator[Tuple[Optional[np.ndarray], Optional[np.ndarray]], None, None]:
                A generator that yields tuples of:
                - curr_seismogram (np.ndarray, optional): A 2D NumPy array representing the
                  seismic traces for the current segment.  None if no traces found.
                - curr_headers (np.ndarray, optional): A 2D NumPy array representing the
                  headers for the current segment.  None if no headers found.
        """
    # Initialize the starting index for the current segment
    start = 0
    if sort_3d_order == "cdp":
        # Use CDP coordinates (CDP_X, CDP_Y) for partitioning
        keys_for_partition = headers[HEADER_CDP_IND]
    else:
        # Use source coordinates (SOU_X, SOU_Y) for partitioning
        keys_for_partition = headers[HEADER_FFID_IND]
    # Iterate through the coordinate keys

    # Old variant of dividing data by keys, may be faster, but need previously sorting data by specified headers
    # for key in range(length := len(keys_for_partition)):
    #     # Check if this is the last trace or if the coordinates change
    #     if (key == length - 1) or keys_for_partition[key] != keys_for_partition[key + 1]:
    #         # Extract the seismic traces and headers for the current segment
    #         curr_seismogram = traces[:, start: key + 1]
    #         curr_headers = headers[:, start: key + 1]
    #
    #         # Yield the current segment
    #         yield curr_seismogram, curr_headers
    #
    #         # Update the starting index for the next segment
    #         start = key + 1

    unique_keys_for_partition = np.unique(keys_for_partition)
    for key in unique_keys_for_partition:
        part_indexes = np.where(keys_for_partition == key)[0]
        curr_seismogram = traces[:, part_indexes]
        curr_headers = headers[:, part_indexes]
        yield curr_seismogram, curr_headers


def step_on_generator(config_parameters, gen, file_path):
    current_indexes = next(gen)

    traces, headers, dt = read_segy(
        file_path,
        name_headers,
        indexes4read=current_indexes,
        endian=config_parameters.endian,
        elevation_scaler=config_parameters.scaler_to_elevation,
        coordinates_scaler=config_parameters.scaler_to_coordinates,
    )

    return traces, headers, dt, file_path


def apply_spectral_processing(
        config_parameters,
        file_path: Path,
        seism_traces: np.ndarray,
        seism_header: np.ndarray,
        dt: float,
        flank_id: int = None,
        num_sector: Optional[int] = 0,
) -> tuple[bool, float, np.ndarray]:
    """
    Applies spectral analysis to a section of seismic data.

    This function applies spectral processing to a portion of seismic data
    (seism_traces, seism_header) based on the provided configuration parameters.
    It saves preprocessed data if `config_parameters.qc_preprocessing` is enabled,
    constructs a unique name for the data section, and then calls the
    `spectral_processing` function.

    Args:
       config_parameters (ConfigParameters): An object containing configuration
           parameters for preprocessing and spectral analysis.
       file_path (Path): The path to the seismic data file.
       seism_traces (np.ndarray): The seismic traces for the current section of data.
       seism_header (np.ndarray): The headers for the current section of data.
       dt (float): The time sampling interval.
       flank_id (Optional[int]): An optional identifier for the flank (left or right).
           Defaults to None.
       num_sector (Optional[int]): An optional sector number, used for 3D data.
           Defaults to 0.

    Returns:
       bool: The `valid_modes` value returned by the `spectral_processing` function, which indicate,
       that extraction of dispersion curves complete successful.
    """

    # Bining by unique offset and sorting (need for left side)
    _, indexes = np.unique(seism_header[HEADER_OFFSET_IND], return_index=True, ) # Find unique quantized offsets
    seism_traces, seism_header = seism_traces[:, indexes], seism_header[:, indexes] # Keep only unique offset traces

    if config_parameters.type_data == "2d":

        desired_step = np.mean(np.abs(np.diff(seism_header[HEADER_OFFSET_IND])))
        seism_traces, seism_header = global_spitz_interpolation(seism_traces, seism_header, desired_step / 2)


        spec_name = (f"{file_path.stem}."
                     f"{seism_header[HEADER_CDP_X_IND][0]}."
                     f"{seism_header[HEADER_CDP_Y_IND][0]}."
                     f"{flank_id}")

        if config_parameters.qc_preprocessing:
            save_path = config_parameters.save_dir_preprocessing[0] / f"{spec_name}{file_path.suffix}"
            save_segy(save_path, seism_traces, seism_header, name_headers, dt)

    else:
        if config_parameters.sort_3d_order == 'csp':

            spec_name = f"{file_path.stem}.{seism_header[HEADER_FFID_IND][0]}.{num_sector}"
        else:
            spec_name = (f"{file_path.stem}."
                         f"{seism_header[HEADER_CDP_IND][0]}")

        desired_step = np.mean(np.abs(np.diff(seism_header[HEADER_OFFSET_IND])))
        seism_traces, seism_header = global_spitz_interpolation(seism_traces, seism_header, desired_step)

        if config_parameters.qc_preprocessing:
            save_path = config_parameters.save_dir_preprocessing[0] / f"{spec_name}{file_path.suffix}"
            save_segy(save_path, seism_traces, seism_header, name_headers, dt)

    snr = get_snr(
        seism_traces,
        seism_header[HEADER_OFFSET_IND],
        config_parameters.vmin,
        config_parameters.vmax,
        dt,
    )

    if snr >= config_parameters.user_snr:
        valid_modes = spectral_processing(config_parameters,
            seism_traces, seism_header, dt, define_spatial_step(seism_header), spec_name  # Seismogram
        )
    else:
        valid_modes = True

    return valid_modes, snr, seism_header

@njit(fastmath=True)
def mutual_correlation_function(signal_1: np.ndarray, signal_2: np.ndarray) -> np.ndarray:
    """
    Calculates the maximum value of the mutual correlation function between two signals.
    The signals should have the same length.
    """
    m = len(signal_1)
    mcf_tmp = np.zeros(m)
    for k in range(m):
         for j in range(m - 1 - k):
            mcf_tmp[k] += signal_1[j] * signal_2[j + k]
    return np.max(mcf_tmp)

def get_snr(data, offsets, vmin, vmax, dt) -> float:
    """
    Calculates the signal-to-noise ratio (SNR) based on the maximum value of mutual correlation function.

    Args:
        data (np.ndarray): Seismic data (time x traces).
        offsets (np.ndarray): Offsets of the traces.
        vmin (float): Minimum velocity.
        vmax (float): Maximum velocity.
        dt (float): Time sampling interval.

    Returns:
        float: The calculated signal-to-noise ratio.
    """

    nt = data.shape[0]
    pick_min = np.minimum(np.int32(offsets/vmax/dt), np.zeros_like(offsets) + nt - 1)
    pick_max = np.minimum(np.int32(offsets/vmin/dt), np.zeros_like(offsets) + nt - 1)
    mask = pick_min != pick_max
    indexes = np.where(mask)[0]

    mcf = np.zeros(len(indexes) - 1)
    for i in range(len(indexes[:-1])):
        signal1 = data[pick_min[indexes[i]]:pick_max[indexes[i]], indexes[i]]
        signal2 = data[pick_min[indexes[i]]:pick_max[indexes[i]], indexes[i+1]]
        mcf[i] = (mutual_correlation_function(signal1, signal2) /
                  (np.sqrt(np.sum(signal1**2)) * np.sqrt(np.sum(signal2**2))))

    mkf_mean = np.mean(mcf)
    snr = mkf_mean/(1- mkf_mean)
    return round(snr, 2)




def find_points_in_square(center: np.ndarray, base: float, points: np.ndarray) -> list[int]:
    """
    Finds all points that fall within a square region.

    Args:
        center (tuple[float, float]): Coordinates of the center of the square (x, y).
        base (float): Half the length of the side of the square (distance from center to side).
        This is so `right` will be at `x_center + base`
        points (list[tuple[float, float]]): List of points [(x1, y1), (x2, y2), ...].

    Returns:
        list[int]: List of indexes of the points inside the square.
    """
    x_center, y_center = center

    # Calculate the boundaries of the square
    left = x_center - base
    right = x_center + base
    bottom = y_center - base
    top = y_center + base

    # Select the points that are inside the square and their indices
    indexes = [
        idx for idx, (x, y) in enumerate(points)
        if left <= x <= right and bottom <= y <= top
    ]
    return indexes


# обговоренная кластеризация по углам (можно также и по удалениям)
def transform_to_polar(rec_coord: np.ndarray, sou_coord: np.ndarray) -> np.ndarray:
    """
    Transforms receiver and source coordinates to polar coordinates (offset and angle).

    Args:
        rec_coord (np.ndarray): Receiver coordinates (x, y).
        sou_coord (np.ndarray): Source coordinates (x, y).

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing offset and angle arrays.
    """
    diff_coord = rec_coord - sou_coord
    alpha = np.arctan2(diff_coord[1], diff_coord[0]) * 180 / np.pi

    return alpha


# попытка обобщения с учетом еще и расстояний
def weighted_clustering(angles: np.ndarray,
                        distances: np.ndarray,
                        angle_weight: float = 0.7,
                        distance_weight: float = 0.5,
                        n_clusters: int = 3
                        ) -> np.ndarray:
    """
    Clusters data points based on angles and distances, considering specified weights.

    Args:
        angles (list[float]): List of angles (in radians or degrees).
        distances (list[float]): List of distances.
        angle_weight (float): Weight for angles.
        distance_weight (float): Weight for distances.
        n_clusters (int): Number of clusters to form.

    Returns:
        np.ndarray: Cluster labels assigned to each data point.
    """
    # Normalize angles and distances to the range [0, 1]
    angles_norm = np.array(angles) / np.abs(angles).max()
    distances_norm = np.array(distances) / np.max(distances)

    # Apply weights to the normalized values
    weighted_angles = angles_norm * angle_weight
    weighted_distances = distances_norm * distance_weight

    # Combine the weighted features into a single feature matrix
    features = np.column_stack((weighted_angles, weighted_distances))

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                    n_init='auto')  # Specify n_init explicitly, or suppress warning
    labels = kmeans.fit_predict(features)

    return labels


def seismogram_without_large_values(cut_seismic: np.ndarray,
                                    cut_headers: np.ndarray,
                                    offset_min: float,
                                    offset_max: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Filters seismic data and headers to remove traces with offsets outside the specified range.

    This function filters seismic data (`cut_seismic`) and corresponding headers
    (`cut_headers`) to keep only the traces whose offsets (found in
    `cut_headers[HEADER_OFFSET_IND]`) fall within the range [offset_min, offset_max].

    Args:
        cut_seismic (np.ndarray): NumPy array of seismic traces.
        cut_headers (np.ndarray): NumPy array of seismic headers, including
            offset values.
        offset_min (float): The minimum acceptable offset value.
        offset_max (float): The maximum acceptable offset value.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the filtered seismic
            traces and the filtered headers.  If no filtering was necessary
            (all offsets were already within the range), then the original
            `cut_seismic` and `cut_headers` are returned.
    """
    offsets = cut_headers[HEADER_OFFSET_IND]
    has_large_values = np.any(np.array(offsets) >= offset_max)
    if has_large_values:  # Verify to data
        valid_indexes = np.where(
            (np.array(offsets) >= offset_min) & (np.array(offsets) <= offset_max)
        )[0]  # create index where values are in range
        return cut_seismic[:, valid_indexes], cut_headers[:, valid_indexes]
    else:
        return cut_seismic, cut_headers



def base_filtration(
        curr_headers: np.ndarray,
        curr_seismograms: np.ndarray,
        base: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filters headers and seismograms based on their proximity to a source coordinate.

    Args:
        curr_headers (np.ndarray): Array of current headers.
        curr_seismograms (np.ndarray): Array of current seismograms.
        base (float): Half the length of the side of the square around the souce.

    Returns:
        tuple[np.ndarray, np.ndarray]: Filtered headers and seismograms.
    """
    rec_coord, sou_coord = (
        curr_headers[HEADER_REC_X_IND:HEADER_REC_Y_IND + 1],
        curr_headers[HEADER_SOU_X_IND:HEADER_SOU_Y_IND + 1]
    )
    indexes = find_points_in_square(sou_coord[:, 0], base, rec_coord.T)

    return curr_headers[:, indexes], curr_seismograms[:, indexes]


def mean_traces_with_equal_offsets(seism: np.ndarray, headers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Averages seismic traces with equal offset values.

    This function takes a seismic data array and its corresponding headers, identifies traces
    with the same offset values, and averages the seismic data for those traces. It returns
    a new seismic data array and a corresponding header array, where each trace represents
    the average of traces with a unique offset value.  The header for each averaged trace
    is taken from the first trace with that offset value.

    Args:
        seism (np.ndarray): A 2D NumPy array representing the seismic data (seismogram).
                              Shape: (number of samples, number of traces).
        headers (np.ndarray): A 2D NumPy array representing the trace headers.
                              Shape: (number of header fields, number of traces). Assumes that
                              the offset values are stored in the row specified by `HEADER_OFFSET_IND`.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the averaged seismic data and headers:
            - seism_averaged (np.ndarray): A 2D NumPy array containing the averaged seismic data.
              Shape: (number of samples, number of unique offset values).
            - headers_unique (np.ndarray): A 2D NumPy array containing the headers for the
              first trace with each unique offset value.  Shape: (number of header fields,
              number of unique offset values).

    """
    seism_no_unique: list[np.ndarray] = []  # Initialize a list to store the averaged seismograms
    headers_no_unique: list[np.ndarray] = []  # Initialize a list to store the corresponding headers

    for unique_offset in np.unique(headers[HEADER_OFFSET_IND]):
        # Create a boolean mask to identify traces with the current unique offset
        mask = (headers[HEADER_OFFSET_IND] == unique_offset)

        # Average the seismic traces that have the current offset value along axis=1
        trace = seism[:, mask].sum(axis=1)

        # Append the averaged trace to the list of averaged seismograms
        seism_no_unique.append(trace)

        # Append the headers of the first trace with the current offset to the headers list
        # Use [:, 0] to select the first trace's headers after applying the mask
        headers_no_unique.append(headers[:, mask][:, 0])

    # Convert the lists of NumPy arrays to a single NumPy array and transpose
    seism_averaged: np.ndarray = np.array(seism_no_unique).T
    headers_unique: np.ndarray = np.array(headers_no_unique).T

    return seism_averaged, headers_unique

