import numpy as np
from scipy.linalg import solve, convolution_matrix
from src import *
from scipy.interpolate import interp1d


def global_spitz_interpolation(traces: np.ndarray, headers: np.ndarray, desired_step: float) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Performs global Spitz interpolation on seismic traces to regularize offsets.

    This function regularizes the offsets of seismic traces by interpolating new traces
    at regular intervals using a Spitz-based method.  It iteratively refines the
    regularization step until the desired step size is achieved.

    Args:
        traces: A 2D NumPy array of seismic traces (num_samples x num_traces).
        headers: A 2D NumPy array of trace headers (num_header_fields x num_traces).
                 Must contain offset (HEADER_OFFSET_IND), elevation (HEADER_ELEV_IND),
                 CDP X (HEADER_CDP_X_IND), and CDP Y (HEADER_CDP_Y_IND) information.
    Returns:
        A tuple containing:
            - The regularized seismic traces (NumPy array with potentially more traces).
            - The updated trace headers (NumPy array with potentially more traces).
    """


    # 2. Prepare for Interpolation
    headers[HEADER_OFFSET_IND] = np.float32(headers[HEADER_OFFSET_IND])
    elevation_function = interp1d(headers[HEADER_OFFSET_IND], headers[HEADER_ELEV_IND]) # Elevation interpolation function
    cdp_x, cdp_y = headers[HEADER_CDP_X_IND][0], headers[HEADER_CDP_Y_IND][0] # Common CDP coordinates

    offsets = headers[HEADER_OFFSET_IND]
    initial_step = np.max(np.diff(np.abs(offsets)))
    steps = np.linspace(initial_step, desired_step, 4)

    # Iterative Regularization
    for cur_step in steps:

        regular_offsets = np.arange(min(offsets), max(offsets) + cur_step, cur_step) # Regular offset grid
        nearest_indices = get_nearest_indexes(offsets, regular_offsets, cur_step / 2)  # Find nearest existing traces to regular offsets

        mask = get_mask(regular_offsets, offsets, cur_step / 2) # Create a mask indicating which regular offsets are missing

        all_edges_alternating_sequences = find_alternating_sequences(mask) # Find sequences of missing offsets

        traces_for_insert, headers_for_insert = [], []  # Store traces and headers to insert

        # Spitz Interpolation and Header Creation for Missing Offsets
        for tmp_ind in all_edges_alternating_sequences: # Iterate through sequences of missing offsets
            ind_true = tmp_ind[0] + np.where(mask[tmp_ind])[0] # Indexes from mask that is True
            ind_false = tmp_ind[0] + np.where(mask[tmp_ind] == False)[0] # Indexes from mask that is False
            traces_after_spitz = spitz_regularization(traces[:, nearest_indices][:, ind_true]) # Apply Spitz regularization

            insert_headers = np.zeros((headers.shape[0], len(ind_false)))  # Create headers for interpolated traces
            insert_headers[HEADER_OFFSET_IND] = regular_offsets[ind_false] # Set offset values in the insert headers
            insert_headers[HEADER_CDP_X_IND], insert_headers[HEADER_CDP_Y_IND] = cdp_x, cdp_y # Set CDP values

            headers_for_insert.append(insert_headers) # Append headers to insertion list
            traces_for_insert.append(traces_after_spitz[:, ind_false - tmp_ind[0]]) # Append traces to insertion list
            # print("headers_for_insert", headers_for_insert[HEADER_OFFSET_IND])

        # Insert New Traces and Headers
        if headers_for_insert: # If there are traces to insert
            headers_for_insert = np.hstack(headers_for_insert) # Concatenate headers to insert
            traces_for_insert = np.hstack(traces_for_insert) # Concatenate traces to insert

            insert_indexes = np.searchsorted(headers[HEADER_OFFSET_IND], headers_for_insert[HEADER_OFFSET_IND]) # Find indices to insert traces
            traces = np.insert(traces, insert_indexes, traces_for_insert, axis=1) # Insert traces
            headers = np.insert(headers, insert_indexes, headers_for_insert, axis=1) # Insert headers
            offsets = np.insert(offsets, insert_indexes, headers_for_insert[HEADER_OFFSET_IND], axis=0) # Update offsets


    # Elevation Correction
    headers[HEADER_ELEV_IND] = elevation_function(headers[HEADER_OFFSET_IND]) # Interpolate elevation values
    offsets = np.int32(headers[HEADER_OFFSET_IND] / desired_step) * desired_step # Quantize offsets
    _, indexes = np.unique(offsets, return_index=True) # Find unique quantized offsets
    traces, headers = traces[:, indexes], headers[:, indexes] # Keep only unique offset traces

    return traces, headers


def get_nearest_indexes(array1: np.ndarray, array2: np.ndarray, thr: float) -> list[int] | None:
    """
    Finds the indices of elements in array1 that have a nearest neighbor in array2 within a specified threshold.

    This function iterates through `array1` and, for each element, finds the minimum absolute difference
    between that element and all elements in `array2`. If this minimum difference is less than the specified
    threshold `thr`, the index of the element in `array1` is added to the list of `nearest_indices`.

    Args:
        array1 (np.ndarray): A 1D NumPy array.
        array2 (np.ndarray): A 1D NumPy array.
        thr (float): The threshold value. Only indices in `array1` whose nearest neighbor in `array2`
                   is within this threshold will be included in the result.

    Returns:
        Optional[List[int]]: A list of integer indices representing the elements in `array1` that meet the
                             threshold criteria. Returns `None` if no such indices are found.
    """
    nearest_indices = []  # Initialize an empty list to store indices
    for ind in range(len(array2)):  # Iterate through array1
        nearest_indices.append(np.argmin(np.abs(array2[ind] - array1)))  # Add index to the list
    if not nearest_indices:
        return None
    else:
        return nearest_indices


def get_mask(array1: np.ndarray, array2: np.ndarray, thr: float) -> np.ndarray | None:
    """
    Generates a boolean mask based on the proximity of elements in array1 to elements in array2.

    For each element in `array1`, this function checks if its nearest neighbor in `array2` is within
    a specified threshold `thr`. If the minimum absolute difference between an element in `array1` and all
    elements in `array2` is less than `thr`, the corresponding element in the mask is set to `True`;
    otherwise, it is set to `False`.

    Args:
        array1 (np.ndarray): A 1D NumPy array.
        array2 (np.ndarray): A 1D NumPy array.
        thr (float): The threshold value.

    Returns:
        Optional[np.ndarray]: A NumPy boolean array (mask) indicating whether each element in `array1` has a
                             nearest neighbor in `array2` within the specified threshold. Returns `None` if
                             `array1` is empty.
    """
    if not array1.size:
        return None
    mask: np.ndarray = np.array([np.min(np.abs(array1[i] - array2)) < thr for i in range(len(array1))], dtype=bool)

    return mask


def find_alternating_sequences(mask: np.ndarray) -> list[list[int]]:
    """
    Finds alternating True/False sequences in a boolean mask, focusing on sequences of the form [True, False, True].

    Args:
        mask: A 1D NumPy array of boolean values (True or False).

    Returns:
        A list of lists, where each inner list represents a sequence of indices that form a [True, False, True] pattern.
        Returns an empty list if no such sequences are found or if the input mask is invalid.
    """

    def lists_match(list1: list, list2: list) -> bool:
        """Checks if two lists are identical (same elements in the same order)."""
        return len(list1) == len(list2) and list(list1) == list(list2)

    def split_on_gap(sequence: list[int], bool_seq: np.ndarray) -> list[list[int]]:
        """Splits a sequence of indices into sub-sequences based on gaps in indices or changes in mask values."""
        result = []
        if len(sequence):
            tmp, prv = [], sequence[0]
            for l in sequence:
                if (l - prv > 1) or (bool_seq[prv] == bool_seq[l]):
                    result.append(tmp)
                    tmp = []
                tmp.append(l)
                prv = l
            result.append(tmp)
        return result[1:]

    ntr = len(mask)
    frst_true_ind = next((ii for ii in range(ntr) if mask[ii]), None)
    end_true_ind = next((ii for ii in reversed(range(ntr)) if mask[ii]), None)




    # Early exist condition, returns empty array in invalid conditions
    if frst_true_ind is None or end_true_ind is None:
        return []
    if frst_true_ind == end_true_ind:
        return []

    mask_from_true_to_true = list(mask[frst_true_ind:end_true_ind + 1])  # Extract the relevant portion of the mask
    ntr: int = len(mask_from_true_to_true)  # Update ntr to the length of the extracted mask
    seq_cur: list[int] = []

    # Find sequences and append to array
    for ii in range(ntr-2):
        if lists_match(mask_from_true_to_true[ii: ii + 3], [True, False, True]):
            for ind in [ii, ii + 1, ii + 2]:
                if not ind in seq_cur:
                    seq_cur.append(ind)

    seq_cur = [x+frst_true_ind for x in seq_cur]
    all_seq: list[list[int]] = split_on_gap(seq_cur, mask) #Split for gaps
    return all_seq


def spitz_regularization(data: np.ndarray, npf: int = 1, prewhite1: float = 1e-4, prewhite2: float = 1e-4) -> np.ndarray:
    """
    Regularizes seismic data using the Spitz regularization method.

    This function applies the Spitz regularization method to seismic data to reduce noise and improve data quality.
    It performs frequency interpolation using a prediction filter and handles both even and odd numbers of time samples.

    Args:
        data (np.ndarray): 2D NumPy array representing the seismic data. The shape is (nt, nx), where nt is the
                           number of time samples and nx is the number of receivers/traces.
        npf (int, optional): The length of the prediction filter. A longer filter can capture more complex
                              patterns in the data but may also be more sensitive to noise. Defaults to 1.
        prewhite1 (float, optional): Pre-whitening factor for the prediction filter. This helps stabilize the
                                      filter and prevent it from being overly influenced by strong frequencies.
                                      Defaults to 1e-4.
        prewhite2 (float, optional): Pre-whitening factor for the frequency interpolation. This helps reduce
                                      artifacts in the interpolated data. Defaults to 1e-4.

    Returns:
        np.ndarray: The regularized seismic data (same shape as input `data`).

    """
    # Get the dimensions of the seismic data
    nt: int
    nx: int
    nt, nx = data.shape

    # Initialize the interpolated frequency data (complex type for frequency domain representation)
    INTDF: np.ndarray = np.zeros((nt, 2 * nx - 1), dtype='complex')

    # Perform FFT once for all data (along the time axis)
    DF1: np.ndarray = np.fft.fft(data, nt * 2, axis=0)
    DF2: np.ndarray = np.fft.fft(data, axis=0)

    # Iterate over time samples (frequency interpolation for each time sample)
    for j in range(nt):
        # Calculate the prediction filter
        PF: np.ndarray = prediction_filter(DF1[j, :], npf, prewhite1)

        # Interpolate frequency data
        INTDF[j, :] = interpolate_freq(DF2[j, :], PF, prewhite2)

    # Handle both even and odd nt (conjugate symmetry for inverse FFT)
    if nt % 2 == 0:  # Even number of time samples
        INTDF[nt // 2:, :] = np.conj(INTDF[1: nt // 2 + 1, :][::-1])
    else:  # Odd number of time samples
        INTDF[(nt + 1) // 2:, :] = np.conj(INTDF[1: (nt + 1) // 2, :][::-1])

    # Using ifft and filling the d_interp array (inverse FFT to return to time domain)
    return np.real(np.fft.ifft(INTDF.T, nt).T)


def prediction_filter(k_vector: np.ndarray, npf: int, prewhite: float) -> np.ndarray:
    """
    Calculates a prediction filter from frequency data using a least-squares approach with pre-whitening.

    This function calculates a prediction filter from input frequency data (k-vector).
    It constructs a system of equations based on the input data and solves it using
    a least-squares approach with pre-whitening regularization. This helps stabilize the
    solution and prevent overfitting to noise.

    Args:
        k_vector (np.ndarray): 1D NumPy array representing the input frequency data (k-vector).
        npf (int): The length of the prediction filter (number of filter coefficients).
        prewhite (float): Pre-whitening factor to stabilize the solution. A small value (e.g., 1e-4)
                         is typically used.

    Returns:
        np.ndarray: 1D NumPy array representing the prediction filter coefficients.

    """
    # Set variable to determine the vector
    ns: int = len(k_vector)

    # Construct the matrix C
    C: np.ndarray = np.array([k_vector[j:j + npf + 1][::-1] for j in range(ns - npf)], dtype='complex')

    # Construct the matrices A and B
    A: np.ndarray = np.vstack((C[:, 1:npf + 1], np.conj(C[:, 0:npf][:, ::-1])))
    B: np.ndarray = np.hstack((C[:, 0], np.conj(C[:, npf])))

    # Solve the system A.PF = B using least-squares with pre-whitening
    R: np.ndarray = np.conj(A).T @ A  # R = A^H * A
    g: np.ndarray = np.conj(A).T @ B  # g = A^H * B
    mu: float = prewhite * np.trace(R) / npf  # Calculate pre-whitening factor

    # Use `scipy.linalg.solve` instead of `np.linalg.inv` to avoid explicit matrix inversion
    PF: np.ndarray = solve(R + mu * np.eye(npf), g)  # Solve the linear system (R + mu*I) * PF = g

    return PF


def interpolate_freq(k_vector_in: np.ndarray, PF: np.ndarray, prewhite: float) -> np.ndarray:
    """
    Interpolates frequency data using a prediction filter.

    This function interpolates frequency data (k-vector) using a prediction filter (PF).
    It leverages convolution matrices to efficiently apply the prediction filter in the
    frequency domain and solves a linear system to estimate the missing frequency components.

    Args:
        k_vector_in (np.ndarray): 1D NumPy array representing the input frequency data (k-vector).
        PF (np.ndarray): 1D NumPy array representing the prediction filter.
        prewhite (float): Pre-whitening factor to stabilize the solution.

    Returns:
        np.ndarray: 1D NumPy array representing the interpolated frequency data.

    """
    # Define filter length and input size
    npf: int = len(PF)
    ns: int = len(k_vector_in)
    n_res: int = 2 * ns - 1  # Length of the resulting interpolated vector

    # Create forward and backward prediction filters
    TMPF1: np.ndarray = np.hstack((PF[::-1], -1))  # Prediction filter with a -1 tap for prediction error
    TMPF2: np.ndarray = np.conj(TMPF1[::-1])  # Complex conjugate of the time-reversed filter

    # Use scipy's convolution_matrix for the first and second filters
    W1: np.ndarray = convolution_matrix(TMPF1, n_res - npf) # convolution matrix is nxn
    W2: np.ndarray = convolution_matrix(TMPF2, n_res - npf)

    # Stack convolution matrices and extract relevant columns
    WT: np.ndarray = np.hstack((W1, W2)).T
    A: np.ndarray = WT[:, 1:n_res:2]  # Matrix A that maps from k_vector_out to k_vector_in
    B: np.ndarray = -1 * (WT[:, 0:n_res:2] @ k_vector_in) # This has the same shape as A

    # Set up the regularization and least-squares problem
    R: np.ndarray = np.conj(A).T @ A # Matrix A^H * A
    g: np.ndarray = np.conj(A).T @ B # Vector A^H * B

    # Compute prewhitening factor for regularization
    mu: float = prewhite * np.trace(R) / npf

    # Solve the system without inverting the matrix explicitly
    k_vector_out: np.ndarray = solve(R + mu * np.eye(ns - 1), g)  # Regularized solution for missing frequencies

    # Construct the result
    res: np.ndarray = np.zeros(n_res, dtype='complex') # allocate size for result
    res[::2] = k_vector_in # set the even cells
    res[1::2] = k_vector_out # set the odd cells, that are missing from the `k_vector_in`

    return res