import numpy as np
from numba import njit
from scipy.interpolate import interp1d

from src import *
from src.spectral_analysis.models import Seismogram
from src.config_reader.models import DispersionCurve
from src.files_processor.savers import save_spec_segy, save_spec_image
from typing import Union

def remove_outliers(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes outliers from a curve by iterative polynomial fitting and replacement.

    This function removes outliers from a curve represented by data points (x, y)
    using an iterative process. It first removes points before the initial maximum,
    then repeatedly fits a 7th-order polynomial and replaces points with large
    deviations from the polynomial with the mean of their neighbors.

    Args:
        x (np.ndarray): NumPy array of x-coordinates.
        y (np.ndarray): NumPy array of y-coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the filtered x and y
        coordinates after outlier removal.
    """
    # remove 1st several points before first maximum (max estimate within first 20% from all amount points)
    indxmax = np.argmax(y[:int(len(y)*0.3)])
    x, y = x[indxmax:], y[indxmax:]

    # replace points to the mean values, which difference from the polinom (7 order)
    # loop checks only 30% of all points
    for ii in range(int(len(x) * 0.2)):
        polinom = np.polyval(np.polyfit(np.arange(len(y)), y, deg=7), np.arange(len(y)))

        # exit from loop if MAPE between curve and polinom less then 5%
        if np.mean((np.abs(y - polinom) / y)) * 100 < 5:
            break

        indx = np.argmax(y - polinom)
        if 1 < indx < len(y)-1:
            y[indx] = np.mean([y[indx - 1], y[indx + 1]])

    return x, y


def _apply_smoothing(seismogram: Seismogram) -> Seismogram:
    """
   Applies a 2D smoothing window to the seismic data.

   This function applies a 2D smoothing window to the seismic data in the
   input `Seismogram` object. The window is constructed by taking the outer product
   of two 1D filters, one for the time dimension and one for the spatial
   dimension. The size of the smoothing window is determined by a fraction
   (1/20) of the number of time samples and spatial samples.

   Args:
       seismogram (Seismogram): A Seismogram object containing seismic data.

   Returns:
       Seismogram: A new Seismogram object with the smoothed seismic data.
   """
    nt, nx = seismogram.time_counts, seismogram.spatial_counts
    length_smooth = map(lambda size: (int(size / 20), size), (nt, nx))
    filter_time, filter_space = [
        np.pad(
            np.ones(size - 2 * length_size),
            pad_width=(length_size, length_size),
            mode="linear_ramp",
        )
        for length_size, size in length_smooth
    ]
    window_2d = np.outer(filter_time, filter_space)
    return Seismogram(window_2d * seismogram.data, seismogram.headers, seismogram.dt, seismogram.dx)


def _apply_padding(seismogram: Seismogram, desired_nx: int, desired_nt: int, only_nt: bool = True) -> Seismogram:
    """
    Applies padding to the seismic data.

    This function applies padding to the seismic data in the input `Seismogram`
    object. It pads the data to achieve the desired number of time samples
    (`desired_nt`) and/or spatial samples (`desired_nx`). Padding is applied
    only in the time dimension if `only_nt` is True; otherwise, padding is
    applied in both time and spatial dimensions.

    Args:
        seismogram (Seismogram): A Seismogram object containing seismic data.
        desired_nx (int): The desired number of spatial samples after padding.
        desired_nt (int): The desired number of time samples after padding.
        only_nt (bool, optional): A flag indicating whether to pad only in the
            time dimension (True) or in both time and spatial dimensions (False).
            Defaults to True.

    Returns:
        Seismogram: A new Seismogram object with the padded seismic data.
    """
    nt, nx = seismogram.time_counts, seismogram.spatial_counts
    new_nx, new_nt = max(nx, desired_nx), max(nt, desired_nt)
    pad_width = ((0, new_nt - nt), (0, 0) if only_nt else (0, new_nx - nx))
    pad_data = np.pad(seismogram.data, pad_width=pad_width)

    return Seismogram(pad_data, seismogram.headers, seismogram.dt, seismogram.dx)


def _get_wavenumbers_and_frequencies(
    seismogram: Seismogram, min_frequency: float, max_frequency: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Calculates wavenumbers and frequencies for spectral analysis.

    This function calculates the wavenumbers (k) and frequencies (freq) based on
    the properties of the input seismogram and the specified minimum and maximum
    frequencies. It also determines the indices corresponding to the minimum and
    maximum frequencies for use in later processing steps.

    Args:
        seismogram (Seismogram): A Seismogram object containing seismic data
            and related parameters (dt, time_counts, dx, spatial_counts).
        min_frequency (float): The minimum frequency for the analysis.
        max_frequency (float): The maximum frequency for the analysis.

    Returns:
        Tuple[np.ndarray, np.ndarray, int, int]: A tuple containing:
            - k (np.ndarray): NumPy array of wavenumbers.
            - freq (np.ndarray): NumPy array of frequencies between min_frequency and max_frequency.
            - ind_min_frequency (int): Index corresponding to the minimum frequency.
            - ind_max_frequency (int): Index corresponding to the maximum frequency.
    """
    df = 1 / seismogram.dt / seismogram.time_counts
    dk = 1 / seismogram.dx / seismogram.spatial_counts

    ind_min_frequency = int(np.round(min_frequency / df))
    ind_max_frequency = int(np.round(max_frequency / df))

    k = np.arange(0, seismogram.spatial_counts * dk, dk)
    freq = np.linspace(min_frequency, max_frequency, ind_max_frequency - ind_min_frequency)

    return k, freq, ind_min_frequency, ind_max_frequency

def get_dc_ranges_from_file(file_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads dispersion curve (DC) ranges from a file.

    This function reads a file containing dispersion curve ranges, using semicolons
    as delimiters. It expects the file to have at least one header row, and
    then three columns of data representing the lower bound, upper bound, and a third value.

    Args:
        file_path (Path): The path to the file containing the DC ranges.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three NumPy arrays:
            - The first array contains the values from the first data column (index 0).
            - The second array contains the values from the second data column (index 1).
            - The third array contains the values from the third data column (index 2).
            The first row from all columns is skipped from all columns.
    """
    tmp = np.genfromtxt(file_path, delimiter=";")
    return tmp[1:, 0], tmp[1:, 1], tmp[1:,2]

@njit(cache=True)
def konno_and_ohmachi(frequencies: np.ndarray,
                      spectrum: np.ndarray,
                      fcs: np.ndarray,
                      bandwidth: float = 40.0) -> np.ndarray:
    """
    Applies Konno and Ohmachi smoothing to a spectrum.

    This function applies Konno and Ohmachi smoothing to an input spectrum.
    Konno and Ohmachi smoothing is a technique used to reduce noise and
    enhance the stability of spectral estimates. It smooths the spectrum by
    averaging the spectral values within a frequency-dependent window.

    Args:
        frequencies (np.ndarray): NumPy array of frequencies corresponding to the spectrum.
        spectrum (np.ndarray): NumPy array representing the input spectrum to be smoothed.
        fcs (np.ndarray): NumPy array of center frequencies around which the smoothing is performed.
        bandwidth (float, optional): Bandwidth parameter controlling the width of the
            smoothing window. Defaults to 40.

    Returns:
        np.ndarray: A NumPy array representing the smoothed spectrum.
    """
    n = 3
    upper_limit = np.power(10, +n/bandwidth)
    lower_limit = np.power(10, -n/bandwidth)


    nf = len(fcs)
    smoothed_spectrum = np.zeros(nf)

    for fc_index, fc in enumerate(fcs):

        if fc < 1e-6:
            smoothed_spectrum[fc_index] = 0
            continue

        sumproduct = 0
        sumwindow = 0

        for f_index, f in enumerate(frequencies):
            f_on_fc = f/fc

            if (f < 1e-6) or (f_on_fc > upper_limit) or (f_on_fc < lower_limit):
                continue
            elif np.abs(f - fc) < 1E-6:
                window = 1.
            else:
                window = bandwidth * np.log10(f_on_fc)
                window = np.sin(window) / window
                window *= window
                window *= window


            sumproduct += window*spectrum[f_index]
            sumwindow += window



        if sumwindow > 0:
            smoothed_spectrum[fc_index] = sumproduct / sumwindow
        else:
            smoothed_spectrum[fc_index] = 0

    return smoothed_spectrum


def curves_processing(frequencies: list[np.ndarray], dcs: list[np.ndarray], dc_error_thr: float) \
        -> tuple[list[np.ndarray], list[np.ndarray], list]:
    """
    Processes dispersion curves by interpolating, removing outliers, smoothing, and checking approximation.

    This function performs a series of processing steps on a list of dispersion
    curves (`dcs`) and their corresponding frequencies (`frequencies`). The steps
    include interpolating curves with fewer than 10 points, removing outliers,
    smoothing using the Konno-Ohmachi function, and checking the approximation
    quality using the `check_approximation` function.

    Args:
        frequencies (List[np.ndarray]): A list of NumPy arrays representing the
            frequencies for each dispersion curve.
        dcs (List[np.ndarray]): A list of NumPy arrays representing the dispersion
            curves.
        dc_error_thr (float): A threshold for the approximation quality check.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[bool]]: A tuple containing:
            - The processed list of frequencies (List[np.ndarray]).
            - The processed list of dispersion curves (List[np.ndarray]).
            - A list of boolean flags indicating whether each dispersion curve
              passed the approximation quality check (List[bool]).
    """
    # Interpolate the dispersion curve if it has fewer than 10 points

    flags_rejecting_modes = []
    for idx, curve in enumerate(zip(frequencies, dcs)):
        frequency, dc = curve
        if len(dc) < 10:
            frequency_new = np.linspace(frequency.min(), frequency.max(), 10)
            dc = interp1d(frequency, dc)(frequency_new)  # Interpolate within the range of frequency
            frequency = np.copy(frequency_new)

        # Remove outliers from the dispersion curve
        frequencies[idx], dcs[idx] = remove_outliers(frequency, dc)

        # Smooth the dispersion curve using the Konno-Ohmachi smoothing function
        dcs[idx] = konno_and_ohmachi(frequencies[idx], dcs[idx], frequencies[idx])

        flags_rejecting_modes.append(check_approximation(dcs[idx], dc_error_thr))

    return frequencies, dcs, flags_rejecting_modes


def check_approximation(dc: np.ndarray, dc_error_thr: float) -> int:
    """
    Checks the quality of a polynomial approximation of a dispersion curve (DC).

    This function fits a 4th-degree polynomial to the input dispersion curve `dc`
    and then checks if the maximum relative error between the original curve and
    the polynomial approximation exceeds the specified threshold `dc_error_thr`.

    Args:
        dc (np.ndarray): A NumPy array representing the dispersion curve.
        dc_error_thr (float): The maximum acceptable relative error threshold.

    Returns:
        bool: True if the maximum relative error is less than or equal to
            `dc_error_thr`, indicating a good approximation; False otherwise.
    """
    polynomial = np.polyval(np.polyfit(np.arange(len(dc)), dc, deg=4), np.arange(len(dc)))
    return False if np.max(np.abs(dc - polynomial) / dc) > dc_error_thr else True


def spectral_processing(traces: np.ndarray,
                      headers: np.ndarray,
                      dt: float,
                      dx: float,
                      spec_name: str,
                      spec_dc_dir: Path,
                      spec_image_dir: Path,
                      spec_segy_dir: Path,
                      peaker,
                      peak_fraction: float,
                      cutoff_fraction: float,
                      fk_transform,
                      vf_transform,
                      qc_spectral: bool,
                      dc_error_thr,
                      ) -> bool:
    """
    Performs spectral processing on seismic data.

    This function performs a series of spectral analysis steps on seismic data,
    including FK transform, VF transform, dispersion curve extraction, and curve
    processing. It also saves the resulting spectral images and SEG-Y files if
    `qc_spectral` is enabled.

    Args:
        traces (np.ndarray): NumPy array of seismic traces.
        headers (np.ndarray): NumPy array of seismic headers.
        dt (float): The time sampling interval.
        dx (float): The spatial step between traces.
        spec_name (str): A name for the spectral data, used for file naming.
        spec_dc_dir (str): Directory to save DC spectra.
        spec_image_dir (str): Directory to save spectral images.
        spec_segy_dir (str): Directory to save spectral SEG-Y files.
        peaker (Peaker): An object with peak_dc method to extract dispersion curves.
        peak_fraction (float): Parameter for peak extraction.
        cutoff_fraction (float): Parameter for peak extraction.
        fk_transform (FkTransform): An object with run method that computes the f-k spectrum.
        vf_transform (VfTransform): An object with run method that computes the v-f spectrum.
        qc_spectral (bool): A flag indicating whether to save spectral images and SEG-Y files.
        dc_error_thr (float): Threshold for DC curves processing.

    Returns:
        Union[None, bool]: Returns None if the function executes successfully.
    """

    # Create a Seismogram object from the traces, headers, dt, and dx
    seismogram = Seismogram(traces, headers, dt, dx)

    # Extract relief from the headers
    num_traces = len(seismogram.headers[HEADER_ELEV_IND])
    relief_cur = seismogram.headers[HEADER_ELEV_IND, int(num_traces / 2)]

    # Extract CMP coordinates from the headers
    cmp_x, cmp_y = headers[HEADER_CDP_X_IND:HEADER_CDP_Y_IND + 1, 0]
    # Calculate the f-k spectrum using the fk_transform object


    spectra = fk_transform.run(seismogram)

    # Calculate the v-f spectrum using the vf_transform object
    vf_transform.run(spectra)

    # Extract the dispersion curve (DC) using the peaker object
    frequencies, dcs, freq_limits, lover_v, upper_v = peaker.peak_dc(spectra, peak_fraction, cutoff_fraction)

    # Curves processing
    frequencies, dcs, flags_rejecting_modes = curves_processing(frequencies, dcs, dc_error_thr)
    if qc_spectral:
        save_spec_image(spec_image_dir / f"{spec_name}.png",
                        spectra,
                        frequencies,
                        freq_limits,
                        lover_v,
                        upper_v,
                        dcs,
                        flags_rejecting_modes)
        save_spec_segy(spec_segy_dir / f"{spec_name}.sgy",
                        spectra,
                        frequencies,
                        dcs,
                        (cmp_x, cmp_y))


    # Save the spectral analysis results if quality control is enabled
    # Create a DispersionCurve object and save it to a file
    frequency_all_modes, dc_all_modes = {}, {}
    for mode_i in range(len(frequencies)):
        if flags_rejecting_modes[mode_i]:
            frequency_all_modes[mode_i] = frequencies[mode_i]
            dc_all_modes[mode_i] = dcs[mode_i]
        else:
            break

    if bool(dc_all_modes):
        DispersionCurve(
            frequency=frequency_all_modes,
            velocity_phase=dc_all_modes,
            cmp_x=cmp_x,
            cmp_y=cmp_y,
            relief=relief_cur,
            spec_name=f"{spec_name}.npz",
            num_modes=len(dc_all_modes.keys()),
        ).save(
            spec_dc_dir / f"{spec_name}.npz"
        )
        return True
    else:
        return False


