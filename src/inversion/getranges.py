import numpy as np

from src.config_reader.enums import GetNumLayers
from src.config_reader.models import Ranges, DispersionCurve, GlobalSearchModel
from typing import Tuple, List

from src.files_processor.readers import load_ranges_from_file
from src.inversion.utils import mean_curve, cluster_dispersion_curves


def calculate_depth_limits(dc: np.ndarray, freq: np.ndarray, xi: float, lmin: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates minimum and maximum depth limits for spectral analysis layers.

    This function iteratively determines a set of depth ranges (d_min, d_max) based on the dispersion
    curve (dc), frequencies (freq), a scaling factor (xi), and a minimum wavelength (lmin).
    The depth ranges are calculated until the maximum depth (d_temp) exceeds a resolution depth (d_res).

    Args:
        dc (np.ndarray): Dispersion curve values (e.g., phase or group velocities).
        freq (np.ndarray): Corresponding frequencies for the dispersion curve.
        xi (float): Scaling factor that controls the growth of depth intervals.
        lmin (float): Minimum wavelength considered in the analysis.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - d_min (np.ndarray): Array of minimum depth values for each layer.
            - d_max (np.ndarray): Array of maximum depth values for each layer.
    """
    d_min: List[float] = []  # Initialize list for minimum depth values
    d_max: List[float] = []  # Initialize list for maximum depth values
    d_temp: float = 0  # Initialize temporary depth variable
    i: int = 0  # Initialize iteration counter
    d_res: float = np.max(dc / freq) / 2  # Calculate resolution depth

    while d_temp < d_res:  # Iterate until maximum depth exceeds resolution depth
        if i == 0:  # First layer
            d_min.append(lmin / 3)  # Set minimum depth for first layer
            d_max.append(lmin)  # Set maximum depth for first layer
        elif i == 1:  # Second layer
            d_min.append(d_max[i - 1])  # Set minimum depth based on previous layer's maximum depth
            d_max.append(d_min[i] + xi * lmin)  # Set maximum depth based on minimum depth and scaling factor
        else:  # Subsequent layers
            d_min.append(d_max[i - 1])  # Set minimum depth based on previous layer's maximum depth
            d_max.append(d_min[i] + xi * (d_max[i - 1] - d_min[i - 1]))  # Set maximum depth based on scaling factor and previous layer's thickness

        d_temp = d_max[-1]  # Update temporary depth value
        i += 1  # Increment iteration counter

    # Remove last element if it exceeds d_res (and if there are more than 1 element)
    if (d_max[-1] > d_res) and (len(d_max) > 1):
        d_max.pop()  # Remove the last d_max value
        d_min.pop()  # Remove the last d_min value

    return np.round(np.array(d_min), 1), np.round(np.array(d_max), 1)  # Convert lists to NumPy arrays, round to 1 decimal place, and return

def calculate_velocity_profile(dc: np.ndarray, depth_range: np.ndarray, vr2vs: float) -> np.ndarray:
    """
    Calculates average shear wave velocities for a given depth range.

    This function calculates the average shear wave velocity (Vs) for each layer defined by the
    provided depth range, using the dispersion curve (dc) and a conversion factor (vr2vs).

    Args:
        dc (np.ndarray): Dispersion curve values (e.g., phase or group velocities).
        depth_range (np.ndarray): A 2D array where each column represents the minimum and maximum
                                  depth for a layer. Shape: (2, number of layers).
        vr2vs (float): Conversion factor to convert Rayleigh wave velocity (Vr) to shear wave velocity (Vs).

    Returns:
        np.ndarray: A NumPy array containing the calculated average shear wave velocities for each layer,
                    plus an additional element representing the shear wave velocity at the surface.
    """
    lc = (dc / np.mean(depth_range[0])) * 0.53  # Изменено для более явного использования freq
    vs_mean = [vr2vs * dc[np.argmin(np.abs(lc - dep_mean))] for dep_mean in np.mean(depth_range, axis=0)]
    vs_mean.append(vr2vs * dc[0]) # Add shear wave velocity at the surface
    return np.array(vs_mean) # Return the velocities


def adjust_velocity_limits(vs_up: np.ndarray, vs_down: np.ndarray, vs_diff: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Adjusts the lower and upper velocity limits to ensure they are within a reasonable range and separated by a minimum difference.

    This function modifies the `vs_down` and `vs_up` arrays to enforce a minimum velocity difference (`vs_diff`)
    between layers and ensure that the upper limit is always greater than the lower limit.

    Args:
        vs_up (np.ndarray): Array of upper velocity limits.
        vs_down (np.ndarray): Array of lower velocity limits.
        vs_diff (float): Minimum allowed velocity difference between layers.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the adjusted lower and upper velocity limit arrays (vs_down, vs_up).
    """
    vs_down = np.where(vs_down < vs_diff, vs_diff, vs_down) # Ensure vs_down is not below vs_diff

    for i in range(1, len(vs_down)): # Enforce minimum velocity difference between layers
        if vs_down[i] - vs_down[i - 1] < vs_diff:
            vs_down[i] = vs_down[i - 1] + vs_diff # Increase vs_down to satisfy the minimum difference

    for i in range(len(vs_up)): # Ensure vs_up is always greater than vs_down
        if vs_up[i] <= vs_down[i]:
            vs_up[i] = vs_down[i] + vs_diff # Increase vs_up to be greater than vs_down

    return vs_down, vs_up # Return the velocity limits


def getranges(dc: np.ndarray, freq: np.ndarray, xi: float = 2) -> Ranges:
    """
    Main function to determine velocity and thickness ranges for layered earth model.

    This function orchestrates the process of calculating shear wave velocity (Vs) and layer thickness ranges
    based on the provided dispersion curve (dc) and frequencies (freq). It calls other functions to:
        1. Calculate depth limits based on the dispersion curve and a scaling factor.
        2. Calculate average shear wave velocities for each layer.
        3. Adjust the lower and upper velocity limits to ensure they are physically plausible.

    Args:
        dc (np.ndarray): Dispersion curve values (e.g., phase or group velocities).
        freq (np.ndarray): Corresponding frequencies for the dispersion curve.
        xi (float): Scaling factor that controls the growth of depth intervals. Defaults to 2.

    Returns:
        Ranges: A `Ranges` object containing the calculated shear wave velocity ranges (`velocity_shear_range`)
                and layer thickness ranges (`thicknesses_range`).
    """
    dc = np.squeeze(dc) # Remove single-dimensional entries from the shape of dc
    freq = np.squeeze(freq) # Remove single-dimensional entries from the shape of freq

    lc = dc / freq # Calculate wavelength
    lmin = np.min(lc) # Find minimum wavelength

    # Get depth limits
    d_min, d_max = calculate_depth_limits(dc, freq, xi, lmin)
    depth_range = np.vstack((d_min, d_max)) # Combine d_min and d_max into a 2D array

    # Calculate thicknesses
    thk_up: np.ndarray = d_max - d_min  # Calculate upper limit of thicknesses
    thk_up[0] = d_max[0]  # Set first layer thickness to d_max[0]
    thk_down: np.ndarray = np.full(len(thk_up), np.round(lmin / 3, 1))  # Set lower limit of thicknesses

    # Calculate mean velocity
    vs_mean: np.ndarray = calculate_velocity_profile(dc, depth_range, 1.16)  # Calculate average shear wave velocity

    # Define velocity limits
    dvs: np.ndarray = vs_mean * 0.8  # Calculate velocity variation
    vs_up: np.ndarray = vs_mean + dvs  # Calculate upper velocity limits
    vs_down: np.ndarray = vs_mean - dvs  # Calculate lower velocity limits

    # Adjust velocity limits
    vs_down: np.ndarray
    vs_up: np.ndarray
    vs_down, vs_up = adjust_velocity_limits(vs_up, vs_down, 50)  # Adjust velocity limits

    # Return ranges
    vs_range: np.ndarray = np.vstack((vs_down, vs_up)).T  # Combine vs_down and vs_up into a 2D array
    thk_range: np.ndarray = np.vstack((thk_down, thk_up)).T  # Combine thk_down and thk_up into a 2D array

    return Ranges(velocity_shear_range=vs_range, thicknesses_range=thk_range)  # Return Ranges object


def calculate_ranges_for_single_curve(disp_curve: DispersionCurve, xi: float, mode: int = 0) -> list[Ranges]:
    """
    Calculates the velocity and thickness ranges for a single dispersion curve.

    This function takes a DispersionCurve object and calculates the Ranges object using the `getranges` function.

    Args:
        disp_curve (DispersionCurve): The DispersionCurve object.
        xi (float): A scaling factor used in the `getranges` function.
        mode (int, optional): The mode number to use for range calculation. Defaults to 0.

    Returns:
        List[Ranges]: A list containing a single Ranges object calculated using the dispersion curve's
                      velocity phase and frequency.
    """
    return [getranges(disp_curve.velocity_phase[mode], disp_curve.frequency[mode], xi)]


def calculate_ranges_for_multiple_curves(disp_curves: list[DispersionCurve], mode: int = 0) -> list[Ranges]:
    """
    Calculates the velocity and thickness ranges for multiple dispersion curves.

    This function iterates through a list of DispersionCurve objects and calculates a Ranges object for each
    curve using the `getranges` function.

    Args:
        disp_curves (List[DispersionCurve]): A list of DispersionCurve objects.
        mode (int, optional): The mode number to use for range calculation. Defaults to 0.

    Returns:
        List[Ranges]: A list of Ranges objects, one for each DispersionCurve in the input list.
    """
    return [getranges(curve.velocity_phase[mode], curve.frequency[mode]) for curve in disp_curves]


def define_model_ranges(model_ranges: GlobalSearchModel, disp_curve: list[DispersionCurve]) -> list[Ranges]:
    """
    Defines the model ranges based on the configuration specified in the GlobalSearchModel object.

    This function determines how to calculate or load the model ranges (velocity shear and thicknesses) based on
    the settings in the `model_ranges` object. It can either load the ranges from a file, calculate them based on
    the mean dispersion curve, calculate them based on clustered dispersion curves, or calculate them for each
    individual dispersion curve.

    Args:
        model_ranges (GlobalSearchModel): A GlobalSearchModel object specifying how to define the model ranges.
        disp_curve (List[DispersionCurve]): A list of DispersionCurve objects.

    Returns:
        List[Ranges]: A list of Ranges objects, representing the calculated or loaded model ranges.
    """
    if model_ranges.path4vs_limits: # Load ranges from file if a path is provided
        return load_ranges_from_file(model_ranges.path4vs_limits, disp_curve)

    if model_ranges.get_num_layers == GetNumLayers.mean:  # Calculate ranges based on the mean dispersion curve
        return calculate_ranges_for_single_curve(mean_curve(disp_curve), model_ranges.xi)

    if model_ranges.get_num_layers == GetNumLayers.classes: # Calculate ranges based on clustered dispersion curves
        dc_by_classes, _ = cluster_dispersion_curves(disp_curve)
        return calculate_ranges_for_multiple_curves(dc_by_classes)

    return calculate_ranges_for_multiple_curves(disp_curve) # Calculate ranges for each individual dispersion curve
