import numpy as np
from scipy.interpolate import interp1d
from sklearn import metrics
from typing import Literal, Union, Optional, Callable
from disba import GroupSensitivity, PhaseSensitivity
from sklearn.mixture import GaussianMixture

from src.config_reader.enums import VelocityType
from src.config_reader.enums import WaveType
from src.config_reader.models import DispersionCurve


def compute_misfit(
    observed_disp_curve: np.ndarray,
    computed_disp_curves: np.ndarray,
    misfit_metric: Callable[[np.ndarray, np.ndarray], float],
) -> Union[float, np.ndarray]:
    """
    Computes the misfit between observed and computed dispersion curves using a given metric.

    This function calculates the misfit, which quantifies the difference between an observed dispersion
    curve and one or more computed dispersion curves. The misfit is calculated using a user-provided
    misfit metric function.

    Args:
        observed_disp_curve (np.ndarray): The observed dispersion curve (target).
        computed_disp_curves (np.ndarray): The computed dispersion curve(s) to compare against the observed curve.
                                          Can be a 1D array (single curve) or a 2D array (multiple curves).
        misfit_metric (Callable[[np.ndarray, np.ndarray], float]): A callable (function) that takes two NumPy arrays
                                                                   (observed and computed dispersion curves) as input
                                                                   and returns a float representing the misfit.

    Returns:
        Union[float, np.ndarray]: If `computed_disp_curves` is a 1D array (single curve), returns a float representing
                                  the misfit. If `computed_disp_curves` is a 2D array (multiple curves), returns a 1D
                                  NumPy array containing the misfit values for each computed curve.  Returns 0 if
                                  computed_disp_curves has a dimension different than 1 or 2.
    """
    if np.ndim(computed_disp_curves) == 2:
        misfit: np.ndarray = np.zeros(np.size(computed_disp_curves, 0))  # Initialize misfit array

        for i in range(np.size(computed_disp_curves, 0)):  # Iterate over each computed dispersion curve
            misfit[i] = misfit_metric(observed_disp_curve, computed_disp_curves[i])  # Calculate misfit
        return misfit  # Return array of misfit values

    if np.ndim(computed_disp_curves) == 1:
        return misfit_metric(observed_disp_curve, computed_disp_curves)  # Return misfit value for single curve

    return 0  # Return 0 if computed_disp_curves has an unsupported dimension

def get_misfit(
    misfit_metric: str,
    observed_disp_curve: np.ndarray,
    computed_disp_curves: np.ndarray,
) -> Union[np.ndarray, float]:
    """
    Calculates the misfit between observed and computed dispersion curves using a specified metric.

    This function retrieves a misfit metric function based on the provided `misfit_metric` string and then
    calls the `compute_misfit` function to calculate the misfit between the observed and computed dispersion curves.

    Args:
        misfit_metric (str): A string identifying the misfit metric to use (e.g., "me", "mae", "rmse", "mape").
                               Supported metrics are max_error ("me"), mean_absolute_error ("mae"),
                               mean_squared_error ("mse"), root_mean_squared_error ("rmse"),
                               mean_absolute_percentage_error ("mape"), median_absolute_error ("medae"),
                               mean_squared_log_error ("msle"), mean_poisson_deviance ("mpd"),
                               mean_gamma_deviance ("mgd"), and d2_absolute_error_score ("d2").
        observed_disp_curve (np.ndarray): The observed dispersion curve (target).
        computed_disp_curves (np.ndarray): The computed dispersion curve(s) to compare against the observed curve.

    Returns:
        Union[np.ndarray, float]: The misfit value(s) as calculated by the specified metric.  If `computed_disp_curves`
                                  is 2 dimensional, returns an `np.ndarray`.  If it is 1 dimensional, returns float.
    """
    metric_dict = {
        "me": metrics.max_error,
        "mae": metrics.mean_absolute_error,
        "mse": metrics.mean_squared_error,
        "rmse": metrics.root_mean_squared_error,
        "mape": metrics.mean_absolute_percentage_error,
        "medae": metrics.median_absolute_error,
        "msle": metrics.mean_squared_log_error,
    }  # Dictionary of supported misfit metrics

    _metric: Callable[[np.ndarray, np.ndarray], float] = metric_dict.get(misfit_metric) # Get the specified misfit metric function

    return compute_misfit(observed_disp_curve, computed_disp_curves, _metric)  # Calculate and return the misfit


def initialize_model(
    velocity_shear_range: np.ndarray,
    thicknesses_range: np.ndarray,
    vp2vs: float,
    count_agents: int,
    lock_vp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initializes a population of velocity models with random shear wave velocities, thicknesses, compressional wave velocities, and densities.

    This function generates a set of `count_agents` velocity models, where each model consists of shear wave
    velocities (Vs), layer thicknesses, compressional wave velocities (Vp), and densities. The Vs and
    thicknesses are randomly sampled from the provided ranges. If `lock_vp` is True, all agents have the
    same Vp.

    Args:
        velocity_shear_range (np.ndarray): A 2D NumPy array defining the minimum and maximum shear wave
                                          velocity for each layer. Shape: (number of layers, 2).
        thicknesses_range (np.ndarray): A 2D NumPy array defining the minimum and maximum thickness for
                                       each layer. Shape: (number of layers, 2).
        vp2vs (float): Vp/Vs ratio used to calculate compressional wave velocities from shear wave velocities.
        count_agents (int): The number of velocity models (agents) to generate.
        lock_vp (bool): If True, all agents have the same compressional wave velocities.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the initialized model parameters:
            - vs_positions (np.ndarray): Shear wave velocities for all agents. Shape: (count_agents, number of layers).
            - thk_positions (np.ndarray): Layer thicknesses for all agents. Shape: (count_agents, number of layers).
            - vp_positions (np.ndarray): Compressional wave velocities for all agents. Shape: (count_agents, number of layers).
            - rho_positions (np.ndarray): Densities for all agents. Shape: (count_agents, number of layers).
    """
    vs_positions: np.ndarray = np.random.uniform(
        velocity_shear_range[:, 0],
        velocity_shear_range[:, 1],
        size=(
            count_agents,
            np.size(velocity_shear_range, 0),
        ),
    )  # Generate random shear wave velocities

    thk_positions: np.ndarray = np.random.uniform(
        thicknesses_range[:, 0],
        thicknesses_range[:, 1],
        size=(
            count_agents,
            np.size(thicknesses_range, 0),
        ),
    )  # Generate random layer thicknesses

    if lock_vp:
        vs_mean: np.ndarray = np.mean(velocity_shear_range, axis=1)  # Calculate mean shear wave velocity for each layer
        vp_positions = np.tile(vs_mean * vp2vs, (count_agents, 1))  # Tile the compressional wave velocities
    else:
        vp_positions: np.ndarray = vs_positions * vp2vs  # Calculate compressional wave velocities from shear wave velocities

    rho_positions: np.ndarray = (1.2475 + 0.3992 * vp_positions * 1e-3 - 0.026 * (vp_positions * 1e-3) ** 2) * 1e3  # Calculate densities (Gardner et al. relationship)

    return vs_positions, thk_positions, vp_positions, rho_positions  # Return the initialized model parameters


def sensitiv(
    num_mode: int,
    f: np.ndarray,
    vs: np.ndarray,
    vp: np.ndarray,
    rho: np.ndarray,
    thk: np.ndarray,
    wave: "WaveType" = "rayleigh",
    veltype: "VelocityType" = "phase",
    parameter: Literal["velocity_s", "velocity_p", "density"] = "velocity_s",
) -> np.ndarray:
    """
    Calculates the sensitivity kernel for a given set of model parameters.

    This function computes the sensitivity of surface wave dispersion to variations in shear wave velocity,
    compressional wave velocity, or density.  It uses the `disba` library to calculate the sensitivity kernel
    based on a layered Earth model.

    Args:
        num_mode (int): The mode number to calculate the sensitivity for (e.g., 0 for fundamental mode).
        f (np.ndarray): Array of frequencies (Hz) at which to calculate the sensitivity.
        vs (np.ndarray): Array of shear wave velocities (m/s) for each layer.
        vp (np.ndarray): Array of compressional wave velocities (m/s) for each layer.
        rho (np.ndarray): Array of densities (kg/m^3) for each layer.
        thk (np.ndarray): Array of layer thicknesses (m).
        wave (Literal["rayleigh", "love"], optional): The type of surface wave ("rayleigh" or "love"). Defaults to "rayleigh".
        veltype (Literal["phase", "group"], optional): The type of dispersion curve ("phase" or "group"). Defaults to "phase".
        parameter (Literal["velocity_s", "velocity_p", "density"], optional): The parameter to calculate the sensitivity for.
            Can be shear wave velocity ("velocity_s"), compressional wave velocity ("velocity_p"), or density ("density").
            Defaults to "velocity_s".

    Returns:
        np.ndarray: A 2D NumPy array representing the sensitivity kernel. The shape is (number of frequencies, number of layers).

    Raises:
        ValueError: If an invalid `veltype` is provided.
    """

    # Construct the velocity model array expected by disba
    velocity_model: np.ndarray = (
        np.vstack(
            (
                np.hstack((np.squeeze(thk), 0))[None, :],  # Layer thicknesses (with half-space) in km
                np.atleast_2d(vp),  # Compressional wave velocities in km/s
                np.atleast_2d(vs),  # Shear wave velocities in km/s
                np.atleast_2d(rho),  # Densities in g/cm^3
            )
        )
        / 1000  # Convert units to km/s and g/cm^3
    )

    t: np.ndarray = 1 / f  # Calculate periods (s) from frequencies (Hz)
    sens: np.ndarray = np.zeros((t.size, vs.size))  # Initialize sensitivity matrix

    # Select the appropriate sensitivity function based on velocity type
    solver_class = PhaseSensitivity(*velocity_model) if veltype == "phase" else GroupSensitivity(*velocity_model)

    # Iterate over frequencies and calculate sensitivity for each layer
    for ifreq in range(t.size):
        try:
            sens[ifreq, :] = solver_class(
                t[ifreq],
                mode = num_mode,
                wave = wave,
                parameter = parameter
            ).kernel
        except Exception:
            sens[ifreq, :] = np.zeros(vs.shape)

    return sens  # Return the sensitivity matrix


def mean_curve(dispersion_curves: list[DispersionCurve], mode: int = 0) -> DispersionCurve:
    """
    Calculates the mean dispersion curve from a list of dispersion curves.

    This function takes a list of DispersionCurve objects, interpolates their phase velocities
    to a common frequency axis (defined by the longest curve's frequency array), and then
    calculates the mean phase velocity at each frequency. The function returns a new
    DispersionCurve object representing the mean dispersion curve.

    Args:
        dispersion_curves (List[DispersionCurve]): A list of DispersionCurve objects to average.
        mode (int, optional): The mode number to use for averaging (e.g., 0 for the fundamental mode). Defaults to 0.

    Returns:
        DispersionCurve: A new DispersionCurve object representing the mean dispersion curve.
    """
    # Find the DispersionCurve object with the longest frequency array
    longest_frequency, velocities = interpolate_dispersion_curves(dispersion_curves)

    # Calculate the mean phase velocities across all dispersion curves
    mean_velocity_phase: np.ndarray = np.mean(velocities, axis=0)

    # Create and return a new DispersionCurve object with the mean phase velocities
    return DispersionCurve(
        frequency=[longest_frequency],
        velocity_phase=[mean_velocity_phase],
        cmp_x=0,
        cmp_y=0,
        inv_path=0,
        relief=0,
        spec_name='',
        num_modes=1,
    )


def cluster_dispersion_curves(dispersion_curves: list[DispersionCurve], mode: int = 0) -> tuple[list[DispersionCurve], np.ndarray]:
    """
    Clusters dispersion curves based on their phase velocities using a Gaussian Mixture Model (GMM).

    This function takes a list of DispersionCurve objects and clusters them based on their phase velocities
    at a common set of frequencies. It uses a Gaussian Mixture Model to identify distinct clusters in the
    velocity data and returns a list of representative DispersionCurve objects, one for each cluster, along
    with the cluster labels assigned to each input DispersionCurve. The representative DispersionCurves are
    created by averaging the phase velocities within each cluster.

    Args:
        dispersion_curves (List[DispersionCurve]): A list of DispersionCurve objects to cluster.
        mode (int, optional): The mode number to use for clustering (e.g., 0 for the fundamental mode). Defaults to 0.

    Returns:
        Tuple[List[DispersionCurve], np.ndarray]: A tuple containing:
            - dc_by_classes (List[DispersionCurve]): A list of representative DispersionCurve objects, one for each cluster.
            - best_labels (np.ndarray): A NumPy array of cluster labels assigned to each input DispersionCurve.
                                         The length of this array is equal to the length of the input `dispersion_curves`.
    """

    # Find the longest frequency array among all dispersion curves
    longest_frequency, velocities = interpolate_dispersion_curves(dispersion_curves)

    # Determine the optimal number of GMM components using Bayesian Information Criterion (BIC)
    lowest_bic: float = np.inf
    best_labels: Optional[np.ndarray] = None  # Use Optional for best_labels

    for n_components in range(1, 5):  # Iterate through different numbers of components
        gmm: GaussianMixture = GaussianMixture(n_components=n_components, random_state=0)  # Instantiate GMM

        gmm.fit(velocities)  # Fit GMM to the interpolated velocities

        bic: float = gmm.bic(velocities)  # Calculate BIC

        if bic < lowest_bic:  # Check for lowest BIC
            lowest_bic = bic
            best_labels = gmm.predict(velocities)  # Predict cluster labels

    # Create representative DispersionCurve objects for each cluster
    classes: np.ndarray = np.unique(best_labels)  # Get unique cluster labels
    dc_by_classes: list[DispersionCurve] = [
        DispersionCurve(
            frequency=[longest_frequency],
            velocity_phase=[np.mean(velocities[best_labels == class_], axis=0)],
            cmp_x=0,
            cmp_y=0,
            inv_path=0,
            num_modes=1,
            relief=0,
            spec_name='',

        )
        for class_ in classes
    ]

    return dc_by_classes, best_labels


def interpolate_dispersion_curves(
    dispersion_curves: list[DispersionCurve],
    mode: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolates a list of dispersion curves to a common frequency array.

    Finds the longest frequency array among the dispersion curves, and interpolates all curves to
    that frequency axis. Returns the longest frequency array and the interpolated velocities.

    Args:
        dispersion_curves: List of DispersionCurve objects.
        mode: The mode number to use.

    Returns:
        A tuple containing:
            - The longest frequency array (np.ndarray).
            - A NumPy array of interpolated velocities.
    """
    # Find the longest frequency array and its corresponding DispersionCurve
    longest_dc: DispersionCurve = max(
        dispersion_curves, key=lambda dc: len(dc.frequency[mode])
    )
    longest_frequency: np.ndarray = longest_dc.frequency[mode]

    # Interpolate all dispersion curves to the longest frequency array
    velocities: np.ndarray = np.array(
        [
            interp1d(
                dc.frequency[mode],
                dc.velocity_phase[mode],
                fill_value="extrapolate",
            )(longest_frequency)
            for dc in dispersion_curves
        ]
    )
    return longest_frequency, velocities