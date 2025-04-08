import numpy as np
from dataclasses import dataclass
from typing import Optional
from disba import DispersionError, GroupDispersion, PhaseDispersion
from joblib import Parallel, delayed

from src.config_reader.enums import WaveType, VelocityType


DC_VALUE = 5e-05 # Phase velocity increment for root finding.

@dataclass
class DCModel:
    """
    Parameter storage class for dispersion curve calculations.

    This dataclass holds the parameters required to compute dispersion curves, including wave type,
    velocity profiles, densities, thicknesses, and frequencies.
    """
    wave_type: "WaveType"  # Type of wave ("rayleigh" or "love").
    velocity_type: "VelocityType"  # Velocity type ("phase" or "group").
    num_mode: Optional[int] = None  # Mode number (e.g., 0 for fundamental mode). Optional, defaults to None
    frequencies: Optional[np.ndarray] = None  # Frequencies for dispersion curve calculation (Hz). Optional, defaults to None
    velocity_shear: Optional[np.ndarray] = None  # Shear wave velocity profile (m/s). Optional, defaults to None
    velocity_press: Optional[np.ndarray] = None  # Compressional wave velocity profile (m/s). Optional, defaults to None
    densities: Optional[np.ndarray] = None  # Density profile (g/cm^3). Optional, defaults to None
    thicknesses: Optional[np.ndarray] = None  # Layer thicknesses (m). Optional, defaults to None
    count_threads: int = -1  # Number of threads for parallel processing. Defaults to -1 (all available cores).

class ComputerDC:
    """
    Computes dispersion curves for surface waves.

    This class takes a DCModel object as input and provides methods to calculate dispersion curves
    using different approaches (single-core and multi-core).
    """
    def __init__(self, dc_params: DCModel) -> None:
        """
        Initializes the ComputerDC object with parameters from a DCModel.

        Args:
            dc_params: A DCModel object containing the parameters for dispersion curve calculation.
        """
        self.num_mode: Optional[int] = dc_params.num_mode
        self.frequencies: Optional[np.ndarray] = dc_params.frequencies
        self.velocity_shear: Optional[np.ndarray] = dc_params.velocity_shear
        self.velocity_press: Optional[np.ndarray] = dc_params.velocity_press
        self.densities: Optional[np.ndarray] = dc_params.densities
        self.thicknesses: Optional[np.ndarray] = dc_params.thicknesses
        self.wave_type: str = dc_params.wave_type
        self.velocity_type: str = dc_params.velocity_type
        self.count_threads: int = dc_params.count_threads

    def __add_halfspace(self) -> np.ndarray:
        """
          Adds a half-space layer to the thickness array.

          Returns:
              A NumPy array representing the thicknesses with an additional layer for the half-space.
          """
        thk_with_halfspace = np.zeros((self.thicknesses.shape[0], self.thicknesses.shape[1] + 1))
        thk_with_halfspace[:, :-1] = self.thicknesses
        return thk_with_halfspace

    def __create_velocity_model(self, index: int) -> np.ndarray:
        """
        Creates a velocity model for a specific layer.

        Args:
            index: The index of the layer for which to create the velocity model.

        Returns:
            A NumPy array representing the velocity model for the specified layer.
        """
        thk_with_halfspace = self.__add_halfspace()
        return (
            np.vstack((
                thk_with_halfspace[index],
                self.velocity_press[index],
                self.velocity_shear[index],
                self.densities[index],
            ))
            / 1000
        )

    def __compute_dc(self, index: int) -> np.ndarray:
        """
        Computes the dispersion curve for a specific velocity model.

        Args:
            index: The index of the velocity model to use for the calculation.

        Returns:
            A NumPy array representing the dispersion curve (velocities) for the specified model.
        """
        velocity_model = self.__create_velocity_model(index)
        solver_class = PhaseDispersion if self.velocity_type == "phase" else GroupDispersion
        forward_solver = solver_class(*velocity_model, dc=DC_VALUE)
        disp_curve = np.zeros_like(self.frequencies)
        try:
            velocities = forward_solver(np.sort(1 / self.frequencies), mode=self.num_mode, wave=self.wave_type).velocity
        except DispersionError:
            disp_curve = np.zeros_like(self.frequencies)
            return disp_curve

        if not len(velocities):
            disp_curve = np.zeros_like(self.frequencies)
        else:
            disp_curve[-len(velocities):, ] = velocities[::-1] * 1000
        return disp_curve

    def __compute_dc_mt(self) -> np.ndarray:
        """
        Computes dispersion curves for multiple velocity models using multi-threading.

        Returns:
            A NumPy array representing the dispersion curves for all velocity models.
        """
        count_curves = self.velocity_shear.shape[0]
        return np.array(
            Parallel(n_jobs=self.count_threads)(delayed(self.__compute_dc)(index) for index in range(count_curves))
        )

    def __adapt_model(self) -> None:
        """
        Adapts the velocity model to ensure it's at least 2D.
        """
        self.velocity_shear = np.atleast_2d(self.velocity_shear)
        self.velocity_press = np.atleast_2d(self.velocity_press)
        self.densities = np.atleast_2d(self.densities)
        self.thicknesses = np.atleast_2d(self.thicknesses)

    def run(self) -> np.ndarray:
        """
        Runs the dispersion curve calculation based on the input model dimensions.

        Returns:
            A NumPy array representing the calculated dispersion curve(s).
        """
        if self.velocity_shear.ndim == 1:
            self.__adapt_model()
            return self.__compute_dc(0)

        if self.velocity_shear.ndim == 2:
            if len(self.velocity_shear) == 1:
                return self.__compute_dc(0)
            return self.__compute_dc_mt()

        return np.zeros_like(self.frequencies)
