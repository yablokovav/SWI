import numpy as np

from src.spectral_analysis.models import Spectra
from src.spectral_analysis.peakers.base_peaker import Peaker


class PeakerMax(Peaker):
    """Peaking of dispersion curves by the maximum amplitude."""

    def peak_dc(self, spectra: Spectra, peak_fraction: float, cutoff_fraction: float) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Determines the peaks by the maximum amplitude.

        Finds the peaks from the max amplitude and returns.

        Args:
            spectra (Spectra): Input spectra object.
            peak_fraction (float): peak fraction to determine amplitude.
            cutoff_fraction (float): cutoff fraction to determine amplitude.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing:
                - frees (List[np.ndarray]): Frequencies.
                - dc (List[np.ndarray]): The DC.
                - freq_limits (np.ndarray): The freq limits.
                - upper_limits (np.ndarray): Upper limits.
                - lower_limits (np.ndarray): Lower limits.
        """
        freq, upper_limits, lower_limits, f_indx_spectrum, mask = self.apply_croping(spectra, peak_fraction)
        freq_limits = np.copy(freq)
        dc = np.argmax((spectra.vf_spectra * mask)[:, f_indx_spectrum], axis=0)

        mask_valid_dc = dc > 0
        ampl = spectra.vf_spectra[dc, f_indx_spectrum[mask_valid_dc]]

        return [freq[mask_valid_dc]], [dc[mask_valid_dc]+spectra.velocities[0]], freq_limits, upper_limits, lower_limits, [ampl]
