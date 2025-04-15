import hdbscan
import numpy as np
from matplotlib import pyplot as plt

from src.spectral_analysis.models import Spectra
from src.spectral_analysis.peakers.base_peaker import Peaker
np.random.seed(42)


class PeakerHDBSCAN(Peaker):
    """Пикировка дисперсионных кривых с использованием DBSCAN."""

    @staticmethod
    def _segments_fit(freq, vel):
        """
        Gets dispersion curve from data of cluster.

        Args:
            freq (np.ndarray): The frequency data.
            vel (np.ndarray): The velocity data.

        Returns:
             unique_freq (np.ndarray): The dispersion curve's frequency.
             unique_vel (np.ndarray): The dispersion curve's velocities.
        """
        unique_freq, indices = np.unique(freq, return_inverse=True)
        unique_vel = np.bincount(indices, weights=vel) / np.bincount(indices)
        return unique_freq, unique_vel

    @staticmethod
    def _remove_outbreaks(mask: np.ndarray, fcount: int, vcount: int, cutoff_fraction: float):
        """
        Removes outliers using HDBSCAN clustering.

        Uses DBSCAN to reject data by using a mask.

        Args:
            mask (np.ndarray): Input mask for outlier removal.
            fcount (int): Frequency count.
            vcount (int): Velocity count.
            cutoff_fraction (float): Threshold for outlier rejection.

        Returns:
             freq (List[np.ndarray]): The frequencies data by clusters.
             vel (List[np.ndarray]): The velocities data by clusters.
        """

        labels = hdbscan.HDBSCAN(min_samples=4,
                                 cluster_selection_epsilon=2,
                                 ).fit(mask).labels_
        count_points_in_all_clusters = np.bincount(labels[labels >= 0])
        indexes_valid_clusters = np.where(count_points_in_all_clusters >= count_points_in_all_clusters.max() * cutoff_fraction)[0]
        all_valid_clusters = [mask[labels == i] for i in indexes_valid_clusters]

        max_count_freq = max([len(np.unique(tmp_cluster[:, 1])) for tmp_cluster in all_valid_clusters])
        valid_clusters = []
        for tmp_cluster in all_valid_clusters:
            count_freq = len(np.unique(tmp_cluster[:, 1]))
            if count_freq >= max_count_freq * cutoff_fraction:
                valid_clusters.append(tmp_cluster)

        metric = np.array([(np.mean(tmp_cluster[:, 1] ) / fcount  ) ** 2 + (np.mean(tmp_cluster[:, 0]) / vcount ) ** 2 for tmp_cluster in valid_clusters])
        sort_indexes = np.argsort(metric)
        valid_clusters =[valid_clusters[i] for i in sort_indexes]

        vel, freq =[], []
        for tmp_cluster in valid_clusters:
            vel.append(tmp_cluster[:, 0])
            freq.append(tmp_cluster[:, 1])

            # print("tmp_cluster shape:", tmp_cluster.shape)
        return freq, vel

    @staticmethod
    def _get_curve(vf_spectra:np.ndarray, data: np.ndarray, fcount: int, vcount: int, vmin: float, cutoff_fraction, shift: int) -> tuple[list, list, list]:
        """
        Gets the dispersion curves.

        Gets dispersion curves by using hdbscan clustering.

        Args:
            data (np.ndarray): Input mask for curve extraction.
            fcount (int): Frequency count.
            vcount (int): Velocity count.
            cutoff_fraction (float): Cutoff fraction for outlier rejection.

        Returns:
             px (List[np.ndarray]): The frequencies of dispersion curves.
             py (List[np.ndarray]): The velocities  of dispersion curves.
        """
        nonzero_coord = np.column_stack(np.nonzero(data))  # (vel, freq)
        freq, vel = PeakerHDBSCAN._remove_outbreaks(nonzero_coord, fcount, vcount, cutoff_fraction)  # (vel, freq)
        px, py, ampl = [], [], []
        for mode in zip(freq, vel):
            px_tmp, py_tmp = PeakerHDBSCAN._segments_fit(mode[0], mode[1])
            px.append(px_tmp - shift)
            py.append(py_tmp + vmin)
            ampl.append(vf_spectra[np.int32(py_tmp), np.int32(px_tmp)])
        return px, py, ampl

    def peak_dc(self, spectra: Spectra, peak_fraction: float, cutoff_fraction: float) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray]:
        """
        Peaks the dispersion curve.

        Peaks data with the hdbscan clustering.

        Args:
            spectra (Spectra): Input mask for outlier removal.
            peak_fraction (float): peaker parameter.
            cutoff_fraction (float): cutoff parameter.

        Returns:
             freq_shifted (List[np.ndarray]): Freq with shift.
             py (List[np.ndarray]): The vel data.
             freq_limits (np.ndarray): The Freq limits.
             lower_limits (np.ndarray): Lower limits for peak finding.
             upper_limits (np.ndarray): Upper limits for peak finding.
        """
        freq, upper_limits, lower_limits, f_indx_spectrum, mask = self.apply_croping(spectra, peak_fraction)
        freq_limits = np.copy(freq)
        px, py, ampl = PeakerHDBSCAN._get_curve(
            spectra.vf_spectra,
            mask,
            len(spectra.frequencies),
            len(spectra.velocities),
            float(spectra.velocities[0]),
            cutoff_fraction,
            f_indx_spectrum[0]
        )
        freq_all = [freq[px_tmp] for px_tmp in px]
        return freq_all, py, freq_limits, lower_limits, upper_limits, ampl
