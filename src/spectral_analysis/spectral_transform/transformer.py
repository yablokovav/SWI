import numpy as np
from numba import njit, prange

from src.spectral_analysis.models import Spectra


class TransformerFK2VF:
    def __init__(self) -> None:
        self.__wave_numbers_multy = 10

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def __smooth_vf2d(vf_spectra: np.ndarray) -> np.ndarray:
        velocity_count, frequency_count = vf_spectra.shape
        for freq_index in prange(frequency_count):
            for vel_index in range(velocity_count):
                if vf_spectra[vel_index, freq_index] == 0:
                    vf_spectra[vel_index, freq_index] = vf_spectra[vel_index - 1, freq_index]
        return vf_spectra

    def __transform_fk2vf(self, spectra: Spectra) -> None:
        # Продлеваем fk спектр, чтобы иметь большую область заполнения vf спектра
        fk_spectra_stacked = np.hstack([spectra.fk_spectra] * self.__wave_numbers_multy)

        wave_numbers_stacked = np.arange(
            spectra.wave_numbers[2],
            2 * self.__wave_numbers_multy * spectra.wave_numbers.max(),
            spectra.wave_numbers[2],
        )

        # Создаём сетки волновых чисел, частот и скоростей
        wave_number_grid, frequency_grid = np.meshgrid(wave_numbers_stacked, spectra.frequencies)
        velocity_grid = frequency_grid / wave_number_grid

        # Убираем те скорости, которые нам не подходят
        valid_mask = (velocity_grid > spectra.velocities.min()) & (velocity_grid < spectra.velocities.max())

        # Переводим сетку скоросей в индексы, определяем индексы частот и волновых чисел
        velocity_indices = (velocity_grid - spectra.velocities.min()).astype(int)
        frequency_indices, wave_number_indices = np.where(valid_mask)

        # Заполняем элегантно и без циклов
        spectra.vf_spectra[velocity_indices[valid_mask], frequency_indices] = fk_spectra_stacked[
            frequency_indices, wave_number_indices
        ]

    def run(self, spectra: Spectra) -> None:
        self.__transform_fk2vf(spectra)
        spectra.vf_spectra = spectra.vf_spectra / (spectra.vf_spectra.max(axis=0) + 1e-15)
        TransformerFK2VF.__smooth_vf2d(spectra.vf_spectra)
