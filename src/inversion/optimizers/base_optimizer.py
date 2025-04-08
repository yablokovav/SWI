"""Базовый класс для оптимизаторов."""

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

from src.config_reader.models import DispersionCurve, InversionModel, Ranges, PWaveVelocityModel
from src.inversion.computer_dc import ComputerDC, DCModel
from src.inversion.utils import initialize_model
np.complex_ = np.complex128

class BaseOptimizator:
    def __init__(self, inversion_params: InversionModel,
                 dispersion_curves: DispersionCurve,
                 ranges: Ranges,
                 vp_model: PWaveVelocityModel,
                 max_depth) -> None:

        # Parameters initialization
        self.max_num_modes = min([inversion_params.max_num_modes, dispersion_curves.num_modes])
        self.frequencies = dispersion_curves.frequency
        self.velocity_phases = dispersion_curves.velocity_phase

        self.velocity_shear_range = ranges.velocity_shear_range
        self.thicknesses_range = ranges.thicknesses_range

        self.test_count = inversion_params.global_search.test_count
        self.max_iteration = inversion_params.niter
        self.misfit_metric = 'mae'
        self.velocity_type = inversion_params.veltype
        self.wave_type = inversion_params.wavetype
        self.count_agents = None

        # Vp model or Vp/Vs ratio
        self.vp_model = inversion_params.vp_model.value
        self.lock_vp = inversion_params.lock_vp
        self.vp_depth = vp_model.depth
        self.vp = vp_model.vp
        self.vp2vs = vp_model.vp2vs


        ### Mean depth for Vs ranges
        vs_mean_depth = np.r_[0, np.cumsum(np.mean(self.thicknesses_range, axis=1))]
        if self.vp_model == 'vp':
            ### Interpolation Vp on Vs grid
            self.vp = interp1d(self.vp_depth, self.vp, kind='nearest', fill_value='extrapolate')(vs_mean_depth)
            ### Vp correction
            self.vp = np.max(np.c_[self.vp, np.zeros_like(self.vp)+100], axis=1)
            ### Vs ranges correction due to Vp
            self.velocity_shear_range[:,1] = np.min(np.c_[self.velocity_shear_range[:,1], self.vp/1.4], axis=1)
            self.velocity_shear_range[:,0] = np.min(np.c_[self.velocity_shear_range[:,0], self.velocity_shear_range[:,1]-30], axis=1)
            ### Mean Vs from Vs ranges
            vs_mean = np.mean(self.velocity_shear_range, axis=1)
            ### Computing Vp/Vs ratio for Vs mean
            self.vp2vs = self.vp / vs_mean
        else:
            ### Vp/Vs ration interpolation on Vs grid
            self.vp2vs = interp1d(self.vp_depth, self.vp2vs, fill_value='extrapolate')(vs_mean_depth)
            ### Vp/Vs correction
            self.vp2vs = np.max(np.c_[self.vp2vs, np.zeros_like(self.vp2vs)+1.4], axis=1)

    def _generate_models(self) -> list:
        """Запуск инверсий параллельно для каждой модели."""
        return Parallel(n_jobs=-1)(delayed(self._solo_run)() for _ in range(self.test_count))

    def _solo_run(self) -> tuple[np.ndarray, np.ndarray]:
        """Запуск одной итерации."""
        raise NotImplementedError("Дочерние классы должны реализовывать этот метод.")

    def _initialize_models(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Random models initialize"""
        return initialize_model(self.velocity_shear_range, self.thicknesses_range, self.vp2vs, self.count_agents, self.lock_vp)

    def _create_computer_dc(
        self, velocity_shear: np.ndarray, velocity_press: np.ndarray, densities: np.ndarray, thicknesses: np.ndarray
    ) -> ComputerDC:
        """Создание решателя для вычисления дисперсионных кривых."""
        model = DCModel(
            self.wave_type,
            self.velocity_type,
            0,
            self.frequencies[0],
            velocity_shear,
            velocity_press,
            densities,
            thicknesses,
            -1,
        )
        return ComputerDC(model)

    @staticmethod
    def _compute_dispersion_curves(computer_dc: ComputerDC, frequency: np.ndarray, num_mode: int) -> np.ndarray:
        """Вычисление дисперсионных кривых."""
        computer_dc.frequencies = frequency
        computer_dc.num_mode = num_mode
        return computer_dc.run()

    def _check_borders(self, thicknesses: np.ndarray, velocity_shear: np.ndarray) -> None:
        """Проверка выхода за границы параметров."""
        thicknesses[:] = np.clip(thicknesses, self.thicknesses_range[:, 0], self.thicknesses_range[:, 1])
        velocity_shear[:] = np.clip(velocity_shear, self.velocity_shear_range[:, 0], self.velocity_shear_range[:, 1])

    def _update_density_and_velocity(self, velocity_shear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Обновление значений скорости и плотности."""
        if self.vp_model == 'vp':
            if self.lock_vp:
                velocity_press = np.tile(self.vp, (self.count_agents, 1))
            else:
                velocity_press = velocity_shear * self.vp2vs
        else:
            velocity_press = velocity_shear * self.vp2vs
        densities = (1.2475 + 0.3992 * velocity_press * 1e-3 - 0.026 * (velocity_press * 1e-3) ** 2)*1e3
        return velocity_press, densities

    @staticmethod
    def _update_computer_dc(
        computer_dc: ComputerDC,
        velocity_shear: np.ndarray,
        velocity_press: np.ndarray,
        densities: np.ndarray,
        thicknesses: np.ndarray,
    ) -> None:
        """Обновление параметров решателя."""
        computer_dc.velocity_shear = velocity_shear
        computer_dc.velocity_press = velocity_press
        computer_dc.densities = densities
        computer_dc.thicknesses = thicknesses

    def find_median_model(self, models: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        vs = np.array([m[0] for m in models])
        thk = np.array([m[1] for m in models])
        vs_best = np.median(vs, axis=0)
        thk_best = np.median(thk, axis=0)

        return vs_best, thk_best

    def compute_dc_rest(self, vs, thk) -> list:
        if self.vp_model == 'vp':
            if self.lock_vp:
                velocity_press = np.copy(self.vp)
            else:
                velocity_press = vs * self.vp2vs
        else:
            velocity_press = vs * self.vp2vs
        densities = (1.2475 + 0.3992 * velocity_press * 1e-3 - 0.026 * (velocity_press * 1e-3) ** 2)*1e3
        computer_dc = self._create_computer_dc(vs, velocity_press, densities, thk)
        dc_rest = [self._compute_dispersion_curves(computer_dc, self.frequencies[mode_i], mode_i) for mode_i in range(self.max_num_modes)]
        return dc_rest

    def run(self) -> tuple[np.ndarray, np.ndarray, list]:
        """Выполняет многократные запуски инверсии дисперсионных кривых и возвращает лучшие параметры."""
        self.count_agents = 10*(np.size(self.velocity_shear_range, 0) + np.size(self.thicknesses_range, 0))
        models = self._generate_models()
        vs_rest, thk_rest = self.find_median_model(models)
        dc_rest = self.compute_dc_rest(vs_rest, thk_rest)
        return vs_rest, thk_rest, dc_rest
