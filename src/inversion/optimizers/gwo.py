"""Модуль инверсии дисперсионных кривых методом GWO."""

import numpy as np

from src.inversion.optimizers.base_optimizer import BaseOptimizator
from src.inversion.utils import get_misfit


class GWO(BaseOptimizator):
    # Основная функция запуска инверсии
    def _solo_run(self) -> tuple[np.ndarray, np.ndarray]:
        # Инициализация моделей и создание решателя
        velocity_shear, thicknesses, velocity_press, densities = self._initialize_models()
        computer_dc = self._create_computer_dc(velocity_shear, velocity_press, densities, thicknesses)

        # Первоначальное вычисление кривых и невязки
        mode_i = 0
        computed_disp_curves = self._compute_dispersion_curves(computer_dc, self.frequencies[mode_i], mode_i)
        alpha_index, beta_index, delta_index = self._calculate_misfit(computed_disp_curves, mode_i)

        # Итерационный процесс оптимизации
        for iteration in range(self.max_iteration):

            for mode_i in range(self.max_num_modes):
                param_iter = 2 * (1 - iteration / self.max_iteration)

                # Обновление velocity_shear и thicknesses с помощью вспомогательной функции
                self._update_parameter(velocity_shear, alpha_index, beta_index, delta_index, param_iter)
                self._update_parameter(thicknesses, alpha_index, beta_index, delta_index, param_iter)

                # Проверка границ значений
                self._check_borders(thicknesses, velocity_shear)

                # Обновление параметров и решателя
                velocity_press, densities = self._update_density_and_velocity(velocity_shear)
                self._update_computer_dc(computer_dc, velocity_shear, velocity_press, densities, thicknesses)

                # Вычисление новых кривых и невязки
                computed_disp_curves = self._compute_dispersion_curves(computer_dc, self.frequencies[mode_i], mode_i)
                alpha_index, beta_index, delta_index = self._calculate_misfit(computed_disp_curves, mode_i)

        return velocity_shear[alpha_index], thicknesses[alpha_index]

    # Вспомогательные функции
    def _calculate_misfit(self, computed_disp_curves: np.ndarray, num_mode: int) -> tuple[int, int, int]:
        """Вычисление невязки и индексов моделей."""
        misfit = get_misfit(self.misfit_metric, self.velocity_phases[num_mode], computed_disp_curves)
        alpha_index, beta_index, delta_index = np.argsort(misfit)[:3]
        return alpha_index, beta_index, delta_index

    def _update_parameter(
        self, parameter: np.ndarray, alpha_index: int, beta_index: int, delta_index: int, param_iter: float
    ) -> None:
        """Обновление одного параметра (velocity_shear или thicknesses)."""
        param_c = np.random.uniform(0, 2, (self.count_agents, 3, parameter.shape[1]))
        param_a = np.random.uniform(-param_iter, param_iter, (self.count_agents, 3, parameter.shape[1]))

        d_alpha = np.abs(param_c[:, 0, :] * parameter[alpha_index] - parameter)
        d_beta = np.abs(param_c[:, 1, :] * parameter[beta_index] - parameter)
        d_delta = np.abs(param_c[:, 2, :] * parameter[delta_index] - parameter)

        x_1 = parameter[alpha_index] - param_a[:, 0, :] * d_alpha
        x_2 = parameter[beta_index] - param_a[:, 1, :] * d_beta
        x_3 = parameter[delta_index] - param_a[:, 2, :] * d_delta

        parameter[:] = (x_1 + x_2 + x_3) / 3
