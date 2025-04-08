"""Модуль инверсии дисперсионных кривых методом SSA."""

import numpy as np

from src.inversion.optimizers.base_optimizer import BaseOptimizator
from src.inversion.utils import get_misfit


class SSA(BaseOptimizator):
    # Основная функция запуска инверсии
    def _solo_run(self) -> tuple[np.ndarray, np.ndarray]:
        # Инициализация моделей и создание решателя
        velocity_shear, thicknesses, velocity_press, densities = self._initialize_models()
        computer_dc = self._create_computer_dc(velocity_shear, velocity_press, densities, thicknesses)
        index_best_model = 0

        for iteration in range(1, self.max_iteration):

            for mode_i in range(self.max_num_modes):
                computed_disp_curves = self._compute_dispersion_curves(computer_dc, self.frequencies[mode_i], mode_i)

                index_best_model = self._calculate_misfit(computed_disp_curves, mode_i)
                c1 = self._calculate_c1(iteration)

                self._update_leaders(c1, velocity_shear, thicknesses, index_best_model)
                self._update_followers(thicknesses, velocity_shear)
                self._check_borders(thicknesses, velocity_shear)

                velocity_press, densities = self._update_density_and_velocity(velocity_shear)
                self._update_computer_dc(computer_dc, velocity_shear, velocity_press, densities, thicknesses)

        return velocity_shear[index_best_model], thicknesses[index_best_model]

    # Вспомогательные функции
    def _calculate_misfit(self, computed_disp_curves: np.ndarray, num_mode: int) -> np.int64:
        """Вычисление невязки и определение лучшей модели."""
        misfit = get_misfit(self.misfit_metric, self.velocity_phases[num_mode], computed_disp_curves)
        return np.argmin(misfit)

    def _calculate_c1(self, iteration: int) -> float:
        """Обновление параметра c1."""
        return 2 * np.exp(-((4 * iteration / self.max_iteration) ** 2))

    def _update_leaders(
        self, c1: float, velocity_shear: np.ndarray, thicknesses: np.ndarray, index_best_model: np.int64
    ) -> None:
        """Обновление параметров первой половины моделей (лидеров)."""
        c2_thk, c3_thk = self._generate_random_parameters(self.thicknesses_range, size=self.count_agents // 2)
        c2_vs, c3_vs = self._generate_random_parameters(self.velocity_shear_range, size=self.count_agents // 2)

        velocity_shear[: self.count_agents // 2] = velocity_shear[index_best_model] + c3_vs * c1 * (
            (self.velocity_shear_range[:, 1] - self.velocity_shear_range[:, 0]) * c2_vs
            + self.velocity_shear_range[:, 0]
        )

        thicknesses[: self.count_agents // 2] = thicknesses[index_best_model] + c3_thk * c1 * (
            (self.thicknesses_range[:, 1] - self.thicknesses_range[:, 0]) * c2_thk + self.thicknesses_range[:, 0]
        )

    @staticmethod
    def _generate_random_parameters(range_values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Генерация случайных параметров для обновления моделей."""
        c2 = np.random.uniform(size=(size, range_values[:, 1].size))
        c3 = np.random.uniform(size=(size, range_values[:, 1].size))
        c3[c3 > 0.5] = 1
        c3[c3 < 0.5] = -1
        return c2, c3.astype(np.int8)

    def _update_followers(self, thicknesses: np.ndarray, velocity_shear: np.ndarray) -> None:
        """Обновление параметров второй половины моделей (последователей)."""
        thicknesses[self.count_agents // 2 :] = (
            thicknesses[self.count_agents // 2 :] + thicknesses[self.count_agents // 2 - 1 : -1]
        ) / 2

        velocity_shear[self.count_agents // 2 :] = (
            velocity_shear[self.count_agents // 2 :] + velocity_shear[self.count_agents // 2 - 1 : -1]
        ) / 2
