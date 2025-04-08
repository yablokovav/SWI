import numpy as np
from scipy.interpolate import interp1d

from src.config_reader.models import DispersionCurve, InversionModel, PWaveVelocityModel, Ranges
from src.inversion.computer_dc import ComputerDC, DCModel
from src.inversion.utils import get_misfit, sensitiv

LYMBDA2DEPTH = 3 # коэффициент для пересчета длины волны в глубину для построения начальной модели

class Occam:
    def __init__(self, inversion_params: InversionModel,
                 dispersion_curve_all: DispersionCurve,
                 ranges: Ranges,
                 vp_model: PWaveVelocityModel,
                 max_depth
                 ) -> None:

        # Данные дисперсионных кривых
        self.max_num_modes = min([inversion_params.max_num_modes, dispersion_curve_all.num_modes])
        self.velocity_phases = dispersion_curve_all.velocity_phase
        self.frequencies = dispersion_curve_all.frequency

        # Параметры инверсии
        self.niter = inversion_params.niter
        self.veltype = inversion_params.veltype
        self.wavetype = inversion_params.wavetype
        self.nlay = inversion_params.local_search.nlay
        self.max_depth = max_depth
        self.iteration = 0

        # Модель Vp или отношение Vp/Vs
        self.vp_model = inversion_params.vp_model.value
        self.lock_vp = inversion_params.lock_vp
        self.vp_depth = vp_model.depth
        self.vp = vp_model.vp
        self.vp2vs = vp_model.vp2vs

        # регуляризирующий параметр (линейный закон)
        self.mu_list = np.linspace(1, 1e-5, inversion_params.niter)

        # Объект для расчета дисперсионной кривой
        self.computer_dc = None


    def _create_computer_dc(self) -> None:
        """Создает объект для расчета дисперсионной кривой."""
        dc_config = DCModel(
            wave_type = self.wavetype,
            velocity_type = self.veltype
        )
        self.computer_dc = ComputerDC(dc_config)


    def _compute_new_dc(self, vs: np.ndarray, thk: np.ndarray, frequency: np.ndarray, num_mode: int) -> np.ndarray:
        """
        Вычисляет новую дисперсионную кривую для текущей модели.
        """

        if self.vp_model == 'vp':
            if self.lock_vp:
                vp = np.copy(self.vp)
                vs = np.min(np.c_[vs, self.vp / 1.4], axis=1)
            else:
                vp = vs * self.vp2vs
        else:
            vp = vs * self.vp2vs
        rho = (1.2475 + 0.3992 * vp * 1e-3 - 0.026 * (vp * 1e-3) ** 2) * 1e3

        self.computer_dc.num_mode = num_mode
        self.computer_dc.frequencies = frequency
        self.computer_dc.velocity_shear = vs
        self.computer_dc.thicknesses = thk
        self.computer_dc.velocity_press = vp
        self.computer_dc.densities = rho

        return self.computer_dc.run()

    def _compute_start_model(self) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Вычисляет начальную модель с использованием интерполяции.
        """
        vs_rest, thk_rest = self._dc2initmodel()
        dc_rest = [self._compute_new_dc(vs_rest, thk_rest, self.frequencies[mode_i], mode_i) for mode_i in range(self.max_num_modes)]
        return vs_rest, thk_rest, dc_rest

    def _dc2initmodel(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет начальные скорости и толщины слоев на основе дисперсионной кривой.
        """
        # Расчет начальной глубины и фильтрация по маске
        depth_init = self.velocity_phases[0] / self.frequencies[0] / LYMBDA2DEPTH
        mask = depth_init <= self.max_depth
        depth_init = depth_init[mask]
        VR2VS = np.round(np.exp(-np.linspace(0.2, 2, len(self.velocity_phases[0]))) + 1, 2) ### найти ссылку
        vs_init = (self.velocity_phases[0] * VR2VS)[mask]

        if len(depth_init) == 0:
            depth_init = [self.max_depth]
            vs_init = [(self.velocity_phases[0] * VR2VS)[-1]]

        # Интерполяция скоростей
        depth_grid = np.linspace(depth_init[-1], depth_init[0], self.nlay)
        vs_init = interp1d(depth_init, vs_init, kind="linear")(depth_grid)

        # Интерполяция Vp и Vp2Vs на глубины Vs
        if self.vp_model == 'vp':
            self.vp = interp1d(self.vp_depth, self.vp, kind="linear", fill_value='extrapolate')(depth_grid)
            ### Vp correction
            self.vp = np.max(np.c_[self.vp, np.zeros_like(self.vp)+100], axis=1)
            vs_init = np.min(np.c_[vs_init, self.vp/1.4], axis=1)
            self.vp2vs = self.vp / vs_init
        else:
            self.vp2vs = interp1d(self.vp_depth, self.vp2vs, kind="linear", fill_value='extrapolate')(depth_grid)
            self.vp2vs = np.max(np.c_[self.vp2vs, np.zeros_like(self.vp2vs) + 1.4], axis=1)

        # Расчет толщин слоев
        thk_init = np.diff(np.concatenate(([0], depth_grid)))[:-1]
        return vs_init, thk_init


    def _inversion(
        self, vs_rest: np.ndarray, thickness_curr: np.ndarray, dc_rest: list
    ) -> tuple[np.ndarray, list, list]:
        """
        Основной цикл инверсии.
        """
        vs_init = np.copy(vs_rest)
        delt = self._get_delt(len(vs_rest))
        loss = [get_misfit("mae", self.velocity_phases[0], dc_rest[0])]
        loss_indx = 0

        for self.iteration in range(self.niter):

            for mode_i in range(self.max_num_modes):
                dvr_dvs = self._compute_sensitivity(vs_rest, thickness_curr, self.frequencies[mode_i], mode_i)

                vs_rest_curr = self._compute_new_vs(delt, dvr_dvs, vs_rest, dc_rest[mode_i], self.velocity_phases[mode_i], float(self.mu_list[loss_indx]), mode_i)
                vs_rest_curr = np.clip(vs_rest_curr, np.min(vs_init), np.max(vs_init) )

                dc_rest_curr = self._compute_new_dc(vs_rest_curr, thickness_curr, self.frequencies[mode_i], mode_i)

                loss_curr = get_misfit("mae", self.velocity_phases[mode_i], dc_rest_curr)

                if loss_curr < loss[loss_indx]:
                    loss.append(loss_curr)
                    vs_rest = np.copy(vs_rest_curr)
                    dc_rest[mode_i] = np.copy(dc_rest_curr)
                    loss_indx += 1

                    mape_losses = np.mean(np.abs((loss[loss_indx] - loss[loss_indx-1])/loss[loss_indx]))*100
                    if mape_losses < 3:
                        return vs_rest, dc_rest, loss

        return vs_rest, dc_rest, loss

    def _compute_sensitivity(self, vs: np.ndarray, thk: np.ndarray, frequency: np.ndarray, num_mode: int) -> np.ndarray:

        if self.vp_model == 'vp':
            if self.lock_vp:
                vp = np.copy(self.vp)
                vs = np.min(np.c_[vs, self.vp / 1.4], axis=1)
            else:
                vp = vs * self.vp2vs
        else:
            vp = vs * self.vp2vs
        rho = (1.2475 + 0.3992 * vp * 1e-3 - 0.026 * (vp * 1e-3) ** 2)*1e3

        return sensitiv(num_mode, frequency, vs, vp, rho, thk, wave=self.wavetype, veltype=self.veltype)

    def _get_delt(self, nlay: int) -> np.ndarray:
        """
        Создает матрицу для регуляризации.
        """
        delt = np.eye(nlay)
        delt[1:, :-1] -= np.eye(nlay - 1)
        delt[0, 0] = 0
        return delt

    def _compute_new_vs(
        self, delt: np.ndarray, dvr_dvs: np.ndarray, vs_rest: np.ndarray, dc_rest: np.ndarray, dc_obs: np.ndarray, mu: float, mode_i: int
    ) -> np.ndarray:
        """
        Обновляет скорости сдвига с учетом текущей ошибки.
        """
        weighs = np.int32(dc_rest != 0)
        dd = weighs*(dc_obs - dc_rest)
        A = dvr_dvs.T @ dvr_dvs + mu * delt.T @ delt
        dvs = (np.linalg.pinv(A) @ dvr_dvs.T) @ dd
        vs_rest_tmp = vs_rest + dvs.T

        vs_perturbation = 0.2/(self.iteration+1)
        vs_rest = vs_rest_tmp + vs_rest_tmp*np.random.uniform(-vs_perturbation, vs_perturbation, len(vs_rest_tmp))
        return vs_rest

    def run(self) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Основной метод запуска инверсии.
        Возвращает рассчитанные сдвиговые скорости, толщины слоев и дисперсионную кривую.
        """
        self._create_computer_dc()
        vs_init, thk_init, dc_rest = self._compute_start_model()
        vs_rest, dc_rest, loss = self._inversion(vs_init, thk_init, dc_rest)
        return vs_rest, thk_init, dc_rest

