from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
from segyio import TraceField
from joblib import Parallel, delayed
from scipy.interpolate import griddata, interp1d

from src.config_reader.models import ModelVCR
from src.postprocessing.utils import define_projection, group_files_by_basename, read_curves, robust_smooth_2d
from src.files_processor.savers import save_segy, save_model_to_bin

class VelocityModelVisualizer:
    def __init__(
        self,
        dir_save_bin: Path,
        dir_save_segy: Path,
        data_type: str,
        dx: int,
        dy: int,
        dz: int,
        max_depth: int,
        interp_type: str = "linear",
        interp_dim: str = "1d",
        smooth_factor: int = 10,
        remove_outliers_smoothing = False,
        num_xslices_3d: int = 5,
        num_yslices_3d: int = 5,
        error_thr: float = 0.2,
        save_segy: bool = True,

    ) -> None:
        self.dir_save_bin = dir_save_bin
        self.dir_save_segy = dir_save_segy
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.max_depth = max_depth+dz
        self.interp_type = interp_type
        self.interp_dim = interp_dim
        self.smooth_factor = smooth_factor
        self.data_type = data_type
        self.remove_outliers_smoothing = remove_outliers_smoothing
        self.num_xslices_3d = num_xslices_3d
        self.num_yslices_3d = num_yslices_3d
        self.error_thr = error_thr
        self.save_segy = save_segy

        self.models: Optional[list[np.ndarray]] = None
        self.size_x: Optional[int] = None
        self.size_y: Optional[int] = None
        self.size_z: Optional[int] = None
        self.z: Optional[np.ndarray] = None
        self.x_new: Optional[np.ndarray] = None
        self.y_new: Optional[np.ndarray] = None
        self.z_new: Optional[np.ndarray] = None

        self.depth: Optional[np.ndarray] = None
        self.elevation: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.output_model: Optional[np.ndarray] = None
        self.coord: Optional[np.ndarray] = None
        self.error: Optional[np.ndarray] = None


    def run(self, models_dir) -> None:

        # Remove all files in directories
        [[item.unlink() for item in dir_.glob("*")] for dir_ in [self.dir_save_bin, self.dir_save_segy]]

        files_by_basename = group_files_by_basename(models_dir)
        if not files_by_basename:
            raise ValueError('Interpolation Velocity Models Error: No files with dispersion curves.')

        for key in files_by_basename.keys():
            #Reading all files with 1d models and preparing ModelVCR-dataclass
            models = [ModelVCR.load(file) for file in files_by_basename[key]]

            #Unpacing ModelVCR-dataclass and preparing models, depth, coordinates and relief
            self.depth, self.velocity, self.coord, self.elevation, self.error = read_curves(models, self.max_depth, self.error_thr)

            # Sorting read models
            VelocityModelVisualizer.sorting_models(self)

            #Binning coordinates
            self.coord[:, 0] = np.int32( self.coord[:, 0] / self.dx ) * self.dx + self.dx // 2
            self.coord[:, 1] = np.int32(self.coord[:, 1] / self.dy) * self.dy + self.dy // 2

            # x = self.coord[:, 0]
            # y = self.coord[:, 1]
            # x_diff = np.abs(np.diff(x[np.lexsort([x, y])]))
            # y_diff = np.abs(np.diff(y[np.lexsort([y, x])]))

            # hist, bin_edges_x = np.histogram(x_diff, bins=np.arange(0, self.dx, self.dx//10))
            # fist_5_freq_dx = bin_edges_x[np.argsort(hist)[::-1][:3]]
            # hist, bin_edges_y = np.histogram(y_diff, bins=np.arange(0, self.dy, self.dy//10))
            # fist_5_freq_dy = bin_edges_y[np.argsort(hist)[::-1][:3]]
            # print("Mean dx in CMP:", fist_5_freq_dx, "Mean dy in CMP:", fist_5_freq_dy)
            #
            # fig, ax = plt.subplots(2, 1)
            # ax[0].hist(x_diff, bins=bin_edges_x)
            # ax[0].set_title("Histogram of steps by X")
            # ax[1].hist(y_diff, bins=bin_edges_y)
            # ax[1].set_title("Histogram of steps by Y")
            # fig.tight_layout()
            # fig.savefig("runs/Histogram of most frequently steps by X and Y CMP", dpi=100)

            #Preparing new grid
            num_models, num_layers = self.velocity.shape[0], self.velocity.shape[1]
            self.x_new = np.arange(min(self.coord[:, 0]), max(self.coord[:, 0]), self.dx)
            self.y_new = np.arange(min(self.coord[:, 1]), max(self.coord[:, 1]), self.dy)
            self.z_new = np.arange(0, self.max_depth, self.dz)
            if len(self.x_new)==1 or len(self.y_new)==1 or len(self.z_new)==1:
                raise ValueError('Interpolation Velocity Models Error:  dimension of interpolation too small, choose less dx od dy or dz.')
            self.size_x = len(self.x_new)
            self.size_y = len(self.y_new)
            self.size_z = int(self.max_depth / self.dz)

            # Interpolation all 1d-models to common depth
            VelocityModelVisualizer.interp1d_in_depth(self, num_models, num_layers)
            print("Interpolation by depth is done.")

            # Select 1d-models with the minimum error within bins
            self.coord, indexes, counts = np.unique(self.coord, axis=0, return_index=True, return_counts=True)
            uniq_velocities, uniq_relief = [], []
            for index, count in zip(indexes, counts):
                ind_best_model = index + np.argmin(self.error[index : index + count])
                uniq_velocities.append(self.output_model[:, ind_best_model])
                # uniq_velocities.append(np.nanmean(self.output_model[:, index: index + count], axis=1)) Averaging 1d-models within bins
                uniq_relief.append(np.mean(self.elevation[index: index + count]))
            self.output_model, self.elevation = np.array(uniq_velocities).T, np.array(uniq_relief)


            if  self.data_type == "2d":
                projection = define_projection(self.coord)

                if self.interp_dim == "1d":
                    VelocityModelVisualizer.interd1d(self, num_models, num_layers, projection)
                else:
                    VelocityModelVisualizer.interd2d(self, num_layers, projection)

                #smoothing model
                self.output_model = robust_smooth_2d(self.output_model, s=self.smooth_factor, robust=self.remove_outliers_smoothing)

                #save binary-file of model
                save_model_to_bin(self.dir_save_bin / f'{key}', self.output_model, self.x_new, self.y_new, self.z_new, self.elevation, projection)

            else:
                self.elevation = self.elevation/10000
                # interpolation on regular grid by every depth slice
                VelocityModelVisualizer.interp2d_by_depth_slices(self)

                # Save xz-slices from 3d model in binary files for visualization
                VelocityModelVisualizer.save_model_xz_slices(self, key)

                # Save yz-slices from 3d model in binary files for visualization
                VelocityModelVisualizer.save_model_yz_slices(self, key)

            # save velocity model to segy-file
            if self.save_segy:
                VelocityModelVisualizer.save_model_to_segy(self, key)


    def sorting_models(self):
        sort_indexes = np.lexsort([self.coord[:,0], self.coord[:,1]])
        self.depth = self.depth[sort_indexes]
        self.velocity = self.velocity[sort_indexes]
        self.coord = self.coord[sort_indexes]
        self.elevation = self.elevation[sort_indexes]
        self.error = self.error[sort_indexes]

    def interp1d_in_depth(self, num_models, num_layers):
        self.output_model = np.zeros((self.size_z, num_models))
        index = np.int32(self.depth / self.dz)
        for i in range(num_models):
            for j in range(num_layers - 1):
                self.output_model[index[i, j]: index[i, j + 1], i] = self.velocity[i, j]


    def interd1d(self, num_models, num_layers, projection):

        # Interpolation 1d all models on regular grid by x or y
        if projection == "xz":
            self.output_model = interp1d(self.coord[:, 0], self.output_model, axis=1)(self.x_new)
            self.elevation = interp1d(self.coord[:, 0], self.elevation)(self.x_new)
            self.y_new = interp1d(self.coord[:, 0], self.coord[:, 1])(self.x_new)

        if projection == "yz":
            self.output_model = interp1d(self.coord[:, 1], self.output_model, axis=1)(self.y_new)
            self.elevation = interp1d(self.coord[:, 1], self.elevation)(self.y_new)
            self.x_new = interp1d(self.coord[:, 1], self.coord[:, 0])(self.y_new)


    def interd2d(self, num_layers, projection):

        # Interpolation 2d all models on regular grid by xz or yz
        z = self.depth.flatten()
        vel = self.velocity.flatten()

        if projection == "xz":
            x = np.repeat(self.coord[:, 0], num_layers)
            x_grid, z_grid = np.meshgrid(self.x_new, self.z_new)
            self.output_model = griddata((x, z), vel, (x_grid, z_grid))
            self.elevation = interp1d(self.coord[:, 0], self.elevation)(self.x_new)
            self.y_new = interp1d(self.coord[:, 0], self.coord[:, 1])(self.x_new)

        elif projection == "yz":
            y = np.repeat(self.coord[:, 1], num_layers)
            y_grid, z_grid = np.meshgrid(self.y_new, self.z_new)
            self.output_model = griddata((y, z), vel, (y_grid, z_grid))
            self.elevation = interp1d(self.coord[:, 1], self.elevation)(self.y_new)
            self.x_new = interp1d(self.coord[:, 1], self.coord[:, 0])(self.y_new)


    def _smooth_slice(self, x_grid, y_grid, output_model):
        vs3d_tmp = griddata((self.coord[:, 0], self.coord[:, 1]), output_model, (x_grid, y_grid))
        vs3d_tmp = robust_smooth_2d(vs3d_tmp, s=self.smooth_factor, robust=self.remove_outliers_smoothing)
        return vs3d_tmp

    def interp2d_by_depth_slices(self):
        # interpolation on regular grid by every depth slice
        x_grid, y_grid = np.meshgrid(self.x_new, self.y_new)  # new grid coordinates

        vs3d_tr = Parallel(n_jobs=8)(
            delayed(self._smooth_slice)(x_grid, y_grid, self.output_model[i_depth, :])
            for i_depth in tqdm(range(self.size_z))
        )
        vs3d = np.rollaxis(np.array(vs3d_tr, dtype=np.float32), 0, 3)
        print("Slise interpolation is done")
        # interpolation relief
        self.elevation = griddata((self.coord[:, 0], self.coord[:, 1]), self.elevation, (x_grid, y_grid))
        self.elevation = robust_smooth_2d(self.elevation, s=self.smooth_factor,
                                             robust=self.remove_outliers_smoothing)

        # prepair for segy writing
        self.x_new, self.y_new = x_grid.reshape(self.size_x * self.size_y), y_grid.reshape(self.size_x * self.size_y)
        self.output_model = vs3d.reshape((self.size_x * self.size_y, self.size_z)).T
        self.elevation = np.int32(self.elevation).reshape(self.size_x * self.size_y)


    def save_model_yz_slices(self, key):
        indx_sort = np.lexsort([self.y_new, self.x_new])
        x, y, z = self.x_new[indx_sort], self.y_new[indx_sort], self.z_new
        elevation, model = self.elevation[indx_sort],  self.output_model[:, indx_sort]
        uniq, indexes, counts = np.unique(x, return_index=True, return_counts=True)

        selected_ind_uniq = np.int32(np.linspace(0, len(uniq), self.num_xslices_3d+2))[1:-1]

        uniq, indexes, counts = uniq[selected_ind_uniq], indexes[selected_ind_uniq], counts[selected_ind_uniq]
        for i, (index, count) in enumerate(zip(indexes, counts)):
            indx_slices = [index.tolist() + i for i in range(count)]
            save_model_to_bin(self.dir_save_bin / f'{key}_yz_{uniq[i]}', model[:, indx_slices],
                                                      x[indx_slices],  y[indx_slices], z, elevation[indx_slices],"yz")


    def save_model_xz_slices(self, key):
        indx_sort = np.lexsort([self.x_new, self.y_new])
        x, y, z = self.x_new[indx_sort], self.y_new[indx_sort], self.z_new
        elevation, model = self.elevation[indx_sort],  self.output_model[:, indx_sort]
        uniq, indexes, counts = np.unique(y, return_index=True, return_counts=True)

        selected_ind_uniq = np.int32(np.linspace(0, len(uniq), self.num_yslices_3d+2))[1:-1]


        uniq, indexes, counts = uniq[selected_ind_uniq], indexes[selected_ind_uniq], counts[selected_ind_uniq]
        for i, (index, count) in enumerate(zip(indexes, counts)):
            indx_slices = [index.tolist() + i for i in range(count)]
            save_model_to_bin(self.dir_save_bin / f'{key}_xz_{uniq[i]}', model[:, indx_slices],
                                         x[indx_slices],  y[indx_slices], z, elevation[indx_slices],"xz")





    def save_model_to_segy(self, key):
        save_segy(
            self.dir_save_segy / f'{key}.segy',
            self.output_model, np.vstack((self.x_new, self.y_new, self.elevation)),
            (TraceField.CDP_X, TraceField.CDP_Y, TraceField.ReceiverGroupElevation),
            self.dz / 1000
        )