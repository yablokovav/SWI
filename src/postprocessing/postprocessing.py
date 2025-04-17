from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
from segyio import TraceField
from joblib import Parallel, delayed
from scipy.interpolate import griddata, interp1d

from src.config_reader.models import ModelVCR
from src.postprocessing import utils
from src.files_processor.savers import save_segy, save_model_to_bin, write_fdm

class VelocityModelVisualizer:
    def __init__(
        self,
        dir_save_bin: Path,
        dir_save_segy: Path,
        dir_save_fdm: Path,
        data_type: str,
        dx: int,
        dy: int,
        dz: int,
        max_depth: int,
        interp_type: str = "linear",
        interp_dim: str = "1d",
        smooth_factor: int = 10,
        remove_outliers_smoothing = False,
        fill_missing_values = True,
        num_xslices_3d: int = 5,
        num_yslices_3d: int = 5,
        num_zslices_3d: int = 5,
        error_thr: float = 0.2,
        save_segy: bool = True,
        save_fdm: bool = True,

    ) -> None:
        """
        A class for processing, interpolating, visualizing, and exporting 3D velocity models.

        This class handles various preprocessing tasks such as interpolation, smoothing,
        outlier removal, and missing value filling. It also supports exporting the resulting
        velocity models to binary, SEGY, and FDM formats, and allows visualization of
        cross-sectional slices in all three dimensions.

        Args:
            dir_save_bin (Path): Directory to save binary output files.
            dir_save_segy (Path): Directory to save SEGY output files.
            dir_save_fdm (Path): Directory to save FDM format files.
            data_type (str): Type of input data to process ( 2d', '3d').
            dx (int): Spatial step size in the X direction.
            dy (int): Spatial step size in the Y direction.
            dz (int): Spatial step size in the Z direction (depth).
            max_depth (int): Maximum depth to process (dz will be added internally).
            interp_type (str): Type of interpolation ('linear', 'cubic', etc.).
            interp_dim (str): Interpolation dimension ('1d' or '2d').
            smooth_factor (int): Factor for smoothing the model.
            remove_outliers_smoothing (bool): Flag for processing outliers.
            fill_missing_values (bool): Whether to fill missing values in the data.
            num_xslices_3d (int): Number of cross-sections to visualize along the X axis.
            num_yslices_3d (int): Number of cross-sections to visualize along the Y axis.
            num_zslices_3d (int): Number of cross-sections to visualize along the Z axis.
            error_thr (float): Threshold for acceptable MAPE metric for dispersion curves.
            save_segy (bool): Whether to save output in SEGY format.
            save_fdm (bool): Whether to save output in FDM format.

        Internal State:
            models (Optional[list[np.ndarray]]): List of input velocity models.
            size_x (Optional[int]): Grid size in the X direction.
            size_y (Optional[int]): Grid size in the Y direction.
            size_z (Optional[int]): Grid size in the Z direction.
            z, x_new, y_new, z_new (Optional[np.ndarray]): Grids for interpolation and modeling.
            depth (Optional[np.ndarray]): Depth data array.
            elevation (Optional[np.ndarray]): Elevation data array.
            velocity (Optional[np.ndarray]): Raw velocity data.
            output_model (Optional[np.ndarray]): Final processed velocity model.
            coord (Optional[np.ndarray]): Coordinate array for the model.
            error (Optional[np.ndarray]): Inversion error map.

        """
        self.dir_save_bin = dir_save_bin
        self.dir_save_segy = dir_save_segy
        self.dir_save_fdm = dir_save_fdm
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.max_depth = max_depth+dz
        self.interp_type = interp_type
        self.interp_dim = interp_dim
        self.smooth_factor = smooth_factor
        self.data_type = data_type
        self.remove_outliers_smoothing = remove_outliers_smoothing
        self.fill_missing_values = fill_missing_values
        self.num_xslices_3d = num_xslices_3d
        self.num_yslices_3d = num_yslices_3d
        self.num_zslices_3d = num_zslices_3d
        self.error_thr = error_thr
        self.save_segy = save_segy
        self.save_fdm = save_fdm

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
        """
        Main pipeline for processing and interpolating velocity models.

        This method performs the following steps:
            1. Groups velocity model files by basename.
            2. Loads and parses 1D velocity models from npz files.
            3. Sorts and preprocesses the data (including depth binning and coordinate alignment).
            4. Interpolates all models to a common depth axis.
            5. Averages models within each spatial bin.
            6. Depending on the mode (`data_type`), performs:
               - 2D interpolation and smoothes along 'xz' or 'yz' planes (with optional smoothing), or
               - 3D interpolation and smoothes slice-by-slice.
            7. Saves output models to binary, SEGY, and FDM formats.
            8. Exports cross-sectional slices (XZ, YZ, XY) for visualization.

        Args:
            models_dir (Path or str): Path to directory containing 1D velocity models.

        Raises:
            ValueError: If no models are found, or if the interpolation grid is invalid.
        """

        # Remove all files in directories
        # [[item.unlink() for item in dir_.glob("*")] for dir_ in [self.dir_save_bin, self.dir_save_segy, self.dir_save_fdm]]

        files_by_basename = utils.group_files_by_basename(models_dir)
        if not files_by_basename:
            raise ValueError('Interpolation Velocity Models Error: No files with dispersion curves.')

        for key in files_by_basename.keys():
            # Reading all files with 1d models and preparing ModelVCR-dataclass
            models = [ModelVCR.load(file) for file in files_by_basename[key]]

            # Unpacing ModelVCR-dataclass and preparing models, depth, coordinates and relief
            self.depth, self.velocity, self.coord, self.elevation, self.error = utils.read_curves(models, self.max_depth, self.error_thr)

            # Sorting read models
            VelocityModelVisualizer.sorting_models(self)

            # Binning coordinates
            self.coord[:, 0] = np.int32(self.coord[:, 0] / self.dx) * self.dx + self.dx / 2
            self.coord[:, 1] = np.int32(self.coord[:, 1] / self.dy) * self.dy + self.dy / 2

            self.z_new = np.arange(0, self.max_depth, self.dz)
            self.size_z = int(np.ceil(self.max_depth / self.dz))
            # Interpolation all 1d-models to common depth
            num_models, num_layers = self.velocity.shape[0], self.velocity.shape[1]
            VelocityModelVisualizer.interp1d_in_depth(self, num_models, num_layers)
            print("Interpolation by depth is done.")

            # Averages the velocity models that fall in the specified bin cell
            self.coord, self.output_model, self.elevation = utils.average_models_in_bin(self.coord, self.error,
                                                                                        self.output_model,
                                                                                        self.elevation)
            # Preparing new grid
            self.x_new = np.arange(min(self.coord[:, 0]), max(self.coord[:, 0]) + self.dx/2, self.dx)
            self.y_new = np.arange(min(self.coord[:, 1]), max(self.coord[:, 1]) + self.dy/2, self.dy)
            self.size_x = len(self.x_new)
            self.size_y = len(self.y_new)
            if (len(self.x_new)==1 or len(self.y_new)==1 or len(self.z_new)==1) and self.data_type == '3d':
                raise ValueError('Interpolation Velocity Models Error:  dimension of interpolation too small, choose less dx od dy or dz.')

            if  self.data_type == "2d":
                projection = utils.define_projection(self.coord)

                if self.interp_dim == "1d":
                    VelocityModelVisualizer.interd1d(self, projection)
                else:
                    VelocityModelVisualizer.interd2d(self, num_layers, projection)
                #smoothing model
                if self.smooth_factor != 0:
                    self.output_model = utils.robust_smooth_2d(self.output_model, s=self.smooth_factor, robust=self.remove_outliers_smoothing)

                #save binary-file of model
                save_model_to_bin(self.dir_save_bin / f'{key}', self.output_model, self.x_new, self.y_new, self.z_new, self.elevation, projection)

            else:
                # interpolation on regular grid by every depth slice
                VelocityModelVisualizer.interp2d_by_depth_slices(self)

                # Save xz-slices from 3d model in binary files for visualization
                VelocityModelVisualizer.save_model_xz_slices(self, key)

                # Save yz-slices from 3d model in binary files for visualization
                VelocityModelVisualizer.save_model_yz_slices(self, key)

                # Save yz-slices from 3d model in binary files for visualization
                VelocityModelVisualizer.save_model_xy_slices(self, key)

            # save velocity model to segy-file
            if self.save_segy:
                VelocityModelVisualizer.save_model_to_segy(self, key)

            if self.save_fdm:
                VelocityModelVisualizer.save_model_to_fdm(self, key)


    def sorting_models(self):
        """
        Sort all model arrays based on spatial coordinates.

        This method ensures consistent ordering by sorting all relevant arrays
        (`depth`, `velocity`, `coord`, `elevation`, `error`) according to
        lexicographical order of X and Y coordinates.
        """
        sort_indexes = np.lexsort([self.coord[:,0], self.coord[:,1]])
        self.depth = self.depth[sort_indexes]
        self.velocity = self.velocity[sort_indexes]
        self.coord = self.coord[sort_indexes]
        self.elevation = self.elevation[sort_indexes]
        self.error = self.error[sort_indexes]

    def interp1d_in_depth(self, num_models, num_layers):
        """
        Perform 1D interpolation along depth for each velocity model.

        Interpolates velocity values between given depth intervals and stores
        the results in a regular grid in `self.output_model`.

        Args:
            num_models (int): Number of individual velocity profiles (models).
            num_layers (int): Number of depth layers per model.
        """
        self.output_model = np.zeros((self.size_z, num_models))
        index = np.int32(self.depth / self.dz)
        for i in range(num_models):
            for j in range(num_layers - 1):
                self.output_model[index[i, j]: index[i, j + 1], i] = self.velocity[i, j]


    def interd1d(self, projection):
        """
        Perform 1D interpolation in the horizontal plane along X or Y axis.

        Applies interpolation to each vertical column of the model and
        realigns data onto a regular grid based on `x_new` or `y_new`.

        Args:
            projection (str): Projection direction, either 'xz' or 'yz'.
        """
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
        """
        Perform 2D interpolation of the model in either XZ or YZ plane.

        Flattens the velocity and depth arrays and interpolates onto a regular grid
        using `griddata`. Updates `output_model` and horizontal coordinate mappings.

        Args:
            num_layers (int): Number of depth layers per model (used for replication).
            projection (str): Plane of interpolation, either 'xz' or 'yz'.
        """
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
        """
        Interpolate and optionally smooth a single depth slice in XY space.

        Args:
            x_grid (np.ndarray): 2D grid of X coordinates.
            y_grid (np.ndarray): 2D grid of Y coordinates.
            output_model (np.ndarray): Interpolated and smoothed slise for one depth.

        Returns:
            np.ndarray: Interpolated and smoothed 2D slice.
        """
        vs3d_tmp = griddata((self.coord[:, 0], self.coord[:, 1]), output_model, (x_grid, y_grid))
        mask_nan = np.argwhere(np.isnan(vs3d_tmp))
        if self.smooth_factor != 0:
            vs3d_tmp = utils.robust_smooth_2d(vs3d_tmp, s=self.smooth_factor, robust=self.remove_outliers_smoothing)
        if not self.fill_missing_values:
            vs3d_tmp[mask_nan[:, 0], mask_nan[:, 1]] = np.nan
        return vs3d_tmp

    def interp2d_by_depth_slices(self):
        """
        Performs 2D interpolation on each depth slice of the 3D velocity model.

        For each Z-level, interpolates the velocity data onto a regular XY grid using the
        `_smooth_slice()` helper method (e.g., smoothing splines or other interpolation).
        Also interpolates and smooths the elevation data.

        Updates:
            - self.output_model: reshaped, interpolated and smoothened model (shape: Z x (X*Y))
            - self.elevation: flattened and smoothed elevation array
        """
        # interpolation on regular grid by every depth slice
        x_grid, y_grid = np.meshgrid(self.x_new, self.y_new)  # new grid coordinates

        vs3d_tr = Parallel(n_jobs=-1)(
            delayed(self._smooth_slice)(x_grid, y_grid, self.output_model[i_depth, :])
            for i_depth in tqdm(range(self.size_z))
        )
        vs3d = np.rollaxis(np.array(vs3d_tr, dtype=np.float32), 0, 3)
        print("Slise interpolation is done")
        # interpolation relief
        self.elevation = griddata((self.coord[:, 0], self.coord[:, 1]), self.elevation, (x_grid, y_grid))
        self.elevation = utils.robust_smooth_2d(self.elevation, s=None, robust=self.remove_outliers_smoothing)

        # prepair for segy writing
        self.x_new, self.y_new = x_grid.reshape(self.size_x * self.size_y), y_grid.reshape(self.size_x * self.size_y)
        self.output_model = vs3d.reshape((self.size_x * self.size_y, self.size_z)).T
        self.elevation = np.int32(self.elevation).reshape(self.size_x * self.size_y)
        self.elevation[np.argwhere(np.isnan(self.output_model[:, 0]))] = np.nan

    def save_model_yz_slices(self, key):
        """
        Save a series of vertical YZ slices through the 3D velocity model.

        The method selects a number of equally spaced X positions (based on `num_xslices_3d`)
        and extracts vertical YZ slices at those locations. Each slice is saved in binary format
        using `save_model_to_bin`.

        Args:
            key (str): Unique identifier for naming the output slice files.
        """
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
        """
        Save a series of vertical XZ slices through the 3D velocity model.

        The method selects a number of equally spaced Y positions (based on `num_yslices_3d`)
        and extracts vertical XZ slices at those locations. Each slice is saved in binary format
        using `save_model_to_bin`.

        Args:
            key (str): Unique identifier for naming the output slice files.
        """
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

    def save_model_xy_slices(self, key):
        """
        Save a series of horizontal XY slices through the 3D velocity model.

        The method selects a number of equally spaced Z-levels (depths)
        and extracts horizontal slices of the model at those depths.
        Each slice is saved in binary format using `save_model_to_bin`.

        Args:
            key (str): Unique identifier for naming the output slice files.
        """
        vs3d = self.output_model.reshape((self.size_z, self.size_y, self.size_x))
        x = np.arange(min(self.coord[:, 0]), max(self.coord[:, 0]), self.dx)
        y = np.arange(min(self.coord[:, 1]), max(self.coord[:, 1]), self.dy)

        selected_ind = np.int32(np.linspace(0, len(self.z_new), self.num_zslices_3d+2))[1:-1]
        for index in selected_ind:
            save_model_to_bin(self.dir_save_bin / f'{key}_xy_{self.z_new[index]}',  vs3d[index, :, :],
                                         x, y, self.z_new, self.elevation,"xy")

    def save_model_to_segy(self, key):
        """
        Export the velocity model to a SEGY file.

        The model is flattened and saved using 3D spatial coordinates: X, Y, and elevation.
        The SEGY format allows integration with standard seismic software and workflows.

        Args:
            key (str): Unique identifier for the SEGY file name.
        """
        save_segy(
            self.dir_save_segy / f'{key}.segy',
            self.output_model, np.vstack((self.x_new, self.y_new, self.elevation)),
            (TraceField.CDP_X, TraceField.CDP_Y, TraceField.ReceiverGroupElevation),
            self.dz / 1000
        )

    def save_model_to_fdm(self, key):
        """
        Save the processed velocity model to an FDM (Finite Difference Method) file.

        This method reshapes the internal `output_model` into a 3D volume using the defined
        dimensions and writes it to disk in FDM format using the provided parameters.
        The output file will be saved in the `dir_save_fdm` directory with the name `{key}.fdm`.

        Args:
            key (str): A unique identifier for the output file name.

        Raises:
            ValueError: If `output_model` is None or any of the size attributes are not set.
        """
        write_fdm(
            self.dir_save_fdm / f'{key}.fdm',
            self.output_model.reshape((self.size_x, self.size_y, self.size_z)),
            self.size_x,
            self.size_y,
            self.size_z,
            self.dx,
            self.dy,
            self.dz,
            cinc = 1,
            sinc = 1,
            dist_unit = 1,
            angle_unit = 2,
            north_angle = np.pi,
            rot_angle = 0,
            utm_x = self.x_new[0],
            utm_y = self.x_new[0],
        )