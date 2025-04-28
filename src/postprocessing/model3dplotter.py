from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from screeninfo import get_monitors
from src.files_processor.readers import get_filenames
from matplotlib import cm


class Model3DPlotter:
    def __init__(self):
        self.slices = []

    @staticmethod
    def calculate_reduction_factors(vec_1: np.ndarray, vec_2: np.ndarray, vec_3: np.ndarray, 
                                   new_shape_1: int, new_shape_2: int, new_shape_3: int) -> tuple:
        k1 = int(np.ceil(np.size(vec_1) / new_shape_1))
        k2 = int(np.ceil(np.size(vec_2) / new_shape_2))     
        k3 = int(np.ceil(np.size(vec_3) / new_shape_3))
        return k1, k2, k3, vec_1[::k1], vec_2[::k2], vec_3[::k3]

    @staticmethod
    def add_relief(matrix: np.ndarray, depth: np.ndarray,
                 relief: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n1, n2 = np.shape(matrix)
        max_relief, min_relief = np.max(relief), np.min(relief)

        step_depth = depth[1] - depth[0]
        n2_new = n2 + int((max_relief - min_relief) / step_depth)
        matrix_new = np.empty((n1, n2_new))
        matrix_new[:] = np.nan

        vec_2_mesh = np.repeat(depth[None,...], n1, axis=0)
        vec_2_mesh_new = np.empty_like(matrix_new)
        vec_2_mesh_new[:] = np.nan

        for ii in range(n1):
            dmr = int((max_relief - relief[ii]) / step_depth)
            matrix_new[ii, dmr:dmr + n2] = matrix[ii,:]
            vec_2_mesh_new[ii, dmr:dmr + n2] = relief[ii] - vec_2_mesh[ii,:]

        vec_2_new = np.linspace(np.nanmax(vec_2_mesh_new), np.nanmin(vec_2_mesh_new), n2_new)
        return matrix_new, vec_2_new

    @staticmethod
    def prepare_slice(matrix: np.ndarray, vec_1: np.ndarray, vec_2: np.ndarray,
                    vec_3: np.ndarray, slice_plane: str, slice_name: str, relief: np.ndarray) -> go.Surface:

        k1, k2, k3, vec_1, vec_2, vec_3 = Model3DPlotter.calculate_reduction_factors(
            vec_1, vec_2, vec_3, 400, 400, 400
        )

        # print(vec_1, vec_2, vec_3)
        if slice_plane == 'xz':
            matrix, vec_3 = Model3DPlotter.add_relief(matrix, vec_3, relief)
            # print(matrix.shape, vec_3.shape, vec_1.shape, relief.shape)


            x, z = np.meshgrid(vec_1, vec_3)
            # print(x.shape, z.shape, matrix[::k1, ::k3].T.shape)

            return go.Surface(x=x, y=vec_2, z=z, surfacecolor=matrix[::k1, ::k3].T, coloraxis='coloraxis',
                              showscale=False, opacity=0.95, name=slice_name, hoverlabel=dict(namelength=100))

        elif slice_plane == 'yz':
            matrix, vec_3 = Model3DPlotter.add_relief(matrix, vec_3, relief)
            y, z = np.meshgrid(vec_2, vec_3)
            return go.Surface(x=vec_1, y=y, z=z, surfacecolor=matrix[::k2, ::k3].T, coloraxis='coloraxis',
                              showscale=False, opacity=0.95, name=slice_name, hoverlabel=dict(namelength=50))

        elif slice_plane == 'xy':
            # x, y = np.meshgrid(vec_1, vec_2)
            # return go.Surface(x=x, y=y, z=vec_3, surfacecolor=matrix[::k1, ::k2].T, coloraxis='coloraxis',
            #                   showscale=False, opacity=0.95, name=slice_name, hoverlabel=dict(namelength=50))
            return None

    @staticmethod
    def _set_coloraxis(model_vmin, model_vmax) -> dict:
        samples = np.linspace(0.0, 1, 256)
        mapper = cm.ScalarMappable(cmap=plt.get_cmap('RdYlBu_r'))
        rgb_array = [[samples[i].tolist(), 'rgba' + str((r.tolist(), g.tolist(), b.tolist(), a.tolist()))] for
                     i, [r, g, b, a] in enumerate(mapper.to_rgba(samples, bytes=True))]
        rgb_array[0][-1] = "rgba(255, 255, 255, 0.01)"
        return dict(colorscale=rgb_array, cmin=model_vmin, cmax=model_vmax)

    @staticmethod
    def _get_axis_limits(slices) -> tuple:
        min_x, max_x = 1e+30, -1e+30
        min_y, max_y = 1e+30, -1e+30
        min_z, max_z = 1e+30, -1e+30
        for slice in slices:
            if min_x > np.min(slice.x): min_x = np.min(slice.x)
            if max_x < np.max(slice.x): max_x = np.max(slice.x)

            if min_y > np.min(slice.y): min_y = np.min(slice.y)
            if max_y < np.max(slice.y): max_y = np.max(slice.y)

            if min_z > np.min(slice.z): min_z = np.min(slice.z)
            if max_z < np.max(slice.z): max_z = np.max(slice.z)
        return min_x, max_x, min_y, max_y, min_z, max_z

    @staticmethod
    def run(path_to_model: Path, path_to_image: Path, model_vmin: float, model_vmax: float) -> None:

        title_html = "result_in_3d_axes"
        _coloraxis = Model3DPlotter._set_coloraxis(model_vmin, model_vmax)
        slices = []

        if not sum(1 for _ in get_filenames(data_dir=path_to_model, suffix=".npz")):
            raise ValueError('Drawing Error: No files with interpolated velocity models.')

        for file in get_filenames(data_dir=path_to_model, suffix=".npz"):
            res = np.load(file, allow_pickle=True)
            slice = Model3DPlotter.prepare_slice(
                matrix = res["vs"],
                vec_1 = res["x"],
                vec_2 = res["y"],
                vec_3 = res["z"],
                slice_plane = res["projection"],
                slice_name = file.stem,
                relief = res["elevation"]
                )
            if slice is not None:
                slices.append(slice)


        min_x, max_x, min_y, max_y, min_z, max_z = Model3DPlotter._get_axis_limits(slices)
        fig = go.Figure(data=slices)

        try:
            monitor_width = get_monitors()[0].width * 0.8
            monitor_height = get_monitors()[0].height * 0.8
        except:
            monitor_width = 2000
            monitor_height = 1000

        fig.update_layout(
            template="plotly_white",
            width=monitor_width,
            height=monitor_height,
            coloraxis=_coloraxis,
            coloraxis_colorbar=dict(
                title='Vs (м/с)',
                titlefont=dict(size=22, color='black'),
                tickfont=dict(size=16, color='black')
            ),
            scene=dict(
                zaxis=dict(
                    nticks=10,
                    range=[min_z, max_z],
                    title='Z (м)',
                    titlefont=dict(size=22, color='black'),
                    tickfont=dict(size=16, color='black')
                ),
                xaxis=dict(
                    nticks=10,
                    range=[min_x, max_x],
                    title='X (м)',
                    titlefont=dict(size=22, color='black'),
                    tickfont=dict(size=16, color='black')
                ),
                yaxis=dict(
                    nticks=10,
                    range=[min_y, max_y],
                    title='Y (м)',
                    titlefont=dict(size=22, color='black'),
                    tickfont=dict(size=16, color='black')
                ),
                aspectratio=dict(x=1, y=1, z=0.5)
            )
        )
        fig.write_html(path_to_image / f"{title_html}.html")
