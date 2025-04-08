"""Модуль для обрисовки 2d моедли поперечных скоростей."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from screeninfo import get_monitors
from src.files_processor.readers import get_filenames

class Model2DPlotter:
    def __init__(self):
        pass

    @staticmethod
    def reduce_matrix(
        matrix: np.ndarray, vec_1: np.ndarray, vec_2: np.ndarray, relief: np.ndarray, new_shape_1: int, new_shape_2: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k1 = int(np.ceil(np.size(vec_1) / new_shape_1))
        k2 = int(np.ceil(np.size(vec_2) / new_shape_2))
        return matrix[::k1, ::k2], vec_1[::k1], vec_2[::k2], relief[::k1]

    @staticmethod
    def add_relief(
        matrix: np.ndarray, depth: np.ndarray, relief: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n1, n2 = np.shape(matrix)

        max_relief, min_relief = np.max(relief), np.min(relief)

        step_depth = depth[1] - depth[0]
        n2_new = n2 + int((max_relief - min_relief) / step_depth)
        matrix_new = np.empty((n1, n2_new))
        matrix_new[:] = np.nan

        vec_2_mesh = np.repeat(depth[None, ...], n1, axis=0)
        vec_2_mesh_new = np.empty_like(matrix_new)
        vec_2_mesh_new[:] = np.nan
        for ii in range(n1):
            dmr = int((max_relief - relief[ii]) / step_depth)
            matrix_new[ii, dmr : dmr + n2] = matrix[ii, :]
            vec_2_mesh_new[ii, dmr : dmr + n2] = relief[ii] - vec_2_mesh[ii, :]

        vec_2_new = np.linspace(np.nanmax(vec_2_mesh_new), np.nanmin(vec_2_mesh_new), n2_new)
        return matrix_new, vec_2_new

    @staticmethod
    def plot2dmodel(
        path_to_model: Path,
        matrix: np.ndarray,
        vec_1: np.ndarray,
        vec_2: np.ndarray,
        slice_plane: str,
        zmin: float,
        zmax: float,
        relief: np.ndarray,
        title_html: str = "",
    ) -> None:
        __dict_22_black = {"size": 22, "color": "black"}
        __dict_18_black = {"size": 18, "color": "black"}

        if slice_plane != "xy":
            matrix, vec_2 = Model2DPlotter.add_relief(matrix, vec_2, relief)

        matrix, vec_1, vec_2, relief = Model2DPlotter.reduce_matrix(matrix, vec_1, vec_2, relief, 400, 200)
        # if (np.min(vec_2) >= 0) and (not len(relief)):
        #     vec_2 *= -1
        fig = go.Figure(
            data=go.Contour(
                x=vec_1,
                y=vec_2,
                z=matrix.T,
                colorscale="RdYlBu_r",
                zmin=zmin,
                zmax=zmax,
                # contours={"coloring": "heatmap", "showlabels": True, "labelfont": {"size": 14, "color": "white"}},
                colorbar={
                    "title": {"text": "Vs (м/с)", "side": "right", "font": __dict_22_black},
                    "tickfont": __dict_18_black,
                },
                ncontours=20,
            )
        )

        try:
            monitor_width = get_monitors()[0].width * 0.55
            monitor_height = get_monitors()[0].height * 0.4
        except:
            monitor_width = 2000
            monitor_height = 1000

        fig.update_layout(width=monitor_width, height=monitor_height, autosize=False)

        if slice_plane == "xz":
            fig.update_layout(
                template="plotly_white",
                xaxis={
                    "nticks": 10,
                    "title": "X (м)",
                    "titlefont": __dict_22_black,
                    "tickfont": __dict_18_black,
                    "zeroline": True,
                },
                yaxis={
                    "nticks": 10,
                    "title": "Z (м)",
                    "titlefont": __dict_22_black,
                    "tickfont": __dict_18_black,
                    "zeroline": True,
                },
            )

        if slice_plane == "yz":
            fig.update_layout(
                template="plotly_white",
                xaxis={
                    "nticks": 10,
                    "title": "Y (м)",
                    "titlefont": __dict_22_black,
                    "tickfont": __dict_18_black,
                    "zeroline": True,
                },
                yaxis={
                    "nticks": 10,
                    "title": "Z (м)",
                    "titlefont": __dict_22_black,
                    "tickfont": __dict_18_black,
                    "zeroline": True,
                },
            )
        if slice_plane == "xy":
            fig.update_layout(
                template="plotly_white",
                xaxis={
                    "nticks": 10,
                    "title": "X (м)",
                    "titlefont": __dict_22_black,
                    "tickfont": __dict_18_black,
                    "zeroline": True,
                },
                yaxis={
                    "nticks": 10,
                    "title": "Y (м)",
                    "titlefont": __dict_22_black,
                    "tickfont": __dict_18_black,
                    "zeroline": True,
                },
            )
        if title_html != "":
            fig.write_html(path_to_model / f"{title_html}.html")

    @staticmethod
    def run(
        path_to_model: Path, path_to_image: Path, model_vmin: float, model_vmax: float
    ) -> None:

        [[item.unlink() for item in dir_.glob("*")] for dir_ in [path_to_image]]


        if not sum(1 for _ in get_filenames(data_dir=path_to_model, suffix=".npz")):
            raise ValueError('Drawing Error: No files with interpolated velocity models.')

        for file in get_filenames(data_dir=path_to_model, suffix=".npz"):
            print("file:", file)

            res = np.load(file, allow_pickle=True)
            v, x, y, z, elevation, projection = (
                res["vs"],
                res["x"],
                res["y"],
                res["z"],
                res["elevation"],
                res["projection"]
            )


            vec1 = None
            vec2 = None
            if projection == "xy":
                vec1 = x
                vec2 = y
            elif projection == "xz":
                vec1 = x
                vec2 = z
            elif projection == "yz":
                vec1 = y
                vec2 = z

            Model2DPlotter.plot2dmodel(
                path_to_image,
                v,
                vec1,
                vec2,
                projection,
                model_vmin,
                model_vmax,
                relief=elevation,
                title_html=file.stem + "_in_2d_axes",
            )
