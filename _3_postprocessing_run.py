from pathlib import Path

from src.postprocessing.model2dplotter import Model2DPlotter
from src.postprocessing.model3dplotter import Model3DPlotter
from src.postprocessing.postprocessing import VelocityModelVisualizer

from src.config_reader.load_params import ConfigReader
from src.config_reader.models import SWIConfigModel
from src.logs.utils import create_log
from src.config_reader.utils import create_directories
from src.logs.Message import Message


root_dir = Path("").resolve()
params_dir = root_dir / "configs/real_3d/main.yaml"
# params_dir = root_dir / "configs/das/main.yaml"
# params_dir = root_dir / "configs/synth_2d/main.yaml"
swi_dir = root_dir / "runs"
preprocessing, spectral, inversion, postprocessing = ConfigReader.read(params_dir, SWIConfigModel, swi_dir, show=False)
module_dirs = create_directories(preprocessing, spectral, inversion, swi_dir)


log_postprocessing = {}
log_postprocessing['Interpolated Vs-models stored'] = ""
log_postprocessing['Image of Vs-model stored in'] = ""
error = Message(is_error=False, is_warning=False, message="")



for i, indx_inv in enumerate(range(0, len(module_dirs["inversion"]), 2)):
    dir_save_bin, dir_save_image, dir_save_segy = module_dirs["postprocessing"][i * 3: (i + 1) * 3]

    VelocityModelVisualizer(
        dir_save_bin,
        dir_save_segy,
        preprocessing.type_data.value,
        dx = postprocessing.d_x,
        dy = postprocessing.d_y,
        dz = postprocessing.d_z,
        max_depth = postprocessing.max_depth,
        smooth_factor = postprocessing.smooth_factor,
        interp_dim = postprocessing.parameters_2d.interp_dim,
        remove_outliers_smoothing = postprocessing.remove_outliers_smoothing,
        num_xslices_3d = postprocessing.parameters_3d.num_xslices_3d,
        num_yslices_3d = postprocessing.parameters_3d.num_yslices_3d,
        num_zslices_3d = 3,
        error_thr=postprocessing.error_thr,
        save_segy = postprocessing.save_segy,
    ).run(
        module_dirs["inversion"][indx_inv]
    )

    Model2DPlotter.run(
        dir_save_bin,
        dir_save_image,
        model_vmin = postprocessing.vmin_in_model,
        model_vmax = postprocessing.vmax_in_model,
    )

    Model3DPlotter.run(
        dir_save_bin,
        dir_save_image,
        model_vmin = postprocessing.vmin_in_model,
        model_vmax = postprocessing.vmax_in_model,
    )

    log_postprocessing['Interpolated Vs-models stored'] += f"\n{dir_save_segy}"
    log_postprocessing['Image of Vs-model stored in'] += f"\n{dir_save_image}"
log_postprocessing['Interpolated Vs-models stored'] += "\n"
log_postprocessing['Image of Vs-model stored in'] += f"\n"
create_log(log_postprocessing, module_dirs['postprocessing'][0].parents[5], "postprocessing", error)