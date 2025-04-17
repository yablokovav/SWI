from os import environ
environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import time
from src.preprocessing.preprocessing_and_spectral import SeismicPreprocessorSpectral

from src.config_reader.load_params import ConfigReader
from src.config_reader.models import SWIConfigModel
from src.config_reader.utils import create_directories

root_dir = Path("").resolve()
# params_dir = root_dir / "configs/real_3d/main.yaml"
params_dir = root_dir / "configs/synth_2d/main.yaml"
swi_dir = root_dir / "runs"
preprocessing, spectral, inversion, postprocessing = ConfigReader.read(params_dir, SWIConfigModel, swi_dir, show=False)
module_dirs = create_directories(preprocessing, spectral, inversion, swi_dir)


tic = time.perf_counter()
SeismicPreprocessorSpectral.open(preprocessing, spectral, module_dirs).run()
toc = time.perf_counter()
print(f" {toc - tic:0.4f} seconds")
