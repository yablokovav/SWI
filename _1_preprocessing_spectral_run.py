# from os import environ
# environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import time
from mpi4py import MPI

from src.preprocessing.preprocessing_and_spectral import SeismicPreprocessorSpectral

from src.config_reader.load_params import ConfigReader
from src.config_reader.models import SWIConfigModel
from src.config_reader.utils import create_directories
from src.config_reader.Checker.exceptions import InvalidConfigurationParameters

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()

if rank == 0:
    root_dir = Path("").resolve()
    # params_dir = root_dir / "configs/real_3d/main.yaml"
    params_dir = root_dir / "configs/synth_2d/main.yaml"
    swi_dir = root_dir / "runs"
    try:
        preprocessing, spectral, inversion, postprocessing = ConfigReader.read(params_dir, SWIConfigModel, swi_dir, show=False)
        module_dirs = create_directories(preprocessing, spectral, inversion, swi_dir)
        config_data = (preprocessing, spectral, inversion, postprocessing, module_dirs)
        for rank in range(1, size):
            comm.send(config_data, dest=rank, tag=1)
    except InvalidConfigurationParameters as e:
        print(e, flush=True)
        comm.Abort(0)
else:
    preprocessing, spectral, inversion, postprocessing, module_dirs = comm.recv(source=0, tag=1, status=status)

comm.Barrier()

tic = time.perf_counter()
SeismicPreprocessorSpectral.open(preprocessing, spectral, module_dirs).run()
toc = time.perf_counter()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    print(f" {toc - tic:0.4f} seconds")
