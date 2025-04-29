import unittest
from os import environ
environ['OMP_NUM_THREADS'] = '1'

from src import root_dir
from src.config_reader.utils import create_directories
from src.preprocessing.preprocessing_and_spectral import SeismicPreprocessorSpectral
from utils import compare_npz_folders, compare_segy_folders, get_params
from mpi4py import MPI

TEST_PATH_NPZ_CSP = root_dir / "unittests/data_test_preprocessing_and_spectral/num_sectors_5/spec_npz"
TEST_PATH_PREP_SEGY_CSP = root_dir / "unittests/data_test_preprocessing_and_spectral/num_sectors_5/prep_segy"
TEST_PATH_SPEC_SEGY_CSP = root_dir / "unittests/data_test_preprocessing_and_spectral/num_sectors_5/spec_segy"

TEST_PATH_NPZ_CDP = root_dir / "unittests/data_test_preprocessing_and_spectral/cdp/spec_npz"
TEST_PATH_PREP_SEGY_CDP = root_dir / "unittests/data_test_preprocessing_and_spectral/cdp/prep_segy"
TEST_PATH_SPEC_SEGY_CDP = root_dir / "unittests/data_test_preprocessing_and_spectral/cdp/spec_segy"


PARAMS_DIR_CSP = root_dir / "unittests/data_test_preprocessing_and_spectral/config/main_csp.yaml"
PARAMS_DIR_CDP = root_dir / "unittests/data_test_preprocessing_and_spectral/config/main_cdp.yaml"
SWI_DIR = root_dir / "unit_test_data"


class TestPreprocessingAndSpectral(unittest.TestCase):

    def test_preprocessing_and_spectral_csp(self):
        preprocessing, spectral, inversion, postprocessing = get_params(
            PARAMS_DIR_CSP, SWI_DIR
        )

        module_dirs = create_directories(preprocessing,
                                         spectral,
                                         inversion,
                                         SWI_DIR)

        SeismicPreprocessorSpectral.open(preprocessing, spectral, module_dirs).run()

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            self.assertTrue(compare_npz_folders(module_dirs["spectral_analysis"][0], TEST_PATH_NPZ_CSP),
                            "Incorrect directory with npz files")
            self.assertTrue(compare_segy_folders(module_dirs["spectral_analysis"][2], TEST_PATH_SPEC_SEGY_CSP),
                            "Incorrect directory with spectral segy files")
            self.assertTrue(compare_segy_folders(module_dirs["preprocessing"][0], TEST_PATH_PREP_SEGY_CSP),
                            "Incorrect directory with preprocessed segy files")
            print("Function made task correctly")

    def test_preprocessing_and_spectral_cdp(self):
        preprocessing, spectral, inversion, postprocessing = get_params(
            PARAMS_DIR_CDP, SWI_DIR
        )

        module_dirs = create_directories(preprocessing,
                                         spectral,
                                         inversion,
                                         SWI_DIR)

        SeismicPreprocessorSpectral.open(preprocessing, spectral, module_dirs).run()

        self.assertTrue(compare_npz_folders(module_dirs["spectral_analysis"][0], TEST_PATH_NPZ_CDP),
                        "Incorrect directory with npz files")
        self.assertTrue(compare_segy_folders(module_dirs["spectral_analysis"][2], TEST_PATH_SPEC_SEGY_CDP),
                        "Incorrect directory with spectral segy files")
        self.assertTrue(compare_segy_folders(module_dirs["preprocessing"][0], TEST_PATH_PREP_SEGY_CDP),
                        "Incorrect directory with preprocessed segy files")
        print("Function made task correctly")

if __name__ == '__main__':
    unittest.main()
