import unittest
from os import environ
environ['OMP_NUM_THREADS'] = '1'

from src import root_dir
from src.config_reader.load_params import ConfigReader
from src.config_reader.models import SWIConfigModel
from src.config_reader.utils import create_directories
from src.preprocessing.preprocessing_and_spectral import SeismicPreprocessorSpectral
from utils import compare_npz_folders, compare_segy_folders

TEST_PATH_NPZ = root_dir / "unittests/data_test_preprocessing_and_spectral/num_sectors_5/spec_npz"
TEST_PATH_PREP_SEGY = root_dir / "unittests/data_test_preprocessing_and_spectral/num_sectors_5/prep_segy"
TEST_PATH_SPEC_SEGY = root_dir / "unittests/data_test_preprocessing_and_spectral/num_sectors_5/spec_segy"


PARAMS_DIR = root_dir / "unittests/data_test_preprocessing_and_spectral/config/main.yaml"
SWI_DIR = root_dir / "unit_test_data"


class TestPreprocessingAndSpectral(unittest.TestCase):

    def test_preprocessing_and_spectral(self):
        preprocessing, spectral, inversion, postprocessing = ConfigReader.read(PARAMS_DIR,
                                                                               SWIConfigModel,
                                                                               SWI_DIR,
                                                                               show=False)


        module_dirs = create_directories(preprocessing,
                                         spectral,
                                         inversion,
                                         SWI_DIR)

        SeismicPreprocessorSpectral.open(preprocessing, spectral, module_dirs).run()

        self.assertTrue(compare_npz_folders(module_dirs["spectral_analysis"][0], TEST_PATH_NPZ),
                        "Incorrect directory with npz files")
        self.assertTrue(compare_segy_folders(module_dirs["spectral_analysis"][2], TEST_PATH_SPEC_SEGY),
                        "Incorrect directory with spectral segy files")
        self.assertTrue(compare_segy_folders(module_dirs["preprocessing"][0], TEST_PATH_PREP_SEGY),
                        "Incorrect directory with preprocessed segy files")
        print("Function made task correctly")

if __name__ == '__main__':
    unittest.main()










