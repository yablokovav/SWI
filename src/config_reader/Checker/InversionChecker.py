from src.config_reader.Checker.utils import *  # Import all utility functions
from pathlib import Path

# Define the expected shape of the velocity model
CORRECT_SHAPE_VS = (4, 5)
# Define valid inversion methods
INVERSION_METHOD = ['ssa', 'gwo', 'fcnn', 'occam']
# Define valid wave types
WAVETYPE = ['rayleigh', 'love']
# Define valid velocity types
VELTYPE = ['phase', 'group']
# Define valid methods for getting the number of layers
GET_NUM_LAYERS = ['every', 'mean', 'classes']
# Define valid mu parameters
MU = ['linear', 'exponential']

# Define the expected parameters for the inversion section
INVERSION_PARAMETERS = [
    'inversion_method',
    'niter',
    'wavetype',
    'veltype',
    'path4vp_model',
    'vp_model',
    'global_search',
    'lock_vp',
    'local_search',
    'qc_inversion',
    'max_num_modes'
]

# Define expected parameters for the occam section
LOCAL_SEARCH_PARAMETERS = ['nlay']

# Define expected parameters for the model_ranges section
GLOBAL_SEARCH_PARAMETERS = ['test_count', 'path4vs_limits', 'xi', 'get_num_layers']

# Define the expected headers for the shear wave velocity (Vs) limits file
HEADERS_VS = ['vs_min', 'vs_max', 'h_min', 'h_max', 'vp2vs']

# Define the expected headers for the compressional wave velocity (Vp) model file
HEADER_VP = ['depth', 'vp', 'vp2vs']

# Define the expected headers for the Vp2Vs ratio file
HEADER_VP2VS = ['depth', 'vp2vs']

# Define valid options for the velocity model type
VP_MODEL = ['vp', 'vp2vs']


class InversionChecker:
    """
    Validates the inversion section of the configuration.
    """

    def __init__(self, inversion_config: dict):
        """
        Initializes the InversionChecker.

        Args:
            inversion_config (dict): Configuration for the inversion section.
        """
        self.inversion_config = inversion_config

    def __check_existence(self, level: str) -> Message:
        """
        Checks for the existence of required inversion parameters.

        Args:
            level (str): Level to check in the configuration.

        Returns:
            Message: Result of the existence check.
        """
        return check_level_correct(
            self.inversion_config, INVERSION_PARAMETERS, level
        )

    def __check_inversion_method(self, count_mistakes_inversion: int, key: str) -> tuple[str, int]:
        """
        Validates the 'inversion_method' parameter.

         Checks if the value is a valid string ('ssa', 'gwo', 'fcnn' or 'occam').

        Args:
            count_mistakes_inversion (int): Current error count.
            key (str): Key for 'inversion_method'.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        inversion_method_value = self.inversion_config[key]
        check_inversion_method = check_valid_string(
            inversion_method_value, INVERSION_METHOD
        )
        result = (
            str(inversion_method_value) + check_inversion_method.message
        )
        return result, count_mistakes_inversion + check_inversion_method.is_error

    def __check_niter(self, count_mistakes_inversion: int, key: str) -> tuple[str, int]:
        """
        Validates the 'niter' parameter (number of iterations).

        Check if the value is int and greater than 10 and less than 100.

        Args:
            count_mistakes_inversion (int): Current error count.
            key (str): Key for 'niter'.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_niter = check_parameter_type_and_value(
            self.inversion_config[key],
            int,  # Expect an integer
            [10, 100],  # Valid range: [10, 100]
            [False, False],  # Inclusive on both ends
        )
        result = str(self.inversion_config[key]) + check_niter.message
        return result, count_mistakes_inversion + check_niter.is_error

    def __check_wavetype(self, count_mistakes_inversion: int, key: str) -> tuple[str, int]:
        """
        Validates the 'wavetype' parameter.

        checks if the value is a valid string ('rayleigh' and 'love').

        Args:
            count_mistakes_inversion (int): Current error count.
            key (str): Key for 'wavetype'.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_wavetype = check_valid_string(
            self.inversion_config[key], WAVETYPE
        )
        result = str(self.inversion_config[key]) + check_wavetype.message
        return result, count_mistakes_inversion + check_wavetype.is_error

    def __check_veltype(self, count_mistakes_inversion: int, key: str) -> tuple[str, int]:
        """
        Validates the 'veltype' parameter.

        Checks if the value is a valid string ('phase' and 'group').

        Args:
            count_mistakes_inversion (int): Current error count.
            key (str): Key for 'veltype'.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_veltype = check_valid_string(
            self.inversion_config[key], VELTYPE
        )
        result = str(self.inversion_config[key]) + check_veltype.message
        return result, count_mistakes_inversion + check_veltype.is_error

    def __check_lock_vp(self, count_mistakes_inversion: int, key: str) -> tuple[str, int]:
        """
        Validates the 'lock_vp' parameter.

        Checks if the value is bool.

        Args:
            count_mistakes_inversion (int): Current error count.
            key (str): Key for 'lock_vp'.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        lock_vp_value = self.inversion_config[key]
        check_lock_vp = check_datatype(lock_vp_value, bool)
        result = str(lock_vp_value) + check_lock_vp.message
        return result, count_mistakes_inversion + check_lock_vp.is_error

    def __cehck_qc_inversion(self, count_mistakes_inversion: int, key: str) -> tuple[str, int]:
        cehck_qc_inversion= check_datatype(self.inversion_config[key], bool)
        result = str(self.inversion_config[key]) + cehck_qc_inversion.message
        return result, count_mistakes_inversion + cehck_qc_inversion.is_error

    def __check_start_model(self, count_mistakes_inversion: int, key: str) -> tuple[str, int]:
        """
        Validates the 'path4vp_model' parameter, which specifies the starting
        compressional wave velocity (Vp) model. Ensures that the path is a valid
        CSV file and that its contents conform to the expected format.

        Args:
            count_mistakes_inversion (int): Current error count.
            key (str): Key for the 'path4vp_model' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        model_path = self.inversion_config[key]
        if model_path is not None:  # Check if path exists

            # Validate that the path is a CSV file
            if Path(model_path).suffix == ".csv":

                check_start_model = check_csv_file(model_path)

                if not check_start_model.is_error:  # If the CSV file is valid

                    # Validate the Vp model content
                    check_vp_model = check_vp(model_path, HEADER_VP, 3)

                    result = str(model_path) + check_vp_model.message
                    return result, count_mistakes_inversion + check_vp_model.is_error

                else:  # If the CSV file is invalid

                    result = str(model_path) + check_start_model.message
                    return result, count_mistakes_inversion + 1
            else:
                if not os.path.exists(model_path):  # Check if the file exists
                    result = str(model_path) + " Error: File not found."
                    return result, count_mistakes_inversion + 1
                else:
                    result = str(model_path)
                    return result, count_mistakes_inversion


        else:  # If no path provided
            result = (
                    str(model_path) + " Error: model file is necessary"
            )  # Add error message if the model file is missing
            return result, count_mistakes_inversion + 1

    def __check_vp_model(self, count_mistakes_inversion: int, key: str) -> tuple[str, int]:
        """
        Validates the 'vp_model' parameter

        Check if the value is a valid string ('vp' and 'vp2vs').

        Args:
            count_mistakes_inversion (int): Current error count.
            key (str): Key for the 'vp_model' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        vp_model_value = self.inversion_config[key]
        check_vp_model = check_valid_string(vp_model_value, VP_MODEL)

        result = str(vp_model_value) + check_vp_model.message
        return result, count_mistakes_inversion + check_vp_model.is_error

    def __check_max_num_modes(self, count_mistakes_inversion: int, key: str) -> tuple[str, int]:
        """
        Validates the 'max_num_modes' parameter

        Check if the value is int and greater than 0.

        Args:
            count_mistakes_inversion (int): Current error count.
            key (str): Key for the 'max_num_modes' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_max_num_modes = check_parameter_type_and_value(
            self.inversion_config[key],
            int,
            [1, None],
            [False, False]
        )
        result = str(count_mistakes_inversion) + check_max_num_modes.message
        return result, count_mistakes_inversion + check_max_num_modes.is_error

    def __check_global_search(self, count_mistakes_inversion: int, level: str) -> tuple[dict, int]:
        """
        Validates the 'global_search' configuration parameters and updates the error count.

        Args:
            count_mistakes_inversion (int): The current count of spectral errors.
            level (str): The key for the 'advanced' configuration parameters.

        Returns:
            tuple: A tuple containing:
                - A dictionary with the validation results for each parameter (str: value + message).
                - An updated count of spectral errors (int).

        The function performs validation checks for the following parameters:
        1. 'test_count': Checks if the value is ain integer and greater than 0.
        2. 'path4vs_limits': Checks that the file with model parameters is set and if it is set, it is correct.
        3. 'xi':  Checks if the value is a float greater or equal than 1.2 and less or equal than 5.
        4. 'get_num_layers':  Checks if the value is a valid string
        according to the layer count method.('every', 'mean' and 'classes')
        """
        check_global_search = {}  # Store validation results

        # Define indices
        test_count_ind = 0
        path4vs_limits_ind = 1
        xi_ind = 2
        get_num_layers_ind = 3

        # Get keys
        key_tset_count = GLOBAL_SEARCH_PARAMETERS[test_count_ind]
        key_path4vs_limits = GLOBAL_SEARCH_PARAMETERS[path4vs_limits_ind]
        key_xi = GLOBAL_SEARCH_PARAMETERS[xi_ind]
        key_get_num_layers = GLOBAL_SEARCH_PARAMETERS[get_num_layers_ind]

        # Validate existence of parameters in the 'model_ranges'
        existence_list = check_level_correct(
            self.inversion_config[level], GLOBAL_SEARCH_PARAMETERS, level
        )

        if existence_list.is_error:
            # Handle existence check error
            check_global_search = existence_list.message
            count_mistakes_inversion += 1

        else:
            # Validate path to Vs limits file
            vs_limits_path = self.inversion_config[level][key_path4vs_limits]
            check_vs_file = check_csv_file(vs_limits_path)

            if (
                    not check_vs_file.is_error and vs_limits_path is not None
            ):  # If file is correct
                # Validate the content of the Vs table
                check_vs_table = check_vs(
                    vs_limits_path, HEADERS_VS, 3, HEADERS_VS[:4]
                )

                check_global_search[key_path4vs_limits] = check_vs_table.message
                count_mistakes_inversion += check_vs_table.is_error

            else:  # If file is not correct
                check_global_search[key_path4vs_limits] = (
                    str(vs_limits_path) + check_vs_file.message
                )
                count_mistakes_inversion += check_vs_file.is_error

            # Validate test_count parameter
            check_test_count = check_parameter_type_and_value(
                self.inversion_config[level][key_tset_count],
                int,
                [0, None],
                [True, False]
            )
            check_global_search[key_tset_count] = (
                    str(self.inversion_config[level][key_tset_count]) + check_test_count.message
            )
            count_mistakes_inversion += check_test_count.is_error

            # Validate xi parameter
            check_xi = check_parameter_type_and_value(
                self.inversion_config[level][key_xi],
                float,
                [1.2, 5],
                [False, False]
            )
            check_global_search[key_xi] = (
                    str(self.inversion_config[level][key_xi]) + check_xi.message
            )
            count_mistakes_inversion += check_xi.is_error

            # Validate the 'get_num_layers' parameter
            check_get_num_layers = check_valid_string(
                self.inversion_config[level][key_get_num_layers],
                GET_NUM_LAYERS
            )
            check_global_search[key_get_num_layers] = (
                str(self.inversion_config[level][key_get_num_layers]) + check_get_num_layers.message
            )
            count_mistakes_inversion += check_get_num_layers.is_error

        return check_global_search, count_mistakes_inversion

    def __check_local_search(self, count_mistakes_inversion: int, level: str) -> tuple[dict, int]:
        """
        Validates the 'local_search' configuration parameters and updates the error count.

        Args:
            count_mistakes_inversion (int): The current count of spectral errors.
            level (str): The key for the 'advanced' configuration parameters.

        Returns:
            tuple: A tuple containing:
                - A dictionary with the validation results for each parameter (str: value + message).
                - An updated count of spectral errors (int).

        The function performs validation checks for the following parameters:
        1. 'n_layers': Checks if the value is an integer ond greater than 0.
        """
        check_local_search = {}  # Initialize dictionary for storing validation results
        nlay_ind = 0  # Index for the nlay (number of layers) parameter

        key_nlay = LOCAL_SEARCH_PARAMETERS[nlay_ind]  # Extract the key

        # Check if all parameters for the Occam section exists
        existence_list = check_level_correct(
            self.inversion_config[level], LOCAL_SEARCH_PARAMETERS, level
        )

        if existence_list.is_error:
            # Handle the error for the existence check
            check_local_search = existence_list.message
            count_mistakes_inversion += 1
        else:
            # Check nlay parameter
            check_nlay = check_parameter_type_and_value(
                self.inversion_config[level][key_nlay],
                int,  # Expect an integer
                [0, None],  # The minimum value is 0, there is no maximum.
                [True, False],  # Check both conditions
            )

            # Store result of the validation for nlay parameter
            check_local_search[key_nlay] = (
                    str(self.inversion_config[level][key_nlay]) + check_nlay.message
            )
            count_mistakes_inversion += check_nlay.is_error  # Update count

        return check_local_search, count_mistakes_inversion

    def check(self) -> tuple[dict, int]:
        """
        Main validation method for the 'inversion' section.
        It orchestrates the validation of parameters and returns any errors.

        Returns:
            tuple: A tuple containing:
                - inversion_errors (dict): A dictionary of validation errors, if any.
                - count_mistakes_inversion (int): The total number of validation errors.
        """
        inversion_errors = {}  # Initialize dict to store validation results
        count_mistakes_inversion = 0  # Initialize error count

        # Check existence of required 'inversion' parameters
        existence_list = self.__check_existence('inversion')

        if existence_list.is_error:
            # If existence check fails, store the error message
            inversion_errors = existence_list.message
            count_mistakes_inversion += 1
        else:
            # If existence check succeeds, proceed to validate individual parameters

            # Validate the 'inversion_method'
            (inversion_errors['inversion_method'],
             count_mistakes_inversion) = self.__check_inversion_method(
                count_mistakes_inversion, 'inversion_method'
            )

            # Validate the 'niter'
            (inversion_errors['niter'],
             count_mistakes_inversion) = self.__check_niter(
                count_mistakes_inversion, 'niter'
            )

            # Validate the 'wavetype'
            (inversion_errors['wavetype'],
             count_mistakes_inversion) = self.__check_wavetype(
                count_mistakes_inversion, 'wavetype'
            )

            # Validate the 'veltype'
            (inversion_errors['veltype'],
             count_mistakes_inversion) = self.__check_veltype(
                count_mistakes_inversion, 'veltype'
            )

            (inversion_errors['qc_inversion'],
             count_mistakes_inversion) = self.__cehck_qc_inversion(
                count_mistakes_inversion, 'qc_inversion'
            )

            # Validate the 'path4vp_model'
            (inversion_errors['path4vp_model'],
             count_mistakes_inversion) = self.__check_start_model(
                count_mistakes_inversion, 'path4vp_model'
            )

            # Validate the 'vp_model'
            (inversion_errors['vp_model'],
             count_mistakes_inversion) = self.__check_vp_model(
                count_mistakes_inversion, 'vp_model'
            )

            # Validate the 'lock_vp'
            (inversion_errors['lock_vp'],
             count_mistakes_inversion) = self.__check_lock_vp(
                count_mistakes_inversion, 'lock_vp'
            )

            inversion_errors['max_num_modes'], count_mistakes_inversion = self.__check_max_num_modes(
                count_mistakes_inversion, 'max_num_modes'
            )

            # Validate the 'model_ranges' section
            (inversion_errors['global_search'],
             count_mistakes_inversion) = self.__check_global_search(
                count_mistakes_inversion, 'global_search'
            )

            # Validate the 'occam' section
            (inversion_errors['local_search'],
             count_mistakes_inversion) = self.__check_local_search(
                count_mistakes_inversion, 'local_search'
            )

        return inversion_errors, count_mistakes_inversion
