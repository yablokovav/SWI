import os  # Import the os module for interacting with the operating system
from src.config_reader.Checker.utils import *  # Import all utility functions
from src.config_reader.Checker.Message import Message  # Import the Message class

# Define constants for valid data types
TYPE_DATA_CONSTANTS = ['2d', '3d']
# Define constants for expected preprocessing parameters
PARAMS_PREPROCESSING = [
    'data_dir',
    'path4ffid_file',
    'type_data',
    'offset_min',
    'offset_max',
    'ffid_start',
    'ffid_stop',
    'ffid_increment',
    'num_sources_on_cpu',
    'snr',
    'qc_preprocessing',
    'parameters_3d',
]
# Define constants for expected 2D preprocessing parameters
# Define constants for expected 3D preprocessing parameters
PARAMS_PREPROCESSING_3D = ['sort_3d_order', 'num_sectors', 'bin_size_x', 'bin_size_y']
# Define constants for valid 3D sorting orders
VALID_SORT_3D_ORDER = ['csp', 'cdp']
# Define a constant for a "good" or empty string (used when no errors found)
GOOD_STRING = ''


class PreprocessingChecker:
    """
    Validates the 'preprocessing' section of the configuration.
    """

    def __init__(self, preprocessing_config: dict):
        """
        Initializes the PreprocessingChecker.

        Args:
            preprocessing_config (dict): Configuration for preprocessing.
        """
        self.preprocessing_config = preprocessing_config

    def __check_existence(self, level: str) -> Message:
        """
        Checks if expected parameters exist in the preprocessing config.

        Args:
            level (str): Key to identify this level in the config.

        Returns:
            Message: Result of the existence check.
        """
        return check_level_correct(
            self.preprocessing_config, PARAMS_PREPROCESSING, level
        )

    def __check_data_dir(self, count_mistakes_preprocessing: int, key: str) -> tuple[str, int]:
        """
        Validates the 'data_dir' parameter.

        Checks if the specified directory exists.

        Args:
            count_mistakes_preprocessing (int): Current error count.
            key (str): Key for the 'data_dir' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        data_dir = self.preprocessing_config[key]
        if not os.path.isdir(data_dir):
            error_msg = f" Error: Catalog {data_dir} not found"
            result = (
                str(data_dir)
                + Message(is_error=True, message=error_msg).message
            )
            return result, count_mistakes_preprocessing + 1
        else:
            result = str(data_dir) + Message(
                is_error=False, message=GOOD_STRING
            ).message
            return result, count_mistakes_preprocessing

    def __check_snr(self, count_mistakes_preprocessing: int, key: str) -> tuple[str, int]:
        check_snr = check_parameter_type_and_value(
            self.preprocessing_config[key],
            float,
            [0, None],
            [True, True],
        )
        result = (
            str(self.preprocessing_config[key]) + check_snr.message
        )
        return result, count_mistakes_preprocessing + +check_snr.is_error

    def __check_qc_preprocessing(self, count_mistakes_preprocessing: int, key: str) -> tuple[str, int]:
        """
        Validates the QC preprocessing setting.

        Checks if the value of the qc_preprocessing parameter is bool.

        Args:
            count_mistakes_preprocessing (int): Current error count.
            key (str): Key for the 'qc_preprocessing' parameter

        Returns:
            tuple: A string with the validation message and the updated error count.
        """
        check_qc_preprocessing = check_datatype(
            self.preprocessing_config[key],
            bool
        )
        result = (
            str(self.preprocessing_config[key]) + check_qc_preprocessing.message
        )
        return result, count_mistakes_preprocessing + check_qc_preprocessing.is_error

    def __check_num_sources_on_cpu(self, count_mistakes_preprocessing: int, key: str) -> tuple[str, int]:
        """
        Validates the 'num_sources_on_cpu' parameter.

        Checks if the value is an integer and is greater or equal than 0.

        Args:
            count_mistakes_preprocessing (int): Current error count.
            key (str): Key for the 'num_sources_on_cpu' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_num_sources_on_cpu = check_parameter_type_and_value(
            self.preprocessing_config[key],
            int,
            [0, None], # Range: >= 0
            [False, False] # Inclusive minimum, no maximum bound
        )
        result = (
                str(self.preprocessing_config[key]) + check_num_sources_on_cpu.message
        )
        return result, count_mistakes_preprocessing + check_num_sources_on_cpu.is_error

    def __check_type_data(self, count_mistakes_preprocessing: int, key: str) -> tuple[str, int]:
        """
        Validates the 'type_data' parameter.

        Checks if the value is a valid data type (2d or 3d).

        Args:
            count_mistakes_preprocessing (int): Current error count.
            key (str): Key for the 'type_data' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_type_data = check_valid_string(
            self.preprocessing_config[key], TYPE_DATA_CONSTANTS
        )
        result = (
            str(self.preprocessing_config[key]) + check_type_data.message
        )
        return result, count_mistakes_preprocessing + check_type_data.is_error

    def __check_ffid(self, count_mistakes_preprocessing: int, keys: tuple) -> tuple[str, str, int]:
        """
        Validates the 'ffid_start' and 'ffid_stop' parameters.

        Checks they are floats, greater or equal than 0 and that 'ffid_stop' is greater than 'ffid_start'.

        Args:
            count_mistakes_preprocessing (int): Current error count.
            keys (tuple): Keys for 'ffid_start' and 'ffid_stop'.

        Returns:
            tuple: Validation messages (str) and updated error count (int).
        """
        key_ffid_start, key_ffid_stop = keys  # Unpack the keys

        # Validate 'ffid_start' and 'ffid_stop'
        check_ffid_start_stop = [
            check_parameter_type_and_value(
                self.preprocessing_config[key_ffid_start],
                int,  # Expected type is int
                [0, None],  # Range: >= 0
                [False, False],  # Inclusive minimum, no maximum bound
            ),
            check_parameter_type_and_value(
                self.preprocessing_config[key_ffid_stop],
                int,  # Expected type is int
                [0, None],  # Range: >= 0
                [False, False],  # Inclusive minimum, no maximum bound
            ),
        ]

        if not check_ffid_start_stop[0].is_error and not check_ffid_start_stop[1].is_error:
            # Check if 'ffid_stop' is greater than 'ffid_start'
            check_ffid_start_stop_relationship = check_values_relationship(
                [
                    self.preprocessing_config[key_ffid_start],
                    self.preprocessing_config[key_ffid_stop],
                ],
                [key_ffid_start, key_ffid_stop],
                False,  # Strict comparison: ffid_stop > ffid_start
            )
            return (
                str(self.preprocessing_config[key_ffid_start])
                + check_ffid_start_stop[0].message,
                str(self.preprocessing_config[key_ffid_stop])
                + check_ffid_start_stop_relationship.message,
                count_mistakes_preprocessing
                + check_ffid_start_stop_relationship.is_error,
            )
        else:
            # Handle type validation errors
            return (
                str(self.preprocessing_config[key_ffid_start])
                + check_ffid_start_stop[0].message,
                str(self.preprocessing_config[key_ffid_stop])
                + check_ffid_start_stop[1].message,
                count_mistakes_preprocessing + 1,
            )

    def __check_ffid_increment(self, count_mistakes_preprocessing: int, key: str) -> tuple[str, int]:
        """
        Validates the 'ffid_increment' parameter.

        Checks if the value is an integer and is greater than or equal to than 0.

        Args:
            count_mistakes_preprocessing (int): Current error count.
            key (str): Key for 'ffid_increment'.

        Returns:
            tuple: Validation message (str) and updated error count (int).
        """
        # Validate 'ffid_increment'
        check_ffid_increment = check_parameter_type_and_value(
            self.preprocessing_config[key],
            int,  # Expected type is integer
            [0, None],  # Range: >= 0
            [False, False],  # Inclusive minimum, no maximum bound
        )

        return (
            str(self.preprocessing_config[key]) + check_ffid_increment.message,
            count_mistakes_preprocessing + check_ffid_increment.is_error,
        )

    def __check_path4ffid_file(self, count_mistakes_preprocessing: int, key: str) -> tuple[str, int]:
        """
        Checks the existence of a file specified by the 'path4ffid_file' parameter.

        Args:
            count_mistakes_preprocessing (int): The current count of preprocessing errors.
            key (str): The key for the 'path4ffid_file' parameter.

        Returns:
            tuple[str, int]: A tuple containing the path (or an error message) and the updated error count.
        """
        # If the path is not None
        if self.preprocessing_config[key] is not None:
            # Check if the path exists
            if os.path.exists(self.preprocessing_config[key]):
                # If the file exists, return the path and the current error count
                return str(self.preprocessing_config[key]), count_mistakes_preprocessing
            else:
                # If the file does not exist, return an error message and increment the error count
                error_msg = " Error: file not found"
                return str(self.preprocessing_config[key]) + error_msg, count_mistakes_preprocessing + 1
        else:
            # If the path is None, return None and the current error count
            return str(self.preprocessing_config[key]), count_mistakes_preprocessing

    def __check_offset_min_max(self, count_mistakes_preprocessing: int, keys: tuple) -> tuple[str, str,  int]:
        """
        Validates 'offset_min' and 'offset_max' parameters.

        Checks they float, 'offset_min' greater or equal than 0 and that 'offset_max' is greater than 'offset_min'.

        Args:
            count_mistakes_preprocessing (int): Current error count.
            keys (tuple): Keys for 'offset_min' and 'offset_max'.

        Returns:
            tuple: Validation messages (str) and updated error count (int).
        """
        key_offset_min, key_offset_max = keys  # Unpack the keys

        # Validate 'offset_min' and 'offset_max'
        check_offset_min_max = [
            check_parameter_type_and_value(
                self.preprocessing_config[key_offset_min],
                float,  # Expected type is float
                [0, None],  # Range: >= 0
                [False, False],  # Inclusive minimum, no maximum bound
            ),
            check_parameter_type_and_value(
                self.preprocessing_config[key_offset_max],
                float,  # Expected type is float
                [0, None],  # Range: >= 0
                [True, False],  # Inclusive minimum, no maximum bound
            ),
        ]

        if not check_offset_min_max[0].is_error and not check_offset_min_max[1].is_error:
            # Check if 'offset_max' is greater than or equal to 'offset_min'
            check_offset_min_max_relationship = check_values_relationship(
                [
                    self.preprocessing_config[key_offset_min],
                    self.preprocessing_config[key_offset_max],
                ],
                [key_offset_min, key_offset_max],
                True,  # Non-strict comparison: offset_max >= offset_min
            )
            return (
                str(self.preprocessing_config[key_offset_min])
                + check_offset_min_max[0].message,
                str(self.preprocessing_config[key_offset_max])
                + check_offset_min_max_relationship.message,
                count_mistakes_preprocessing
                + check_offset_min_max_relationship.is_error,
            )
        else:
            # Handle type validation errors
            return (
                str(self.preprocessing_config[key_offset_min])
                + check_offset_min_max[0].message,
                str(self.preprocessing_config[key_offset_max])
                + check_offset_min_max[1].message,
                count_mistakes_preprocessing + 1,
            )


    def __check_parameters_3d(self, count_mistakes_preprocessing: int, level: str) -> tuple[dict, int]:
        """
        Validates the parameters specific to 3D preprocessing configurations
        and updates the preprocessing error count.

        Args:
            count_mistakes_preprocessing (int): The current count of preprocessing errors.
            level (str): The key used to access 3D parameters in the preprocessing configuration.

        Returns:
            tuple: A tuple containing:
                - A dictionary with the validation results for each parameter (str: value + message).
                - An updated count of preprocessing errors (int).

        The function performs validation checks for the following parameters:
        1. 'sort_3d_order': Checks if the value is a valid string within the acceptable sorting order ('csp' and 'cdp').
        2. 'num_sectors': Checks if the value is an integer greater than or equal to 0.
        3. 'bin_size_x': Checks if the value is an integer greater than or equal to 0.
        4. 'bin_size_y': Checks if the value is an integer greater than or equal to 0.

        The error count is updated based on the validation results of each parameter.
        """
        sort_3d_order_ind = 0  # Indices for parameters
        num_sectors_ind = 1
        bin_size_x_ind = 2
        bin_size_y_ind = 3

        key_sort_3d_order = PARAMS_PREPROCESSING_3D[sort_3d_order_ind]  # Keys
        key_num_sector = PARAMS_PREPROCESSING_3D[num_sectors_ind]
        key_bin_size_x = PARAMS_PREPROCESSING_3D[bin_size_x_ind]
        key_bin_size_y = PARAMS_PREPROCESSING_3D[bin_size_y_ind]

        check_parameters_3d = {}  # Store validation results

        # Check if all expected 3D parameters exist
        existence_check = check_level_correct(
            self.preprocessing_config[level],
            PARAMS_PREPROCESSING_3D,
            level
        )

        if existence_check.is_error:
            # Handle existence check error
            check_parameters_3d = existence_check.message
            count_mistakes_preprocessing += 1
        else:
            # Validate parameters if existence check passed

            # Validate 'sort_3d_order'
            check_sort_3d_order = check_valid_string(
                self.preprocessing_config[level][key_sort_3d_order],
                VALID_SORT_3D_ORDER  # Valid sorting orders
            )
            check_parameters_3d[key_sort_3d_order] = (
                    str(self.preprocessing_config[level][key_sort_3d_order])
                    + check_sort_3d_order.message  # Store value and msg
            )
            count_mistakes_preprocessing += check_sort_3d_order.is_error  # Update

            # Validate 'num_sectors'
            check_num_sectors = check_parameter_type_and_value(
                self.preprocessing_config[level][key_num_sector],
                int,  # Expect int type
                [0, None],  # Range: >= 0
                [True, False]  # Inclusive lower bound, exclusive upper
            )
            check_parameters_3d[key_num_sector] = (
                    str(self.preprocessing_config[level][key_num_sector])
                    + check_num_sectors.message
            )
            count_mistakes_preprocessing += check_num_sectors.is_error

            # Validate 'bin_size_x'
            check_bin_size_x = check_parameter_type_and_value(
                self.preprocessing_config[level][key_bin_size_x],
                int,  # Expect int type
                [0, None],  # Range: >= 0
                [True, False]  # Inclusive lower bound, exclusive upper (corrected to be consistent)
            )
            check_parameters_3d[key_bin_size_x] = (
                    str(self.preprocessing_config[level][key_bin_size_x])
                    + check_bin_size_x.message
            )
            count_mistakes_preprocessing += check_bin_size_x.is_error

            # Validate 'bin_size_y'
            check_bin_size_y = check_parameter_type_and_value(
                self.preprocessing_config[level][key_bin_size_y],
                int,  # Expect int type
                [0, None],  # Range: >= 0
                [False, False]  # Inclusive lower bound, exclusive upper (corrected to be consistent)
            )
            check_parameters_3d[key_bin_size_y] = (
                    str(self.preprocessing_config[level][key_bin_size_y])
                    + check_bin_size_y.message
            )
            count_mistakes_preprocessing += check_bin_size_y.is_error

        return check_parameters_3d, count_mistakes_preprocessing

    def check(self) -> tuple[dict, int]:
        """
        Validates the 'preprocessing' configuration section by checking
        the existence, data types, and relationships of parameters.

        Returns:
            tuple: A tuple containing:
                - A dictionary storing validation results for each parameter.
                - An integer representing the total number of errors found.
        """
        preprocessing_errors = {}  # Store validation results for each parameter
        count_mistakes_preprocessing = 0  # Initialize error counter

        # Check if the 'preprocessing' section exists and has mandatory parameters
        check_existence = self.__check_existence('preprocessing')

        if check_existence.is_error:
            # Handle existence check error
            preprocessing_errors = check_existence.message  # Store the error
            count_mistakes_preprocessing += 1  # Increment error counter
        else:
            # Validate individual parameters if existence check passed
            (preprocessing_errors['data_dir'],
             count_mistakes_preprocessing) = self.__check_data_dir(
                count_mistakes_preprocessing,  # Pass current error count
                'data_dir'  # Specify parameter to check
            )

            (preprocessing_errors['type_data'],
             count_mistakes_preprocessing) = self.__check_type_data(
                count_mistakes_preprocessing,  # Pass current error count
                'type_data'  # Specify parameter to check
            )

            (preprocessing_errors['ffid_start'],
             preprocessing_errors['ffid_stop'],
             count_mistakes_preprocessing) = self.__check_ffid(
                count_mistakes_preprocessing,  # Pass current error count
                ('ffid_start', 'ffid_stop')  # Specify parameters to check
            )

            (preprocessing_errors['ffid_increment'],
             count_mistakes_preprocessing) = self.__check_ffid_increment(
                count_mistakes_preprocessing,  # Pass current error count
                'ffid_increment'  # Specify parameter to check
            )
            (preprocessing_errors['path4ffid_file'],
             count_mistakes_preprocessing) = self.__check_path4ffid_file(
                count_mistakes_preprocessing,  # Pass current error count
                'path4ffid_file'  # Specify parameters to check
            )

            (preprocessing_errors['num_sources_on_cpu'],
             count_mistakes_preprocessing) = self.__check_num_sources_on_cpu(
                count_mistakes_preprocessing, # Pass current error count
                'num_sources_on_cpu' # Specify parameters to check
            )

            (preprocessing_errors['offset_min'],
             preprocessing_errors['offset_max'],
             count_mistakes_preprocessing) = self.__check_offset_min_max(
                count_mistakes_preprocessing,  # Pass current error count
                ('offset_min', 'offset_max')  # Specify parameters to check
            )

            (preprocessing_errors['snr'],
             count_mistakes_preprocessing) = self.__check_snr(
                count_mistakes_preprocessing,
                'snr'
            )

            (preprocessing_errors['qc_preprocessing'],
             count_mistakes_preprocessing) = self.__check_qc_preprocessing(
                count_mistakes_preprocessing,
                'qc_preprocessing'
            )

            (preprocessing_errors['parameters_3d'],
             count_mistakes_preprocessing) = self.__check_parameters_3d(
                count_mistakes_preprocessing,  # Pass current error count
                'parameters_3d'  # Specify parameters to check
            )

        return preprocessing_errors, count_mistakes_preprocessing
