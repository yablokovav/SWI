from src.config_reader.Checker.utils import *  # Import all utilities

# Define valid spectral methods
SPECTRAL_METHOD = ['fkt', 'sfk']
# Define valid methods for extracting DC components
EXTRACT_DC_METHOD = ['max', 'ae', 'dbscan']
# Define the expected shape for DC correction
CORRECT_SHAPE_DC = (6, 3)
# Define expected parameters for the spectral section
SPECTRAL_PARAMETERS = [
    'spectral_method',
    'extract_dc_method',
    'fmin',
    'fmax',
    'vmin',
    'vmax',
    'qc_spectral',
    'advanced',
    'path4dc_limits',
]
# Define expected parameters for advanced settings
ADVANCED_PARAMETERS = [
    'desired_nt',
    'desired_nx',
    'smooth_data',
    'width',
    'peak_fraction',
    'cutoff_fraction',
    'dc_error_thr'
]
# Define headers for DC component data
HEADERS_DC = [
    'freq',
    'v_min',
    'v_max'
]


class SpectralChecker:
    """
    Validates the spectral section of the configuration.
    """

    def __init__(self, spectral_config: dict):
        """
        Initializes the SpectralChecker.

        Args:
            spectral_config (dict): Configuration for the spectral section.
        """
        self.spectral_config = spectral_config

    def __check_existence(self, level: str) -> Message:
        """
        Checks for the existence of required spectral parameters.

        Args:
            level (str): Level to check in the config.

        Returns:
            Message: Result of the existence check.
        """
        return check_level_correct(
            self.spectral_config, SPECTRAL_PARAMETERS, level
        )

    def __check_spectral_method(self, count_mistakes_spectral: int, key: str) -> tuple[str, int]:
        """
        Validates the 'spectral_method' parameter.

        Checks if the value is a valid data type ('sfk' or 'fkt').

        Args:
            count_mistakes_spectral (int): Current error count.
            key (str): Key for 'spectral_method'.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_spectral_method = check_valid_string(
            self.spectral_config[key], SPECTRAL_METHOD
        )
        result = (
            str(self.spectral_config[key]) + check_spectral_method.message
        )
        return result, count_mistakes_spectral + check_spectral_method.is_error

    def __check_extract_dc_method(self, count_mistakes_spectral: int, key: str) -> tuple[str, int]:
        """
        Validates the 'extract_dc_method' parameter.

        Checks if the value is a valid data type ('max', 'ae' or 'dbscan').

        Args:
            count_mistakes_spectral (int): Current error count.
            key (str): Key for 'extract_dc_method'.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        extract_dc_method_value = self.spectral_config[key]
        check_extract_dc_method = check_valid_string(
            extract_dc_method_value, EXTRACT_DC_METHOD
        )
        result = (
            str(extract_dc_method_value) + check_extract_dc_method.message
        )
        return result, count_mistakes_spectral + check_extract_dc_method.is_error

    def __check_f_min_max(self, count_mistakes_spectral: int, keys: tuple) -> tuple[str, str, int]:
        """
        Validates 'fmin' and 'fmax' (minimum and maximum frequency).

        Checks they are floats, 'fmin' greater or equal than 0 and that 'fmax' is greater than 'fmin'.

        Args:
            count_mistakes_spectral (int): Current error count.
            keys (tuple): Keys for 'fmin' and 'fmax'.

        Returns:
            tuple: Validation results (str) and updated error count (int).
        """
        key_fmin, key_fmax = keys  # Unpack keys

        check_f_min_max = [
            check_parameter_type_and_value(
                self.spectral_config[key_fmin],
                float,  # Expected float type
                [0, None],  # Min 0, no max
                [False, False],  # Inclusive min, no exclusive max
            ),
            check_parameter_type_and_value(
                self.spectral_config[key_fmax],
                float,  # Expected float type
                [0, None],  # Min 0, no max
                [True, False],  # Inclusive min, no exclusive max
            ),
        ]

        if not check_f_min_max[0].is_error and not check_f_min_max[1].is_error:
            check_f_min_max_relationship = check_values_relationship(
                [
                    self.spectral_config[key_fmin],
                    self.spectral_config[key_fmax],
                ],
                [key_fmin, key_fmax],
                True,  # strict=True means fmax must be >= fmin
            )
            result = (
                str(self.spectral_config[key_fmin]) + check_f_min_max[0].message,
                str(self.spectral_config[key_fmax])
                + check_f_min_max_relationship.message,
                count_mistakes_spectral
                + check_f_min_max_relationship.is_error,
            )
            return result
        else:
            result = (
                str(self.spectral_config[key_fmin]) + check_f_min_max[0].message,
                str(self.spectral_config[key_fmax]) + check_f_min_max[1].message,
                count_mistakes_spectral + 1,
            )
            return result

    def __check_v_min_max(self, count_mistakes_spectral: int, keys: tuple) -> tuple[str, str, int]:
        """
        Validates 'vmin' and 'vmax' (minimum and maximum velocity).

        Checks they are floats, greater than 0 and that 'vmax' is greater than 'vmin'.

        Args:
            count_mistakes_spectral (int): Current error count.
            keys (tuple): Keys for 'vmin' and 'vmax'.

        Returns:
            tuple: Validation results (str) and updated error count (int).
        """
        key_vmin, key_vmax = keys  # Unpack keys

        check_v_min_max = [
            check_parameter_type_and_value(
                self.spectral_config[key_vmin],
                float,  # Expected float type
                [0, None],  # Min 0, no max
                [True, False],  # Inclusive min, no exclusive max
            ),
            check_parameter_type_and_value(
                self.spectral_config[key_vmax],
                float,  # Expected float type
                [0, None],  # Min 0, no max
                [True, False],  # Inclusive min, no exclusive max
            ),
        ]

        if not check_v_min_max[0].is_error and not check_v_min_max[1].is_error:
            check_v_min_max_relationship = check_values_relationship(
                [
                    self.spectral_config[key_vmin],
                    self.spectral_config[key_vmax],
                ],
                [key_vmin, key_vmax],
                True,  # strict=True means vmax must be >= vmin
            )
            result = (
                str(self.spectral_config[key_vmin]) + check_v_min_max[0].message,
                str(self.spectral_config[key_vmax])
                + check_v_min_max_relationship.message,
                count_mistakes_spectral
                + check_v_min_max_relationship.is_error,
            )
            return result
        else:
            result = (
                str(self.spectral_config[key_vmin]) + check_v_min_max[0].message,
                str(self.spectral_config[key_vmax]) + check_v_min_max[1].message,
                count_mistakes_spectral + 1,
            )
            return result

    def __check_qc_spectral(self, count_mistakes_spectral: int, key: str) -> tuple[str, int]:
        check_qc_spectral = check_datatype(
            self.spectral_config[key],
            bool
        )
        result = (
            str(self.spectral_config[key]) + check_qc_spectral.message
        )
        return result, count_mistakes_spectral + check_qc_spectral.is_error

    def __check_path4dc_limits(self, count_mistakes_spectral: int, key: str) -> tuple[str, int]:
        """
        Validates the 'path4dc_limits' parameter, ensuring it's a valid CSV file
        and that the data within conforms to the expected format.

        Args:
            count_mistakes_spectral (int): Current spectral error count.
            key (str): Key for the 'path4dc_limits' parameter.

        Returns:
            tuple: Validation results (str) and updated error count (int).
        """
        check_path4dc_limits = check_csv_file(
            self.spectral_config[key]
        )  # Check if it is a valid CSV file
        # If the path is valid and not None
        if not check_path4dc_limits.is_error and self.spectral_config[key] is not None:
            # Validate contents of the DC table using check_dc function
            check_dc_table = check_dc(
                self.spectral_config[key],  # Path to the file
                HEADERS_DC,  # Expected headers for the DC table
                3,  # Number of expected columns
                HEADERS_DC[1:],  # Expected headers for the data columns
            )
            result = (
                    str(self.spectral_config[key]) + check_dc_table.message
            )  # Create the results string

            return result, count_mistakes_spectral + check_dc_table.is_error  # Update and return the error count

        else:  # If the file path is invalid
            result = (
                    str(self.spectral_config[key]) + check_path4dc_limits.message
            )  # Create the results string
            return result, count_mistakes_spectral + 1  # Update and return the error count

    def __check_advanced(self, count_mistakes_spectral: int, level: str) -> tuple[dict, int]:
        """
        Validates the 'advanced' configuration parameters and updates the error count.

        Args:
            count_mistakes_spectral (int): The current count of spectral errors.
            level (str): The key for the 'advanced' configuration parameters.

        Returns:
            tuple: A tuple containing:
                - A dictionary with the validation results for each parameter (str: value + message).
                - An updated count of spectral errors (int).

        The function performs validation checks for the following parameters:
        1. 'desired_nt':  Checks if the value is an integer greater than 0 or equal than 0 and less or equal than 5000.
        2. 'desired_nx':  Checks if the value is an integer greater than 0 or equal than 0 and less or equal than 1000.
        3. 'smooth_data':  Checks if bool.
        4. 'width':  Checks if the value is a float and greater than 0 and less or equal than 1000.
        5. 'peak_fraction':  Checks if the value is a float and greater than 0 and less than 1.
        6. 'cutoff_fraction':  Checks if the value is a float and greater than 0 and less than 1.
        7. 'dc_error_thr':  Checks if the value is a float and greater than 0 and less than 1.
        """
        check_advanced = {}  # Initialize the dict to store the validation checks
        desired_nt_ind = 0  # Define indices for the parameters
        desired_nx_ind = 1
        smooth_data_ind = 2
        width_ind = 3
        peak_fraction_ind = 4
        cutoff_fraction_ind = 5
        dc_error_thr_ind = 6

        key_desired_nt = ADVANCED_PARAMETERS[desired_nt_ind]  # Define keys
        key_desired_nx = ADVANCED_PARAMETERS[desired_nx_ind]
        key_smooth_data = ADVANCED_PARAMETERS[smooth_data_ind]
        key_peak_fraction = ADVANCED_PARAMETERS[peak_fraction_ind]
        key_cutoff_fraction = ADVANCED_PARAMETERS[cutoff_fraction_ind]
        key_dc_error_thr = ADVANCED_PARAMETERS[dc_error_thr_ind]
        key_width = ADVANCED_PARAMETERS[width_ind]

        # Validate the existence of the advanced parameters
        existence_list = check_level_correct(
            self.spectral_config[level], ADVANCED_PARAMETERS, level
        )

        if existence_list.is_error:  # If the existence check failed
            check_advanced = existence_list.message  # Store the error message
            count_mistakes_spectral += 1  # Increment the error count
        else:  # If the existence check succeeded

            # Validate desired_nt (expected type: int, range: [0, 5000])
            check_desired_nt = check_parameter_type_and_value(
                self.spectral_config[level][key_desired_nt],
                int,
                [0, 5000],
                [True, False],
            )
            check_advanced[key_desired_nt] = (
                    str(self.spectral_config[level][key_desired_nt])
                    + check_desired_nt.message  # Combine value and message
            )
            count_mistakes_spectral += check_desired_nt.is_error  # Update counter

            # Validate desired_nx (expected type: int, range: [0, 1000])
            check_desired_nx = check_parameter_type_and_value(
                self.spectral_config[level][key_desired_nx],
                int,
                [0, 1000],
                [True, False],
            )
            check_advanced[key_desired_nx] = (
                    str(self.spectral_config[level][key_desired_nx])
                    + check_desired_nx.message  # Combine value and message
            )
            count_mistakes_spectral += check_desired_nx.is_error  # Update counter

            # Validate smooth_data (expected type: bool)
            check_smooth_data = check_datatype(
                self.spectral_config[level][key_smooth_data], bool
            )
            check_advanced[key_smooth_data] = (
                    str(self.spectral_config[level][key_smooth_data])
                    + check_smooth_data.message  # Combine value and message
            )
            count_mistakes_spectral += check_smooth_data.is_error  # Update counter

            # Validate width (expected type: float, range: [0, 100])
            check_width = check_parameter_type_and_value(
                self.spectral_config[level][key_width],
                float,
                [0, 100],
                [False, True],
            )
            check_advanced[key_width] = (
                    str(self.spectral_config[level][key_width])
                    + check_width.message  # Combine value and message
            )
            count_mistakes_spectral += check_width.is_error  # Update counter

            check_peak_fraction = check_parameter_type_and_value(
                self.spectral_config[level][key_peak_fraction],
                float,
                [0, 1],
                [True, True]
            )
            check_advanced[key_peak_fraction] = (
                str(self.spectral_config[level][key_peak_fraction]) +
                check_peak_fraction.message
            )
            count_mistakes_spectral += check_peak_fraction.is_error

            check_cutoff_fraction = check_parameter_type_and_value(
                self.spectral_config[level][key_cutoff_fraction],
                float,
                [0, 1],
                [True, True]
            )
            check_advanced[key_cutoff_fraction] = (
                str(self.spectral_config[level][key_cutoff_fraction]) +
                check_cutoff_fraction.message
            )
            count_mistakes_spectral += check_cutoff_fraction.is_error

            check_dc_error_thr = check_parameter_type_and_value(
                self.spectral_config[level][key_dc_error_thr],
                float,
                [0, 1],
                [True, True]
            )
            check_advanced[key_dc_error_thr] = (
                str(self.spectral_config[level][key_dc_error_thr]) +
                check_dc_error_thr.message
            )
            count_mistakes_spectral += check_dc_error_thr.is_error

        return check_advanced, count_mistakes_spectral  # Return the validation result and mistake count

    def check(self) -> tuple[dict, int]:
        """
        Validates the spectral section of the configuration by checking
        parameter existence, data types, and relationships.

        Returns:
            tuple: A tuple containing:
                - spectral_errors (dict): Validation results for each parameter.
                - count_mistakes_spectral (int): Total number of errors found.
        """
        spectral_errors = {}  # Store validation results for each parameter
        count_mistakes_spectral = 0  # Initialize error counter

        # Check if the 'spectral' section exists and has mandatory parameters
        existence_list = self.__check_existence('spectral')

        if existence_list.is_error:
            # Handle existence check error
            spectral_errors = existence_list.message  # Store error message
            count_mistakes_spectral += 1  # Increment error count
        else:
            # Validate parameters if existence check passed

            # Validate 'spectral_method' parameter
            (spectral_errors['spectral_method'],
             count_mistakes_spectral) = self.__check_spectral_method(
                count_mistakes_spectral,  # Current error count
                'spectral_method',  # Parameter to check
            )

            # Validate 'extract_dc_method' parameter
            (spectral_errors['extract_dc_method'],
             count_mistakes_spectral) = self.__check_extract_dc_method(
                count_mistakes_spectral,  # Current error count
                'extract_dc_method',  # Parameter to check
            )

            # Validate 'fmin' and 'fmax' parameters
            (spectral_errors['fmin'],
             spectral_errors['fmax'],
             count_mistakes_spectral) = self.__check_f_min_max(
                count_mistakes_spectral,  # Current error count
                ('fmin', 'fmax'),  # Parameters to check
            )

            # Validate 'vmin' and 'vmax' parameters
            (spectral_errors['vmin'],
             spectral_errors['vmax'],
             count_mistakes_spectral) = self.__check_v_min_max(
                count_mistakes_spectral,  # Current error count
                ('vmin', 'vmax'),  # Parameters to check
            )

            # Validate 'qc_spectral' parameter
            (spectral_errors['qc_spectral'],
             count_mistakes_spectral) = self.__check_qc_spectral(
                count_mistakes_spectral,
                'qc_spectral'
            )

            # Validate 'path4dc_limits' parameter
            (spectral_errors['path4dc_limits'],
             count_mistakes_spectral) = self.__check_path4dc_limits(
                count_mistakes_spectral,  # Current error count
                'path4dc_limits',  # Parameter to check
            )

            # Validate 'advanced' section
            (spectral_errors['advanced'],
             count_mistakes_spectral) = self.__check_advanced(
                count_mistakes_spectral,  # Current error count
                'advanced',  # Parameter to check
            )

        return spectral_errors, count_mistakes_spectral
