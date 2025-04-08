from src.config_reader.Checker.utils import *  # Import all utility functions


POSTPROCESSING_PARAMETERS = [
    'max_depth',
    'd_x',
    'd_y',
    'd_z',
    'smooth_factor',
    'remove_outliers_smoothing',
    'vmin_in_model',
    'vmax_in_model',
    'save_segy',
    'error_thr',
    'parameters_2d',
    'parameters_3d'
]
POSTPROCESSING_2D_PARAMETERS = [
    'interp_dim'
]
POSTPROCESSING_3D_PARAMETERS = [
    'num_xslices_3d',
    'num_yslices_3d'
]
INTRRP_DIM = [
    '1d',
    '2d'
]

class PostprocessingChecker:
    """
    Validates the inversion section of the configuration.
    """

    def __init__(self, postprocessing_config: dict):
        """
        Initializes the PostprocessingChecker.

        Args:
            postprocessing_config (dict): Configuration for the postprocessing section.
        """
        self.postprocessing_config = postprocessing_config

    def __check_existence(self, level: str) -> Message:
        """
        Checks for the existence of required inversion parameters.

        Args:
            level (str): Level to check in the configuration.

        Returns:
            Message: Result of the existence check.
        """
        return check_level_correct(
            self.postprocessing_config, POSTPROCESSING_PARAMETERS, level
        )

    def __check_max_depth(self, count_mistakes_postprocessing: int, key: str):
        """
        Validates the 'max_depth' parameter.

        Check if value is float and greater than 0.

        Args:
            count_mistakes_postprocessing (int): Current error count.
            key (str): The key for the 'max_depth' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_max_depth = check_parameter_type_and_value(
            self.postprocessing_config[key],
            float,
            [0, None],
            [True, False]
        )
        result = str(self.postprocessing_config[key]) + check_max_depth.message
        return result, count_mistakes_postprocessing + check_max_depth.is_error

    def __check_d_x(self, count_mistakes_postprocessing: int, key: str):
        """
        Validates the 'd_x' parameter.

        Check if value is int and greater than 0.

        Args:
            count_mistakes_postprocessing (int): Current error count.
            key (str): The key for the 'd_x' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_d_x = check_parameter_type_and_value(
            self.postprocessing_config[key],
            int,
            [0, None],
            [True, False]
        )
        result = str(self.postprocessing_config[key]) + check_d_x.message
        return result, count_mistakes_postprocessing + check_d_x.is_error

    def __check_d_y(self, count_mistakes_postprocessing: int, key: str):
        """
        Validates the 'd_y' parameter.

        Check if value is int and greater than 0.

        Args:
            count_mistakes_postprocessing (int): Current error count.
            key (str): The key for the 'd_y' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_d_y = check_parameter_type_and_value(
            self.postprocessing_config[key],
            int,
            [0, None],
            [True, False]
        )
        result = str(self.postprocessing_config[key]) + check_d_y.message
        return result, count_mistakes_postprocessing + check_d_y.is_error

    def __check_d_z(self, count_mistakes_postprocessing: int, key: str):
        """
        Validates the 'd_z' parameter.

        Check if value is int and greater than 0.

        Args:
            count_mistakes_postprocessing (int): Current error count.
            key (str): The key for the 'd_z' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_d_z = check_parameter_type_and_value(
            self.postprocessing_config[key],
            int,
            [0, None],
            [True, False]
        )
        result = str(self.postprocessing_config[key]) + check_d_z.message
        return result, count_mistakes_postprocessing + check_d_z.is_error

    def __check_smooth_factor(self, count_mistakes_postprocessing: int, key: str):
        """
        Validates the 'smooth_factor' parameter.

        Check if value is float and greater than 0.

        Args:
            count_mistakes_postprocessing (int): Current error count.
            key (str): The key for the 'smooth_factor' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_smooth_factor = check_parameter_type_and_value(
            self.postprocessing_config[key],
            float,
            [0, None],
            [True, False]
        )
        result = str(self.postprocessing_config[key]) + check_smooth_factor.message
        return result, count_mistakes_postprocessing + check_smooth_factor.is_error

    def __check_remove_outliers_smoothing(self, count_mistakes_postprocessing: int, key: str):
        """
        Validates the 'remove_outliers_smoothing' parameter.

        Check if value is bool.

        Args:
            count_mistakes_postprocessing (int): Current error count.
            key (str): The key for the 'remove_outliers_smoothing' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_remove_outliers_smoothing = check_datatype(
            self.postprocessing_config[key],
            bool
        )
        result = str(self.postprocessing_config[key]) + check_remove_outliers_smoothing.message
        return result, count_mistakes_postprocessing + check_remove_outliers_smoothing.is_error

    def __check_model_vmin_vmax(self, count_mistakes_postprocessing: int, keys: tuple) -> tuple[str, str, int]:
        """
        Validates 'model_vmin' and 'model_vmax' (minimum and maximum frequency).

        Checks they are floats, greater than 0 and that 'model_vmax' is greater than 'model_vmin'.

        Args:
            count_mistakes_postprocessing (int): Current error count.
            keys (tuple): Keys for 'model_vmin' and 'model_vmax'.

        Returns:
            tuple: Validation results (str) and updated error count (int).
        """
        key_modev_vmin, key_model_vmax = keys  # Unpack the keys

        # Validate 'ffid_start' and 'ffid_stop'
        check_model_vmin_vmax = [
            check_parameter_type_and_value(
                self.postprocessing_config[key_modev_vmin],
                float,  # Expected type is float
                [0, None],  # Range: >= 0
                [True, False],  # Inclusive minimum, no maximum bound
            ),
            check_parameter_type_and_value(
                self.postprocessing_config[key_model_vmax],
                float,  # Expected type is float
                [0, None],  # Range: >= 0
                [True, False],  # Inclusive minimum, no maximum bound
            ),
        ]

        if not check_model_vmin_vmax[0].is_error and not check_model_vmin_vmax[1].is_error:
            # Check if 'ffid_stop' is greater than 'ffid_start'
            check_ffid_start_stop_relationship = check_values_relationship(
                [
                    self.postprocessing_config[key_modev_vmin],
                    self.postprocessing_config[key_model_vmax],
                ],
                [key_modev_vmin, key_model_vmax],
                False,  # Strict comparison: ffid_stop > ffid_start
            )
            return (
                str(self.postprocessing_config[key_modev_vmin])
                + check_model_vmin_vmax[0].message,
                str(self.postprocessing_config[key_model_vmax])
                + check_ffid_start_stop_relationship.message,
                count_mistakes_postprocessing
                + check_ffid_start_stop_relationship.is_error,
            )
        else:
            # Handle type validation errors
            return (
                str(self.postprocessing_config[key_modev_vmin])
                + check_model_vmin_vmax[0].message,
                str(self.postprocessing_config[key_model_vmax])
                + check_model_vmin_vmax[1].message,
                count_mistakes_postprocessing + 1,
            )

    def __check_save_segy(self, count_mistakes_postprocessing: int, key: str) -> tuple[str, int]:
        """
        Validates the 'save_segy' parameter.

        Check if value is bool.

        Args:
            count_mistakes_postprocessing (int): Current error count.
            key (str): The key for the 'save_segy' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_save_segy = check_datatype(
            self.postprocessing_config[key],
            bool
        )
        result = str(self.postprocessing_config[key]) + check_save_segy.message
        return result, count_mistakes_postprocessing + check_save_segy.is_error

    def __check_error_thr(self, count_mistakes_postprocessing: int, key: str):
        """
        Validates the 'error_thr' parameter.

        Check if value is float and greater than 0.

        Args:
            count_mistakes_postprocessing (int): Current error count.
            key (str): The key for the 'error_thr' parameter.

        Returns:
            tuple: Validation result (str) and updated error count (int).
        """
        check_error_thr = check_parameter_type_and_value(
            self.postprocessing_config[key],
            float,
            [0, 1],
            [True, True]
        )
        result = str(self.postprocessing_config[key]) + check_error_thr.message
        return result, count_mistakes_postprocessing + check_error_thr.is_error

    def __check_parameters_2d(self, count_mistakes_postprocessing: int, level: str) -> tuple[dict, int]:

        """
        Validates the 'parameters_3d' configuration parameters and updates the error count.

        Args:
            count_mistakes_postprocessing (int): The current count of spectral errors.
            level (str): The key for the 'advanced' configuration parameters.

        Returns:
            tuple: A tuple containing:
                - A dictionary with the validation results for each parameter (str: value + message).
                - An updated count of spectral errors (int).

        The function performs validation checks for the following parameters:
        1. 'interp_dim': Checks if the value is a valid string
        within the acceptable degree of interpolation ('1d' and '2d').
        """
        check_parameters_2d = {}  # Initialize the dict to store the validation checks
        interp_dim_ind = 0
        key_interp_dim = POSTPROCESSING_2D_PARAMETERS[interp_dim_ind]

        existence_list = check_level_correct(
            self.postprocessing_config[level], POSTPROCESSING_2D_PARAMETERS, level
        )
        if existence_list.is_error:  # If the existence check failed
            check_parameters_2d = existence_list.message  # Store the error message
            count_mistakes_postprocessing += 1  # Increment the error count
        else:  # If the existence check succeeded
            check_interp_dim = check_valid_string(
                self.postprocessing_config[level][key_interp_dim],
                INTRRP_DIM
            )
            check_parameters_2d[key_interp_dim] = (
                str(self.postprocessing_config[level][key_interp_dim])
                + check_interp_dim.message
            )
            count_mistakes_postprocessing += check_interp_dim.is_error

        return check_parameters_2d, count_mistakes_postprocessing

    def __check_parameters_3d(self, count_mistakes_postprocessing: int, level: str) -> tuple[dict, int]:
        """
        Validates the 'parameters_3d' configuration parameters and updates the error count.

        Args:
            count_mistakes_postprocessing (int): The current count of spectral errors.
            level (str): The key for the 'advanced' configuration parameters.

        Returns:
            tuple: A tuple containing:
                - A dictionary with the validation results for each parameter (str: value + message).
                - An updated count of spectral errors (int).

        The function performs validation checks for the following parameters:
        1. 'num_xslices_3d_ind': Checks if the value is an integer ond greater than 0.
        2. 'num_yslices_3d_ind': Checks if the value is an integer ond greater than 0.
        """
        check_parameters_3d = {}
        num_xslices_3d_ind = 0
        num_yslices_3d_ind = 1
        key_num_xslices_3d = POSTPROCESSING_3D_PARAMETERS[num_xslices_3d_ind]
        key_num_yslices_3d = POSTPROCESSING_3D_PARAMETERS[num_yslices_3d_ind]

        existence_list = check_level_correct(
            self.postprocessing_config[level], POSTPROCESSING_3D_PARAMETERS, level
        )

        if existence_list.is_error:  # If the existence check failed
            check_parameters_3d = existence_list.message  # Store the error message
            count_mistakes_postprocessing += 1  # Increment the error count
        else:  # If the existence check succeeded
            check_num_xslices_3d = check_parameter_type_and_value(
                self.postprocessing_config[level][key_num_xslices_3d],
                int,
                [0, None],
                [True, True]
            )
            check_parameters_3d[key_num_xslices_3d] = (
                str(self.postprocessing_config[level][key_num_xslices_3d])
                + check_num_xslices_3d.message
            )
            count_mistakes_postprocessing += check_num_xslices_3d.is_error

            check_num_yslices_3d = check_parameter_type_and_value(
                self.postprocessing_config[level][key_num_yslices_3d],
                int,
                [0, None],
                [True, True]
            )
            check_parameters_3d[key_num_yslices_3d] = (
                str(self.postprocessing_config[level][key_num_yslices_3d])
                + check_num_yslices_3d.message
            )
            count_mistakes_postprocessing += check_num_yslices_3d.is_error

        return check_parameters_3d, count_mistakes_postprocessing

    def check(self) -> tuple[dict, int]:
        """
        Main validation method for the 'postprocessing' section.
        It orchestrates the validation of parameters and returns any errors.

        Returns:
            tuple: A tuple containing:
                - postprocessing_errors (dict): A dictionary of validation errors, if any.
                - count_mistakes_postprocessing (int): The total number of validation errors.
        """
        postprocessing_errors = {}  # Store validation results for each parameter
        count_mistakes_postprocessing = 0  # Initialize error counter

        # Check if the 'preprocessing' section exists and has mandatory parameters
        check_existence = self.__check_existence('postprocessing')
        if check_existence.is_error:
            # Handle existence check error
            postprocessing_errors = check_existence.message  # Store the error
            count_mistakes_postprocessing += 1  # Increment error counter
        else:
            # Validate individual parameters if existence check passed

            (postprocessing_errors['max_depth'],
             count_mistakes_postprocessing) = self.__check_max_depth(
                count_mistakes_postprocessing,  # Pass current error count
                'max_depth'  # Specify parameter to check
            )

            (postprocessing_errors['d_x'],
             count_mistakes_postprocessing) = self.__check_d_x(
                count_mistakes_postprocessing,  # Pass current error count
                'd_x'  # Specify parameter to check
            )

            (postprocessing_errors['d_y'],
             count_mistakes_postprocessing) = self.__check_d_y(
                count_mistakes_postprocessing,  # Pass current error count
                'd_y'  # Specify parameter to check
            )

            (postprocessing_errors['d_z'],
             count_mistakes_postprocessing) = self.__check_d_z(
                count_mistakes_postprocessing,  # Pass current error count
                'd_z'  # Specify parameter to check
            )

            (postprocessing_errors['smooth_factor'],
             count_mistakes_postprocessing) = self.__check_smooth_factor(
                count_mistakes_postprocessing,  # Pass current error count
                'smooth_factor'  # Specify parameter to check
            )
            (postprocessing_errors['remove_outliers_smoothing'],
             count_mistakes_postprocessing) = self.__check_remove_outliers_smoothing(
                count_mistakes_postprocessing,  # Pass current error count
                'remove_outliers_smoothing'  # Specify parameter to check
            )
            (postprocessing_errors['vmin_in_model'],
             postprocessing_errors['vmax_in_model'],
             count_mistakes_postprocessing) = self.__check_model_vmin_vmax(
                count_mistakes_postprocessing,  # Pass current error count
                ('vmin_in_model', 'vmax_in_model')  # Specify parameter to check
            )

            (postprocessing_errors['save_segy'],
             count_mistakes_postprocessing) = self.__check_save_segy(
                count_mistakes_postprocessing,
                'save_segy'
            )

            (postprocessing_errors['error_thr'],
             count_mistakes_postprocessing) = self.__check_error_thr(
                count_mistakes_postprocessing,
                'error_thr'
            )

            (postprocessing_errors['parameters_2d'],
             count_mistakes_postprocessing) = self.__check_parameters_2d(
                count_mistakes_postprocessing,
                'parameters_2d'
            )

            (postprocessing_errors['parameters_3d'],
             count_mistakes_postprocessing) = self.__check_parameters_3d(
                count_mistakes_postprocessing,
                'parameters_3d'
            )

        return postprocessing_errors, count_mistakes_postprocessing





