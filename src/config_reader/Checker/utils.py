import os.path  # Import the os.path module for path manipulations
import numpy as np  # Import the numpy library for numerical operations
from src.config_reader.Checker.Message import Message  # Import the Message class

# Define a constant for a "good" or empty string (used when no errors found)
GOOD_STRING = ''


def check_parameter_type_and_value(param, param_type, bounds: list, strict: list) -> Message:
    """
    Checks if a parameter's type is correct and its value is within valid bounds.

    Args:
        param: The parameter to check.
        param_type: The expected data type of the parameter.
        bounds (list): Valid bounds for the parameter value [min, max].
        strict (list): Boolean values to indicate if the bounds are strict [min_strict, max_strict]
                        (e.g., [True, False] indicates min < value <= max).

    Returns:
        Message: Result of the validation check.
    """
    check_value_type = check_datatype(param, param_type)  # First, check the datatype
    if not check_value_type.is_error:  # If datatype is valid, check the value
        return check_valid_value(param, bounds, strict)  # Check if value is within valid bounds
    else:  # If datatype is invalid, return the datatype error message
        return check_value_type


def check_level_correct(params: dict, correct_params: list, key: str) -> Message:
    """
    Checks if the level (section) in the configuration is a dictionary and has
    all the correct parameters.

    Args:
        params (dict): The parameters for this level in the configuration.
        correct_params (list): Expected parameters for this level.
        key (str): The key representing this level in the configuration.

    Returns:
        Message: Result of the validation check.
    """
    check_on_dict = check_is_dict(params, key)  # Check if the level is a dictionary
    if not check_on_dict.is_error:  # If it's a dictionary, check the parameters
        return check_parameters_existence(params, correct_params)  # Check if required parameters exist
    else:  # If it's not a dictionary, return the dictionary check error
        return check_on_dict


def check_parameters_existence(params: dict, keys: list) -> Message:
    """
    Checks for the existence of required parameters and the absence of
    unexpected parameters.

    Args:
        params (dict): List of required parameters.
        keys (list): List of parameters read from the configuration file.

    Returns:
        Message: Result of the validation check.
    """
    for parameter in params:  # Check for missing parameters
        if parameter not in keys:
            # string = f" Error: Parameter is missing: {parameter}"
            string = f" Error: Unknown parameter: {parameter}"
            return Message(is_error=True, message=string)
    for parameter in keys:  # Check for unexpected parameters
        if parameter not in params:
            # string = f" Error: Unknown parameter: {parameter}"
            string = f" Error: Parameter is missing: {parameter}"
            return Message(is_error=True, message=string)
    return Message(is_error=False, message=GOOD_STRING)  # All checks passed


def check_valid_string(value, valid_values: list) -> Message:
    """
    Checks if the given value is one of the allowed valid values.

    Args:
        value: The value to check.
        valid_values (list): List of valid string values.

    Returns:
        Message: Result of the validation check.
    """
    if value not in valid_values:
        valid_values_str = ", ".join(valid_values)
        string = f" Error: Possible parameter options: {valid_values_str}"
        return Message(is_error=True, message=string)
    else:
        return Message(is_error=False, message=GOOD_STRING)


def check_datatype(value, datatype) -> Message:
    """
    Checks if the given value's datatype matches the expected datatype.

    Args:
        value: The value to check.
        datatype: The expected datatype (e.g., int, float, bool).

    Returns:
        Message: Result of the validation check.
    """
    if datatype is not float:  # If not a float, do a simple type check
        if not isinstance(value, datatype):  # Use isinstance instead of type()
            string = (
                " Error: Invalid parameter type"
                f", type found: {type(value).__name__}"
                f", type is required: {datatype.__name__}"
            )
            return Message(is_error=True, message=string)
        else:
            return Message(is_error=False, message=GOOD_STRING)
    else:  # If expecting a float, also allow integers
        if not isinstance(value, (float, int)):  # Check for both float and int
            string = (
                " Error: Invalid parameter type"
                f", type found: {type(value).__name__}"
                f", type is required: {datatype.__name__}"
            )
            return Message(is_error=True, message=string)
        else:
            return Message(is_error=False, message=GOOD_STRING)


def check_valid_value(value: float | int, bounds: list, strict: list) -> Message:
    """
    Checks if the given value is within the specified bounds, considering
    strictness (inclusive or exclusive).

    Args:
        value: The value to check.
        bounds (list): Valid bounds for the value [min, max].
        strict (list): Boolean values to indicate if the bounds are strict [min_strict, max_strict]
                        (e.g., [True, False] indicates min < value <= max).

    Returns:
        Message: Result of the validation check.
    """
    if strict[0]:  # Check the lower bound strictly
        if value <= bounds[0]:
            string = \
                f" Error: Invalid parameter value. The value must be greater then {bounds[0]}"
            return Message(is_error=True, message=string)
    else:  # Check the lower bound non-strictly
        if value < bounds[0]:
            string = \
                f" Error: Invalid parameter value. The value must be greater or equal then {bounds[0]}"
            return Message(is_error=True, message=string)

    if bounds[1] is not None:  # If there is an upper bound
        if strict[1]:  # Check the upper bound strictly
            if value >= bounds[1]:
                string = \
                    f" Error: Invalid parameter value. The value must be less then {bounds[1]}"
                return Message(is_error=True, message=string)
        else:  # Check the upper bound non-strictly
            if value > bounds[1]:
                string = \
                    f" Error: Invalid parameter value. The value must be less or equal then {bounds[1]}"
                return Message(is_error=True, message=string)

    return Message(is_error=False, message=GOOD_STRING)  # All checks passed


def check_values_relationship(values: list, keys: list, strict: bool) -> Message:
    """
    Checks if the relationship between two values is correct
    (greater than or greater than or equal to).

    Args:
        values (list): The two values to compare.
        keys (list): The keys associated with the values (for error messages).
        strict (bool): If True, the first value must be strictly less than the second.
                       If False, the first value must be less than or equal to the second.

    Returns:
        Message: Result of the validation check.
    """
    if strict:
        # Strict comparison: values[0] < values[1]
        if values[0] >= values[1]:
            string = (
                " Error: incorrect value parameter "
                f"{keys[1]}: {keys[1]} must be greater then {keys[0]}"
            )
            return Message(is_error=True, message=string)
    else:
        # Non-strict comparison: values[0] <= values[1]
        if values[0] > values[1]:
            string = (
                " Error: incorrect value parameter "
                f"{keys[1]}: {keys[1]} must be greater or equal then {keys[0]}"
            )
            return Message(is_error=True, message=string)

    return Message(is_error=False, message=GOOD_STRING)  # Relationship is valid


def check_is_dict(value, key: str) -> Message:
    """
    Checks if the provided value is a dictionary.

    Args:
        value: The value to check.
        key (str): The name of the parameter being checked (for error message).

    Returns:
        Message: Result of the validation check.
    """
    if not isinstance(value, dict):
        string = f" Error: {key} must be the title"
        return Message(is_error=True, message=string)

    return Message(is_error=False, message=GOOD_STRING)  # Value is a dictionary


def check_csv_file(value: str) -> Message:
    """
    Checks if the provided value is a valid path to an existing CSV file
    that can be opened and read by numpy.

    Args:
        value (str): The path to the CSV file.

    Returns:
        Message: Result of the validation check.
    """
    if value is not None:  # If the value is not None
        if not os.path.exists(value):  # Check if the file exists
            string = " Error: File not found."
            return Message(is_error=True, message=string)
        else:
            # Attempt to open and read the CSV file using numpy
            try:
                np.genfromtxt(value, delimiter=";")
                return Message(is_error=False, message=GOOD_STRING)  # CSV is valid
            except (ValueError, FileNotFoundError):  # Handle exceptions
                string = " Error: The file cannot be opened. The table size may be incorrect."
                return Message(is_error=True, message=string)
    else:
        return Message(is_error=False, message=GOOD_STRING)  # Value is None


def check_vs(path: str, headers: list, min_height: int, keys: list) -> Message:
    """
    Validates the contents of a shear wave velocity (Vs) model file.

    Checks for correct table width, headers, minimum number of lines,
    numeric values, and valid relationships between parameters.

    Args:
        path (str): Path to the Vs model file.
        headers (list): Expected headers of the Vs model file.
        min_height (int): Minimum number of data rows required.
        keys (list): Keys for Vs parameters ('vs_min', 'vs_max', 'h_min', 'h_max').

    Returns:
        Message: Result of the validation check.
    """
    key_vs_min, key_vs_max, key_thk_min, key_thk_max = keys  # Unpack keys
    try:
        file = np.genfromtxt(path, delimiter=';', skip_header=1)  # Skip the header row
        file_str = np.genfromtxt(path, delimiter=';', dtype=str)
    except Exception as e:
        return Message(is_error=True, message=f"Error while reading file: {e}")

    # Check that the number of columns in headers matches the shape of the loaded data
    if len(headers) != file_str.shape[1]:
        string = (
                array_to_str(file_str) +
                f" Error: Incorrect table width, required width: {len(headers)}, Current width: {file_str.shape[1]}"
        )
        return Message(is_error=True, message=string)

    # Check for header values
    headers_array = np.array(headers)
    file_headers = file_str[0]

    # Check if any of the values don't match
    if not np.array_equal(headers_array, file_headers):
        string = (
                array_to_str(file_str) +
                f" Error: Incorrect headers, expected: {headers}, found: {list(file_headers)}"
        )
        return Message(is_error=True, message=string)

    # Ensure the file has enough lines in the body
    if file.shape[0] < min_height:
        string = (
                array_to_str(file_str) +
                f" Error, Too few lines, minimum count of lines: {min_height}, current count: {file.shape[0]}"
        )
        return Message(is_error=True, message=string)

    if np.isnan(file).any():
        string = (
                array_to_str(file_str) +
                " Error: Non-numeric values were found in the file. Numeric values are required"
        )
        return Message(is_error=True, message=string)

        # Validate min and max wave values are > 0

    if (file[:, :2] <= 0).any():
        string = (
                array_to_str(file_str) +
                " Error: vs_min, vs_max must be > 0"
        )
        return Message(is_error=True, message=string)

    if (file[:-1, 2:4] <= 0).any():
        string = (
                array_to_str(file_str) +
                " Error: h_min, h_max must be > 0"
        )
        return Message(is_error=True, message=string)

    if (file[:, 4] <= 1.4).any():
        string = (
                array_to_str(file_str) +
                " Error: The velocity ratio must be greater than 1.4"
        )
        return Message(is_error=True, message=string)

    # Check relationships between values in each row.
    for i in range(file.shape[0]):

        check_vs_relationship = check_values_relationship(
                    [file[i][0], file[i][1]],
                    [key_vs_min, key_vs_max],
                    True
                )
        check_thk_relationship = check_values_relationship(
                    [file[i][2], file[i][3]],
                    [key_thk_min, key_thk_max],
                    False
                )

        if check_vs_relationship.is_error:
            return Message(is_error=True, message=array_to_str(file_str) + check_vs_relationship.message)

        if check_thk_relationship.is_error:
            return Message(is_error=True, message=array_to_str(file_str) + check_thk_relationship.message)

    return Message(is_error=False, message=array_to_str(file_str))  # All checks passed


def check_dc(path: str, headers: list, min_height: int, keys: list) -> Message:
    """
    Validates the contents of a dispersion curve (DC) file.

    Checks for correct table width, headers, minimum number of lines,
    numeric values, and valid relationships between parameters.

    Args:
        path (str): Path to the DC file.
        headers (list): Expected headers of the DC file.
        min_height (int): Minimum number of data rows required.
        keys (list): Keys for v_min and v_max ('v_min', 'v_max').

    Returns:
        Message: Result of the validation check.
    """
    key_v_min, key_v_max = keys  # Unpack keys
    try:
        file = np.genfromtxt(path, delimiter=';', skip_header=1)  # Skip header
        file_str = np.genfromtxt(path, delimiter=';', dtype=str)  # Load as string
    except Exception as e:
        return Message(is_error=True, message=f"Error reading file: {e}")

    # Check if the number of columns in headers matches the shape of the loaded data
    if len(headers) != file_str.shape[1]:
        string = (
                array_to_str(file_str) +
                f" Error: Incorrect table width, required width: {len(headers)}, Current width: {file_str.shape[1]}"
        )
        return Message(is_error=True, message=string)

    # Check for header values
    headers_array = np.array(headers)
    file_headers = file_str[0]

    # Check if any of the values don't match
    if not np.array_equal(headers_array, file_headers):
        string = (
                array_to_str(file_str) +
                f" Error: Incorrect headers, expected: {headers}, found: {list(file_headers)}"
        )
        return Message(is_error=True, message=string)

    if file.shape[0] < min_height:
        string = (
                array_to_str(file_str) +
                f" Error, Too few lines, minimum count of lines: {min_height}, Current count: {file.shape[0]}"
        )
        return Message(is_error=True, message=string)

    if np.isnan(file).any():
        string = (
                array_to_str(file_str) +
                " Error: Non-numeric values were found in the file. Numeric values are required"
        )
        return Message(is_error=True, message=string)

    # Validate frequencies are >=0.0
    if (file[:, 0] < 0).any():
        string = (
                array_to_str(file_str) +
                " Error: freq must be >= 0"
        )
        return Message(is_error=True, message=string)

    # Validate min and max velocities are > 0
    if (file[:, 1:3] <= 0).any():
        string = (
                array_to_str(file_str) +
                " Error: v_min, v_max must be > 0"
        )
        return Message(is_error=True, message=string)

    for i in range(file.shape[0]):
        # Validate min velocity is less than or equal to the max velocity
        check_v_relationship = check_values_relationship(
            [file[i][-2], file[i][-1]], [key_v_min, key_v_max], True
        )  # strict=True checks for <

        if check_v_relationship.is_error:
            return Message(
                is_error=True, message=array_to_str(file_str) + check_v_relationship.message
            )

    return Message(is_error=False, message=array_to_str(file_str))  # All checks passed


def check_vp(path: str, headers: list, min_height: int) -> Message:
    """
    Validates the contents of a compressional wave velocity (Vp) model file.

    Args:
        path (str): Path to the Vp model file.
        headers (list): Expected headers of the Vp model file.
        min_height (int): Minimum number of data rows required.

    Returns:
        Message: Result of the validation check.
    """
    try:
        file = np.genfromtxt(path, delimiter=';', skip_header=1)  # Skip header
        file_str = np.genfromtxt(path, delimiter=';', dtype=str)
    except Exception as e:
        return Message(is_error=True, message=f"Error reading file: {e}")

    if len(headers) != file_str.shape[1]:
        string = (
                array_to_str(file_str) +
                f" Error: Incorrect table width, required width: {len(headers)}, Current width: {file_str.shape[1]}"
        )
        return Message(is_error=True, message=string)

    # Check for header values
    headers_array = np.array(headers)
    file_headers = file_str[0]

    # Check if any of the values don't match
    if not np.array_equal(headers_array, file_headers):
        string = (
                array_to_str(file_str) +
                f" Error: Incorrect headers, expected: {headers}, found: {list(file_headers)}"
        )
        return Message(is_error=True, message=string)

    if file.shape[0] < min_height:
        string = (
                array_to_str(file_str) +
                f" Error, Too few lines, minimum count of lines: {min_height}, current count: {file.shape[0]}"
        )
        return Message(is_error=True, message=string)

    if np.isnan(file).any():
        string = (
                array_to_str(file_str) +
                " Error: Non-numeric values were found in the file. Numeric values are required"
        )
        return Message(is_error=True, message=string)

    if (file[:, 0] < 0).any():
        string = (
                array_to_str(file_str) +
                " Error: Depths must be >= 0"
        )
        return Message(is_error=True, message=string)

    if (np.diff(file[:, 0]) <= 0).any():
        string = (
                array_to_str(file_str) +
                " Error: Depths must increase"
        )
        return Message(is_error=True, message=string)

    if (file[:, 1] <= 0).any():
        string = (
                array_to_str(file_str) +
                " Error: vp must be > 0"
        )
        return Message(is_error=True, message=string)

    if (file[:, 2] <= 1.4).any():
        string = (
                array_to_str(file_str) +
                " Error: vp2vs must be > 1.4"
        )
        return Message(is_error=True, message=string)

    return Message(is_error=False, message=array_to_str(file_str))  # All checks passed


def array_to_str(mas: np.ndarray) -> str:
    """
    Converts a 2D NumPy array to a formatted string representation.

    Args:
        mas (np.ndarray): The 2D NumPy array to convert.

    Returns:
        str: Formatted string representation of the array.
    """
    col_size = 10  # Define column width
    string = "\n"

    for i in range(len(mas)):
        string_prom = str(mas[i]).replace("[", '', 1)
        string_prom = string_prom.replace("]", '', 1)
        mas_prom = string_prom.split(' ')

        for j in range(len(mas_prom)):
            string += mas_prom[j] + ' ' * (col_size - len(mas_prom[j]))

        string += "\n"

    return string
