import os  # Module for interacting with the operating system
import yaml  # YAML parsing library
from pathlib import Path  # Object-oriented filesystem paths
from pydantic import BaseModel  # Data validation and settings management

from src.config_reader.Checker.Checker import Checker  # Custom class for config validation
from src.config_reader.Checker.exceptions import (InvalidConfigurationParameters,
                                                  InvalidConfigurationFileAddress)  # Custom exceptions


class ConfigReader:
    """
    Reads, parses, validates, and processes configuration files.
    """

    @staticmethod
    def __parse_config(path: Path) -> dict:
        """
        Parses the YAML configuration file.

        Args:
            path (Path): Path to the YAML file.

        Returns:
            dict: Dictionary containing the configuration parameters.

        Raises:
            InvalidConfigurationFileAdress: If the specified file is not found.
        """

        with path.open() as data:
            params = yaml.load(data, Loader=yaml.FullLoader)  # Load YAML

        return params

    @staticmethod
    def __get_model(params: dict, model: BaseModel) -> BaseModel:
        """
        Creates and returns a Pydantic model from the parsed parameters.

        Args:
            params (dict): Dictionary of configuration parameters.
            model (BaseModel): The Pydantic model class to instantiate.

        Returns:
            BaseModel: An instance of the Pydantic model.
        """
        return model(**params).extract_core_params()

    @staticmethod
    def read(path: Path, model, swi_dir: Path, show: bool = True) -> BaseModel:
        """
        Orchestrates the reading, parsing, validation, and processing
        of a configuration file.

        This method reads a YAML configuration file, parses its contents,
        validates the parsed data against a Pydantic model using a custom Checker,
        and then loads the validated data into the Pydantic model.

        Args:
            path (Path): Path to the YAML configuration file.
            model (BaseModel): Pydantic model for data validation.
            swi_dir (Path): Directory to store processing results and logs.
            show (bool): Whether to print the configuration after reading
                         and parsing (default: True).

        Returns:
            BaseModel: An instance of the validated Pydantic model populated
                       with the configuration data.

        Raises:
            InvalidConfigurationParameters: If the configuration parameters
                                            are invalid according to the Checker.
            InvalidConfigurationFileAddress: If the configuration file is not
                                            found or cannot be parsed.
        """
        # Ensure the SWI directory exists, creating it if necessary.
        if not os.path.isdir(swi_dir):
            os.makedirs(swi_dir)  # Create directory, including any parent directories.

        try:
            # Parse the configuration file using the ConfigReader's internal parsing method.
            my_config = ConfigReader.__parse_config(path)
        except InvalidConfigurationFileAddress as e:
            # Re-raise the exception if the file cannot be parsed
            raise e

        # Validate the parsed configuration using the Checker class.
        count_mistakes = Checker(my_config).check(swi_dir)

        # Print the configuration (if requested).
        if show:
            print(yaml.dump(my_config))

        # Raise an exception if validation finds any errors.
        if count_mistakes > 0:
            raise InvalidConfigurationParameters("Error: invalid config parameters, check logs")

        # Load the configuration into the Pydantic model.
        my_model = ConfigReader.__get_model(my_config, model)
        return my_model
