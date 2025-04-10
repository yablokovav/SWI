import logging  # Module for logging messages
from pathlib import Path  # Module for object-oriented filesystem paths

from src.config_reader.Checker.PreprocessingChecker import PreprocessingChecker  # Class to check preprocessing section
from src.config_reader.Checker.SpectralChecker import SpectralChecker  # Class to check 'spectral' section
from src.config_reader.Checker.InversionChecker import InversionChecker  # Class to check 'inversion' section
from src.config_reader.Checker.PostprocessingChecker import PostprocessingChecker
from src.config_reader.Checker.utils import check_level_correct  # Function to check validation of dictionary


# List of mandatory parameters in the configuration
PARAMETERS = ['preprocessing', 'spectral', 'inversion', 'postprocessing']


class Checker:
    """
    The Checker class validates the SWI configuration, ensuring the presence
    of mandatory parameters and delegating section-specific validation
    to specialized checker classes.
    """

    def __init__(self, config: dict):
        """
        Initializes the Checker with the configuration to validate.

        Args:
            config (dict): The configuration dictionary.
        """
        self.config = config

    @staticmethod
    def __create_log(check_config: dict, config_logger: logging.Logger) -> None:
        """
        Creates a structured log of the configuration check results.

        This method recursively iterates through the validation results and
        logs each section and its corresponding information, providing a
        hierarchical representation of the configuration status.

        Args:
            check_config (dict): A dictionary containing the validation results.
                                 The structure is assumed to be nested dictionaries
                                 representing different sections and sub-sections
                                 of the configuration.
            config_logger (logging.Logger): The logger instance to use for
                                            writing the validation information to
                                            the log file.
        """
        if isinstance(check_config, dict):  # Check if the config is a dictionary
            for key, value in check_config.items():  # Iterate through the key-value pairs
                config_logger.info("%s:", key)  # Log the section name (key)

                if isinstance(value, dict):  # Check if the value is a dictionary (subsection)
                    for sub_key, sub_value in value.items():  # Iterate through the subsection
                        if isinstance(sub_value, dict):  # Check for further nesting
                            config_logger.info("\t%s:", sub_key)  # Log the subsection name
                            for item_key, item_value in sub_value.items():  # Iterate through items
                                config_logger.info("\t\t%s %s", item_key, item_value)  # Log item details
                        else:
                            config_logger.info("\t%s %s", sub_key, sub_value)  # Log subsection value
                else:
                    config_logger.info("\t%s", value)  # Log a top-level (non-nested) value
        else:
            config_logger.debug(check_config)  # Log config if it's not a dictionary

    def check(self, swi_dir: Path) -> int:
        """
        Performs the configuration validation process.

        This method checks the configuration parameters for correctness and
        delegates validation to section-specific checkers.  It also sets up
        logging to record any errors found during the validation process.

        Args:
            swi_dir (Path): Path to the SWI directory
             where the config.log file will be created.

        Returns:
            int: The number of errors found during the validation. A value of 0,
             indicates that the configuration is valid.
        """
        count_mistakes = 0  # Initialize the error counter
        check_config = {}  # Dictionary to store validation results for each section

        # Check for mandatory parameters using the 'check_level_correct' function.
        existence_list = check_level_correct(
            self.config,
            PARAMETERS,
            'config'
        )

        if existence_list.is_error:
            # If a mandatory parameter is missing, capture the error message
            check_config = existence_list.message
            count_mistakes += 1  # Increment the error counter
        else:
            # If all mandatory parameters exist, delegate validation to section-specific checkers.
            preprocessing_checker = PreprocessingChecker(self.config['preprocessing'])
            check_config['preprocessing'], count_mistakes_preprocessing = preprocessing_checker.check()
            count_mistakes += count_mistakes_preprocessing

            spectral_checker = SpectralChecker(self.config['spectral'])
            check_config['spectral'], count_mistakes_spectral = spectral_checker.check()
            count_mistakes += count_mistakes_spectral

            inversion_checker = InversionChecker(self.config['inversion'])
            check_config['inversion'], count_mistakes_inversion = inversion_checker.check()
            count_mistakes += count_mistakes_inversion

            postprocessing_checker = PostprocessingChecker(self.config['postprocessing'])
            check_config['postprocessing'], count_mistakes_postprocessing = postprocessing_checker.check()
            count_mistakes += count_mistakes_postprocessing

        # Set up logging to record the validation results.
        path_log = swi_dir / "config.log"  # Define the log file path.
        config_logger = logging.getLogger("config_logger")  # Get a logger instance.
        config_logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG.

        # Configure a file handler to write log messages to the specified file.
        handler = logging.FileHandler(path_log, mode="w")
        formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)  # Set the log message format.
        config_logger.addHandler(handler)  # Add the handler to the logger.

        # Create a structured log of the validation results.
        Checker.__create_log(check_config, config_logger)

        # Log an error message if any errors were found during validation.
        if count_mistakes > 0:
            config_logger.error("Error: invalid parameters value, check and fix")

        # Clean up the logger handlers to prevent duplication of log messages.
        config_logger.handlers = []
        handler.close()

        return count_mistakes  # Return the total number of errors found.
