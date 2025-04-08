"""
Module of exceptions.
"""


class ConfigurationExceptions(Exception):
    """Base class of exception for configurations and preprocessing."""
    pass


class InvalidConfigurationParameters(ConfigurationExceptions):
    """Raising in preprocessing errors"""
    pass


class InvalidConfigurationFileAddress(ConfigurationExceptions):
    """Raising for incorrect file address"""
    pass
