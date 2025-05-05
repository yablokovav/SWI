import logging  # For logging events and errors during processing
from pathlib import Path  # For working with file paths in an object-oriented way

import numpy as np  # For numerical operations, especially array manipulation

from src import *  # For constants
from src.logs.Message import Message  # Special dataclass for log messages
from src.preprocessing.utils import define_spatial_step  # Function to calculate spatial step size


def setup_loger(config_parameters):
    """
    Sets up a logger for preprocessing and spectral analysis.

    The logger is configured to write debug-level messages to a file.

    Args:
        config_parameters: An object containing configuration parameters,
                           including the directory to save the log file.  It is expected
                           to have the attribute `save_dir_preprocessing` which is a list.
                           The Path of the log file will be 3 levels up from the first
                           Path object in that list and named "preprocessing_and_spectral_analysing.log".

    Returns:
        A tuple containing the configured logger and the file handler.
    """
    path_log = config_parameters.save_dir_preprocessing[0].parents[2] / "preprocessing_and_spectral_analysing.log"
    logger = logging.getLogger("preprocessing_and_spectral_logger")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(path_log, mode="w")  # Create a file handler
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")  # define format
    handler.setFormatter(formatter)  # set the handler
    logger.addHandler(handler)  # Add the handler to the logger
    return logger, handler

def close_logger(config_parameters):
    """
    Closes the logger and logs summary information about the preprocessing and spectral analysis.

    Logs the number of preprocessed seismograms and saved dispersion curves.
    Also logs messages indicating where the results were saved, and errors if no
    seismograms were preprocessed or no dispersion curves were saved.  Closes the file handler.

    Args:
        config_parameters: An object containing configuration parameters, including:
                           - `logger`: The logger object to close.
                           - `handler`: The file handler to close.
                           - `count_preprocessed_seismograms`: The number of preprocessed seismograms.
                           - `count_dispersion_curves`: The number of saved dispersion curves.
                           - `qc_preprocessing`: A boolean indicating whether preprocessing QC was enabled.
                           - `qc_spectral`: A boolean indicating whether spectral analysis QC was enabled.
                           - `save_dir_preprocessing`: A list containing the directory where preprocessed
                             seismograms were saved.
                           - `save_dir_spectral`: A list containing the directories where dispersion curves,
                             images of dispersion curves, and SEGY files of dispersion curves were saved. It is
                             expected to have at least 3 Path objects.
    """
    config_parameters.logger.info(f"Preprocessed seismograms: {config_parameters.count_preprocessed_seismograms}")
    config_parameters.logger.info(f"Saved dispersion curves: {config_parameters.count_dispersion_curves}")

    if config_parameters.count_preprocessed_seismograms == 0:
        config_parameters.logger.error("No valid seismograms")
    elif config_parameters.qc_preprocessing:
        config_parameters.logger.info(f"Preprocessed seismograms saved in: \n{config_parameters.save_dir_preprocessing[0]}")

    if config_parameters.count_dispersion_curves == 0:
        config_parameters.logger.error("No one dispersion curves was saved")
    else:
        config_parameters.logger.info(f"Dispersion curves saved in: \n{config_parameters.save_dir_spectral[0]}")
        if config_parameters.qc_spectral:
            config_parameters.logger.info(f"Images of dispersion curves saved in: \n{config_parameters.save_dir_spectral[1]}")
            config_parameters.logger.info(f"Segy files of dispersion curves saved in: \n{config_parameters.save_dir_spectral[2]}")

    config_parameters.logger.handlers = []
    config_parameters.handler.close()


def create_log(data, folder: Path, name: str,
               log_message: Message = Message(is_error=False, is_warning=False, message="")
               ) -> None:
    """
    Creates a log file and writes data and messages to it.

    This function sets up a logger, writes data from a dictionary to the log
    file, and then logs an error or warning message (if provided). Finally, it
    clears the logger's handlers to prevent duplicate logging in subsequent calls.

    Args:
        data (dict): A dictionary of data to be logged. Each key-value pair
                     will be written as a separate info-level log message.
        folder (Path): The directory in which to create the log file.
        name (str): The base name of the log file. The actual filename will be "{name}.log".
        log_message (Message, optional): A Message object containing an optional
                                       error or warning message.  Defaults to an
                                       empty Message (no error or warning).
    """
    path_log = folder / f"{name}.log"  # Construct the log file path
    logger = logging.getLogger("config_logger")  # Get the logger instance
    logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG
    handler = logging.FileHandler(path_log, mode="w")  # Create a file handler
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")  # define format
    handler.setFormatter(formatter)  # set the handler
    logger.addHandler(handler)  # Add the handler to the logger

    for i in data:
        logger.info(f"{i}: {data[i]}")  # Log each item in the data dictionary
    if log_message.is_error:
        logger.error(log_message.message)  # Log the error message
    elif log_message.is_warning:
        logger.warning(log_message.message)  # Log the warning message
    logger.handlers = []  # Remove all handlers to prevent duplicate logging
    handler.close()


def log_one_flank(header: np.ndarray, flank_id: int, snr: float) -> str:
    """
    Generates a formatted log string summarizing information about a
    single seismic data flank, intended for 2D data.

    This function extracts key parameters from a seismic data header, formats
    them into a human-readable string, and returns this string alongside
    the Common MidPoint (CMP) coordinates for that flank.

    Args:
        header (np.ndarray): A NumPy array containing seismic data header
                             information for a single flank.
        flank_id (int): An integer identifier for the flank being logged.
                        Used for easy identification in logging output.
        snr (float): The Signal-to-Noise Ratio for the current flank.

    Returns:
        tuple[float, float, str]: A tuple containing:
            - cmp_x (float): The X coordinate of the CMP.
            - cmp_y (float): The Y coordinate of the CMP.
            - log_flank (str): A formatted string containing the SP, receiver
                                position step, CMP, number of traces, flank ID,
                                and SNR. The string is formatted with aligned
                                columns for improved readability.
    """
    log_flank = ""  # Initialize the logging string

    sp_x, sp_y = float(header[HEADER_SOU_X_IND][0]), float(header[HEADER_SOU_Y_IND][0])  # Source point coordinates
    dx = np.abs(define_spatial_step(header))  # Receiver position step
    cmp_x, cmp_y = float(header[HEADER_CDP_X_IND][0]), float(header[HEADER_CDP_Y_IND][0])  # CMP coordinates
    number_traces = len(header[HEADER_OFFSET_IND])  # Number of traces

    # Format and append the logging information
    log_flank += f"SP: [{sp_x}, {sp_y}]: "  # Source Point Position
    log_flank += " " * (COLL_SIZE - len(f"SP: [{sp_x}, {sp_y}]: "))  # Align columns

    log_flank += f"Rp step: {dx}, "  # Receiver Position step
    log_flank += " " * (WIDTH_FOR_RP - len(f"Rp step: {dx}, "))  # Align columns

    log_flank += f"CMP: [{cmp_x}, {cmp_y}], "  # Common MidPoint
    log_flank += " " * (COLL_SIZE - len(f"CMP: [{cmp_x}, {cmp_y}], "))  # Align columns

    log_flank += f"Ntr: {number_traces}, "  # Number of traces
    log_flank += " " * (WIDTH_FOR_RP - len(f"Ntr: {number_traces}, "))  # Align columns

    log_flank += f"Flank {flank_id} "

    log_flank += f"SNR: %.2f" % snr  # Value of SNR
    log_flank += " " * (WIDTH_FOR_RP - len(f"SNR: %.2f" % snr))  # Align columns


    return log_flank  # Return coordinates and the formatted string


def create_table_preprocessing_3d_csp(item: int, header: np.ndarray, snr: float) -> str:
    """
    Generates a formatted string containing preprocessing information for a
    single sector of 3D CSP (Common Source Point) seismic data.

    This function extracts relevant information from a seismic data header
    (specifically for 3D CSP data), formats it into a string, and returns the
    string along with the FFID and sector ID.

    Args:
       item (int): The ID of the sector being processed.
       header (np.ndarray): A NumPy array containing seismic data
                                  header information for a single sector.
       snr (float): The Signal-to-Noise Ratio for the current sector.

    Returns:
       Tuple[int, int, str]: A tuple containing:
           - ffid (int): The Field File ID.
           - item (int): The sector ID (same as the input `item`).
            - sectors (str): A formatted string containing preprocessing
                             information about the sector, including the FFID,
                             sector number, number of traces (Ntr), receiver
                             position step (RP step), source position (SP)
                             coordinates, CMP coordinates, and SNR. The
                             string is formatted with aligned columns for
                             improved readability.
    """

    ffid = int(header[HEADER_FFID_IND][0])  # file ID
    number_traces = len(header[0])  # number of traces
    rp_step = define_spatial_step(header)  # Receiver position step
    sp_x, sp_y = float(header[HEADER_SOU_X_IND][0]), float(header[HEADER_SOU_Y_IND][0])  # Source point coordinates
    cmp_x, cmp_y = float(header[HEADER_CDP_X_IND][0]), float(header[HEADER_CDP_Y_IND][0])  # CMP coordinates

    # Format and append the logging information
    sectors = f"FFID: {ffid} "  # File ID
    sectors += " " * (WIDTH_FOR_RP - len(f"FFID: {ffid}"))  # Align columns

    sectors += f"sector: {item} "  # Sector Number
    sectors += " " * (WIDTH_FOR_RP - len(f"sector # {item}"))  # Align columns

    sectors += f"Ntr: {number_traces}"  # Number of traces
    sectors += " " * (WIDTH_FOR_RP - len(f"Ntr: {number_traces}"))  # Align columns

    sectors += f"RP step: {rp_step} "  # Receiver position step
    sectors += " " * (WIDTH_FOR_RP - len(f"RP step {rp_step} "))  # Align columns

    sectors += f"SP: [{sp_x}, {sp_y}] "  # Source position coordinates
    sectors += " " * (COLL_SIZE - len(f"SP [{sp_x}, {sp_y}], "))  # Align columns

    sectors += f"CMP: [{cmp_x} {cmp_y}] "  # CMP Coordinates
    sectors += " " * (COLL_SIZE - len(f"CMP [{cmp_x} {cmp_y}],"))  # Align columns

    sectors += f"SNR: %.2f" % snr
    sectors += " " * (WIDTH_FOR_RP - len(f"SNR: %.2f" % snr))

    return sectors  # Return values and the string


def create_table_preprocessing_3d_cdp(expand_headers: np.ndarray, snr: float) -> str:
    """
    Generates a formatted string containing preprocessing information for a
    single CDP (Common Depth Point) in 3D seismic data.

    This function extracts relevant information from a seismic data header
    (specifically for 3D data), formats it into a string, and returns the
    string along with the CDP coordinates.

    Args:
        expand_headers (np.ndarray): A NumPy array containing seismic data
                                     header information.
        snr (float): The Signal-to-Noise Ratio for the current CDP.

    Returns:
        Tuple[float, float, str]: A tuple containing:
            - cdp_x (float): The X coordinate of the CDP.
            - cdp_y (float): The Y coordinate of the CDP.
            - sectors (str): A formatted string containing the CDP coordinates,
                              receiver position step, number of traces, and
                              SNR. The string is formatted with aligned
                              columns for improved readability.
    """
    cdp_x, cdp_y = float(expand_headers[HEADER_CDP_X_IND][0]), float(expand_headers[HEADER_CDP_Y_IND][0])  # CDP Coordinates
    rp_step = define_spatial_step(expand_headers)  # Receiver position step
    number_traces = len(expand_headers[HEADER_OFFSET_IND])  # Number of traces


    # Format and append the logging information
    cdp = f"CDP: [{cdp_x}, {cdp_y}]: "  # Common Deep Point
    cdp += " " * (COLL_SIZE - len(f"CDP: [{cdp_x}, {cdp_y}]: "))  # Align columns

    cdp += f"RP step: {rp_step}, "  # Receiver position step
    cdp += " " * (WIDTH_FOR_RP - len(f"RP step: {rp_step}"))  # Align columns

    cdp += f"Ntr: {number_traces}"  # Number of traces
    cdp += " " * (WIDTH_FOR_RP - len(f"Ntr: {number_traces}"))

    cdp += f"SNR: %.2f" % snr
    cdp += " " * (WIDTH_FOR_RP - len(f"SNR: %.2f" % snr))

    return cdp  # Return values and the formatted string


def log_one_flank_no_mpi(header: np.ndarray, flank_id: int, snr: float) -> tuple[float, float, str]:
    """
    Generates a formatted log string summarizing information about a
    single seismic data flank, intended for 2D data.

    This function extracts key parameters from a seismic data header, formats
    them into a human-readable string, and returns this string alongside
    the Common MidPoint (CMP) coordinates for that flank.

    Args:
        header (np.ndarray): A NumPy array containing seismic data header
                             information for a single flank.
        flank_id (int): An integer identifier for the flank being logged.
                        Used for easy identification in logging output.
        snr (float): The Signal-to-Noise Ratio for the current flank.

    Returns:
        tuple[float, float, str]: A tuple containing:
            - cmp_x (float): The X coordinate of the CMP.
            - cmp_y (float): The Y coordinate of the CMP.
            - log_flank (str): A formatted string containing the SP, receiver
                                position step, CMP, number of traces, flank ID,
                                and SNR. The string is formatted with aligned
                                columns for improved readability.
    """
    log_flank = ""  # Initialize the logging string

    sp_x, sp_y = float(header[HEADER_SOU_X_IND][0]), float(header[HEADER_SOU_Y_IND][0])  # Source point coordinates
    dx = np.abs(define_spatial_step(header))  # Receiver position step
    cmp_x, cmp_y = float(header[HEADER_CDP_X_IND][0]), float(header[HEADER_CDP_Y_IND][0])  # CMP coordinates
    number_traces = len(header[HEADER_OFFSET_IND])  # Number of traces

    # Format and append the logging information
    log_flank += f"SP: [{sp_x}, {sp_y}]: "  # Source Point Position
    log_flank += " " * (COLL_SIZE - len(f"SP: [{sp_x}, {sp_y}]: "))  # Align columns

    log_flank += f"Rp step: {dx}, "  # Receiver Position step
    log_flank += " " * (WIDTH_FOR_RP - len(f"Rp step: {dx}, "))  # Align columns

    log_flank += f"CMP: [{cmp_x}, {cmp_y}], "  # Common MidPoint
    log_flank += " " * (COLL_SIZE - len(f"CMP: [{cmp_x}, {cmp_y}], "))  # Align columns

    log_flank += f"Ntr: {number_traces}, "  # Number of traces
    log_flank += " " * (WIDTH_FOR_RP - len(f"Ntr: {number_traces}, "))  # Align columns

    log_flank += f"Flank {flank_id} "

    log_flank += f"SNR: %.2f " % snr  # Value of SNR
    log_flank += " " * (WIDTH_FOR_RP - len(f"SNR: %.2f " % snr))  # Align columns

    log_flank += "\n"


    return float(cmp_x), float(cmp_y), log_flank  # Return coordinates and the formatted string


def create_table_preprocessing_3d_csp_no_mpi(item: int, header: np.ndarray, snr: float) -> tuple[int, int, str]:
    """
    Generates a formatted string containing preprocessing information for a
    single sector of 3D CSP (Common Source Point) seismic data.

    This function extracts relevant information from a seismic data header
    (specifically for 3D CSP data), formats it into a string, and returns the
    string along with the FFID and sector ID.

    Args:
       item (int): The ID of the sector being processed.
       header (np.ndarray): A NumPy array containing seismic data
                                  header information for a single sector.
       snr (float): The Signal-to-Noise Ratio for the current sector.

    Returns:
       Tuple[int, int, str]: A tuple containing:
           - ffid (int): The Field File ID.
           - item (int): The sector ID (same as the input `item`).
            - sectors (str): A formatted string containing preprocessing
                             information about the sector, including the FFID,
                             sector number, number of traces (Ntr), receiver
                             position step (RP step), source position (SP)
                             coordinates, CMP coordinates, and SNR. The
                             string is formatted with aligned columns for
                             improved readability.
    """

    ffid = int(header[HEADER_FFID_IND][0])  # file ID
    number_traces = len(header[0])  # number of traces
    rp_step = define_spatial_step(header)  # Receiver position step
    sp_x, sp_y = float(header[HEADER_SOU_X_IND][0]), float(header[HEADER_SOU_Y_IND][0])  # Source point coordinates
    cmp_x, cmp_y = float(header[HEADER_CDP_X_IND][0]), float(header[HEADER_CDP_Y_IND][0])  # CMP coordinates

    # Format and append the logging information
    sectors = f"FFID: {ffid} "  # File ID
    sectors += " " * (WIDTH_FOR_RP - len(f"FFID: {ffid}"))  # Align columns

    sectors += f"sector: {item} "  # Sector Number
    sectors += " " * (WIDTH_FOR_RP - len(f"sector # {item}"))  # Align columns

    sectors += f"Ntr: {number_traces}"  # Number of traces
    sectors += " " * (WIDTH_FOR_RP - len(f"Ntr: {number_traces}"))  # Align columns

    sectors += f"RP step: {rp_step} "  # Receiver position step
    sectors += " " * (WIDTH_FOR_RP - len(f"RP step {rp_step} "))  # Align columns

    sectors += f"SP: [{sp_x}, {sp_y}] "  # Source position coordinates
    sectors += " " * (COLL_SIZE - len(f"SP [{sp_x}, {sp_y}], "))  # Align columns

    sectors += f"CMP: [{cmp_x} {cmp_y}] "  # CMP Coordinates
    sectors += " " * (COLL_SIZE - len(f"CMP [{cmp_x} {cmp_y}],"))  # Align columns

    sectors += f"SNR: %.2f " % snr
    sectors += " " * (WIDTH_FOR_RP - len(f"SNR: %.2f " % snr))

    sectors += "\n"
    return ffid, item, sectors  # Return values and the string


def create_table_preprocessing_3d_cdp_no_mpi(expand_headers: np.ndarray, snr: float) -> tuple[float, float, str]:
    """
    Generates a formatted string containing preprocessing information for a
    single CDP (Common Depth Point) in 3D seismic data.

    This function extracts relevant information from a seismic data header
    (specifically for 3D data), formats it into a string, and returns the
    string along with the CDP coordinates.

    Args:
        expand_headers (np.ndarray): A NumPy array containing seismic data
                                     header information.
        snr (float): The Signal-to-Noise Ratio for the current CDP.

    Returns:
        Tuple[float, float, str]: A tuple containing:
            - cdp_x (float): The X coordinate of the CDP.
            - cdp_y (float): The Y coordinate of the CDP.
            - sectors (str): A formatted string containing the CDP coordinates,
                              receiver position step, number of traces, and
                              SNR. The string is formatted with aligned
                              columns for improved readability.
    """
    cdp_x, cdp_y = float(expand_headers[HEADER_CDP_X_IND][0]), float(expand_headers[HEADER_CDP_Y_IND][0])  # CDP Coordinates
    rp_step = define_spatial_step(expand_headers)  # Receiver position step
    number_traces = len(expand_headers[HEADER_OFFSET_IND])  # Number of traces


    # Format and append the logging information
    sectors = f"CDP: [{cdp_x}, {cdp_y}]: "  # Common Deep Point
    sectors += " " * (COLL_SIZE - len(f"CDP: [{cdp_x}, {cdp_y}]: "))  # Align columns

    sectors += f"RP step: {rp_step}, "  # Receiver position step
    sectors += " " * (WIDTH_FOR_RP - len(f"RP step: {rp_step}"))  # Align columns

    sectors += f"Ntr: {number_traces}"  # Number of traces
    sectors += " " * (WIDTH_FOR_RP - len(f"Ntr: {number_traces}"))

    sectors += f"SNR: %.2f " % snr
    sectors += " " * (WIDTH_FOR_RP - len(f"SNR: %.2f " % snr))

    sectors += "\n"
    return cdp_x, cdp_y, sectors  # Return values and the formatted string



def reformating_log_data_2d(config_parameters, file_path: Path) -> tuple[dict, Message]:
    """
    Reformats log data after processing a 2D seismic data file.

    This function updates the log data based on the preprocessing results of a
    2D seismic data file. It checks if any seismograms were found within the
    specified offset interval, updates counters and messages accordingly, and
    then reformats and sorts the log data for a more organized output.

    Args:
        config_parameters: An object containing configuration parameters and
                           log data. Must contain attributes named
                           'log_data', 'log_data_preprocessing', 'offset_min',
                           'offset_max', and 'error'.  This object will
                           be updated in place.
        file_path (Path): The path to the processed seismic data file.

    Returns:
        Tuple[dict, Message]: A tuple containing:
            - log_data (dict): The updated log data dictionary.
            - error (Message): The Message object, potentially updated with
                               a warning message if no seismograms were found.
    """
    config_parameters.log_data["Read seismogram files"] += 1  # Increment the count of read files

    count_sources_in_file_path = 0
    for cor_data in config_parameters.log_data_preprocessing:
        count_sources_in_file_path += cor_data[1]  # Sum the counts of sources

    count_saved_curves = 0
    for cor_data in config_parameters.log_data_preprocessing:
        count_saved_curves += cor_data[2]

    if count_sources_in_file_path == 0:
        # If no seismograms found in the offset interval, set a warning
        config_parameters.error.is_warning = True
        config_parameters.error.message += (f"\n{file_path} Not found seismograms in offset interval: "
                              f"[{config_parameters.offset_min}, {config_parameters.offset_max}]")
    else:
        # If seismograms were found, update the log data
        config_parameters.log_data["Preprocessed seismograms"] += count_sources_in_file_path
        config_parameters.log_data["System"] += f"\n{file_path}\n"
        config_parameters.log_data["Saved dispersion curves"] += count_saved_curves

        # Iterate through the flank tables (Right flank, Left flank)
        system_table = []
        first_sort_table = []
        second_sort_table = []
        third_sort_table = []

        for cor_table in config_parameters.log_data_preprocessing:  # loop with log data
            for i in range(len(cor_table[0][0])):  # loop thought the log with the index
                system_table.append(cor_table[0][0][i])  # Append the data
                first_sort_table.append(cor_table[0][1][i])  # Append the data
                second_sort_table.append(cor_table[0][2][i])  # add the data
                third_sort_table.append(cor_table[0][3][i])

        indexes_sort = np.lexsort([np.array(third_sort_table), np.array(second_sort_table), np.array(first_sort_table)])  # sort index
        system_table = np.array(system_table)[indexes_sort]  # sort

        for line in system_table:  # loop thought the system table
            config_parameters.log_data['System'] += line  # add the line in data

    return config_parameters.log_data, config_parameters.error  # Return the updated log data


def reformating_log_data_3d(config_parameters, file_path: Path) -> tuple[dict, Message]:
    """
        Reformats log data after processing a 3D seismic data file.

        This function updates the log data based on the preprocessing results of a
        3D seismic data file. It checks if any seismograms were found, updates
        counters and messages accordingly, and then reformats and sorts the log
        data for a more organized output. The behavior differs slightly based on
        the `sort_3d_order` configuration parameter.

        Args:
            config_parameters: An object containing configuration parameters and
                               log data.  Must contain attributes called
                               'log_data', 'log_data_preprocessing', 'offset_min',
                               'offset_max', 'sort_3d_order', 'num_sectors', and
                               'error'.  This object will be updated in place.
            file_path (Path): The path to the processed seismic data file.

        Returns:
            Tuple[dict, Message]: A tuple containing:
                - log_data (dict): The updated log data dictionary.
                - error (Message): The Message object, potentially updated with a
                                   warning message if no seismograms were found.
        """
    config_parameters.log_data["Read seismogram files"] += 1
    all_count_seism = 0
    for core_table in config_parameters.log_data_preprocessing:
        all_count_seism += core_table[3]

    all_count_saved_curves = 0
    for cor_table in config_parameters.log_data_preprocessing:
        all_count_saved_curves += cor_table[4]

    if all_count_seism == 0:  # Verify the quantity of the number
        config_parameters.error.is_warning = True  # the flag warning for value
        if config_parameters.sort_3d_order == 'csp':  # validation of the data order
            # Add the message
            config_parameters.error.message += (f"\n{file_path} Not found seismograms in offset interval:"
                                   f" [{config_parameters.offset_min}, {config_parameters.offset_max}] "
                                                f"and NS {config_parameters.num_sectors}")
        else:
            # Add the message
            config_parameters.error.message += (f"\n{file_path} Not found seismograms in offset interval: "
                                   f"[{config_parameters.offset_min}, {config_parameters.offset_max}] "
                                                f"for all CDP seismograms")
    else:
        system_table = []  # create list
        first_sort_table = []  # create list
        second_sort_table = []  # create list
        config_parameters.log_data['Preprocessed seismograms'] += all_count_seism  # add log
        config_parameters.log_data["Saved dispersion curves"] += all_count_saved_curves
        for core_table in config_parameters.log_data_preprocessing:  # lopp thought the data
            for i in range(len(core_table[0])):  # range the data
                system_table.append(core_table[0][i])  # add the list
                first_sort_table.append(core_table[1][i])  # add the list
                second_sort_table.append(core_table[2][i])  # add the list

        indexes_sort = np.lexsort([np.array(second_sort_table), np.array(first_sort_table)])  # Sort data
        system_table = np.array(system_table)[indexes_sort]  # Assign data with index

        config_parameters.log_data['System'] += f"\n{file_path}\n"  # Add File path to Log

        for line in system_table:  # Loop in list
            config_parameters.log_data['System'] += line  # Write line
    return config_parameters.log_data, config_parameters.error


