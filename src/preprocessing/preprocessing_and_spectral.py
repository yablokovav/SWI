# import python libraries
from collections.abc import Generator  # For type hinting generators
from typing import Optional, Any  # For optional type hints
import numpy as np  # For numerical operations
from joblib import Parallel, delayed
from tqdm import tqdm  # For progress bars

# import constants
from src import *

# import modules for logging
from src.logs.Message import Message
from src.logs.utils import (
    create_table_preprocessing_3d_csp,
    create_table_preprocessing_3d_cdp,
    log_one_flank,
    reformating_log_data_2d,
    reformating_log_data_3d,
    create_log,
)
from src.config_reader.Checker.exceptions import InvalidConfigurationParameters

# import models of parameters
from src.config_reader.models import PreprocessingModel, SpectralModel

# import modules for preprocessing
from src.preprocessing.config import name_headers
import src.preprocessing.utils as utils
from src.files_processor.readers import get_filenames, read_segy, get_indexes4read

# imports modules for spectral analysis
from src.spectral_analysis.spectral_transform.fk import FKT
from src.spectral_analysis.spectral_transform.sfk import SFK
from src.spectral_analysis.spectral_transform.transformer import TransformerFK2VF
from src.spectral_analysis.peakers.max_peaker import PeakerMax
from src.spectral_analysis.peakers.hdbscan_peaker import PeakerHDBSCAN
from src.config_reader.enums import SpectralMethod, ExtractDCMethod




class SeismicPreprocessorSpectral:

    def __init__(
            self,
            preprocessing: PreprocessingModel,
            save_dir_preprocessing: list[Path],
            save_dir_spectral: list[Path],
            fk_transform: [SFK, FKT],
            vf_transform: TransformerFK2VF,
            peaker: [PeakerMax, PeakerHDBSCAN],
            spectral: SpectralModel,
            endian: str = "big",
    ) -> None:
        """
        Initializes the seismic data processor.

        This method initializes the `SeismicDataProcessor` with the necessary
        configurations and parameters for preprocessing and spectral analysis of
        seismic data.

        Args:
            preprocessing (PreprocessingModel): Parameters for data preprocessing.
            save_dir_preprocessing (List[Path]): List of paths to save preprocessed data.
            save_dir_spectral (List[Path]): List of paths to save spectral analysis results.
            fk_transform (Union[SFK, FKT]): The FK transform method to use (SFK or FKT).
            vf_transform (TransformerFK2VF): The transform for FK to VF.
            peaker (Union[PeakerMax, PeakerHDBSCAN]): The method to extract the DC component from the power spectrum.
            spectral (SpectralModel): Parameters for spectral analysis.
            endian (str, optional): The endianness of the seismic data. Defaults to "big".
        """

        # Initialize logging data
        self.log_data_preprocessing = []
        self.error = Message(is_error=False, is_warning=False, message="")
        self.log_data = {
            "Read seismogram files": 0,
            "Data dimension": preprocessing.type_data.value,
            "Sorting type": preprocessing.parameters_3d.sort_3d_order.value,
            "Preprocessed seismograms": 0,
            "System": "\n",
            "Saved dispersion curves": 0,
        }

        # Assign preprocessing configurations to attributes
        self.ffid_start = preprocessing.ffid_start
        self.ffid_stop = preprocessing.ffid_stop
        self.ffid_increment = preprocessing.ffid_increment
        self.path4ffid_file = preprocessing.path4ffid_file
        self.num_sources_on_cpu = preprocessing.num_sources_on_cpu
        self.data_dir = preprocessing.data_dir
        self.save_dir_preprocessing = save_dir_preprocessing
        self.type_data = preprocessing.type_data
        self.offset_min = preprocessing.offset_min
        self.offset_max = preprocessing.offset_max
        self.qc_preprocessing = preprocessing.qc_preprocessing
        self.user_snr = preprocessing.snr
        self.endian = endian

        # 3d parameters
        self.sort_3d_order = preprocessing.parameters_3d.sort_3d_order.value
        self.num_sectors = preprocessing.parameters_3d.num_sectors
        self.bin_size_x = preprocessing.parameters_3d.bin_size_x
        self.bin_size_y = preprocessing.parameters_3d.bin_size_y

        # Assign configurations for spectral analysis
        self.fk_transform = fk_transform
        self.vf_transform = vf_transform

        self.peaker = peaker
        self.save_dir_spectral = save_dir_spectral
        self.qc_spectral = spectral.qc_spectral
        self.vmin = spectral.vmin
        self.vmax = spectral.vmax

        # advanced parameters
        self.peak_fraction = spectral.advanced.peak_fraction
        self.cutoff_fraction = spectral.advanced.cutoff_fraction
        self.dc_error_thr = spectral.advanced.dc_error_thr

        # Initialize other attributes as None
        self.gen_files: Optional[Generator[Path, None, None]] = None
        self.dir2save: Optional[Path] = None
        self.data_partition: Optional[utils.data_partition] = None

    @classmethod
    def open(
            cls,
            preprocessing_params: PreprocessingModel,
            spectral: SpectralModel,
            save_dirs: dict[str, list[Path]],
            endian: str = "big",
    ) -> "SeismicPreprocessorSpectral":
        """
        Initializes a SeismicPreprocessorSpectral instance for seismic data processing.

        This class method initializes a `SeismicPreprocessorSpectral` instance based on
        the provided configuration parameters, spectral analysis settings, and save
        directories. It sets up components for DC extraction and FK transformation and
        cleans the specified directories.

        Args:
            preprocessing_params (PreprocessingModel): Parameters related to the
                                                    preprocessing steps.
            spectral (SpectralModel): Parameters related to spectral analysis.
            save_dirs (Dict[str, List[Path]]): A dictionary containing lists of
                                                `Path` objects for saving preprocessed
                                                data and spectral analysis results.
                                                It should have keys 'preprocessing' and
                                                'spectral_analysis'.
            endian (str, optional): The endianness of the seismic data. Defaults to "big".

        Returns:
            SeismicPreprocessorSpectral: An instance of the `SeismicPreprocessorSpectral` class,
                                        ready for processing seismic data.

        Notes:
            This method cleans all files in the directories specified in `save_dirs`.  Make
            sure these directories are dedicated to this process and do not contain
            important data.
        """
        # Initialize peaker based on the configuration
        if spectral.extract_dc_method == ExtractDCMethod.max:
            peaker = PeakerMax.initialize(spectral.path4dc_limits)
        else:
            peaker = PeakerHDBSCAN.initialize(spectral.path4dc_limits)  # type: ignore

        # Initialize FK transform based on the configuration
        if spectral.spectral_method == SpectralMethod.fkt:
            fk_transform = FKT(spectral)
        else:
            fk_transform = SFK(spectral)

        # Create an instance of the class
        instance = cls(
            preprocessing_params,
            save_dirs['preprocessing'],
            save_dirs["spectral_analysis"],
            fk_transform,
            TransformerFK2VF(),
            peaker,
            spectral,
            endian
        )


        # Remove all files in the specified directories
        for dir_ in save_dirs['preprocessing']:
            for item in dir_.glob("*"):
                item.unlink()  # Delete each file

        for dir_ in save_dirs["spectral_analysis"]:
            for item in dir_.glob("*"):
                item.unlink()  # Delete each file

        return instance

    def _parallel_preprocessing_section(self, file_path: Path, indexes4read: list[int]) \
            -> list[list[str| float], Any] | list[str| float]:
        """
        Performs data division and preprocessing based on the data sorting order.

        This method reads seismic data using the `read_segy` function, then
        performs data division and preprocessing based on the sorting order
        specified by `self.type_data` and `self.sort_3d_order`. Different preprocessing
        functions are called based on whether the data is 2D or 3D, and for 3D
        data, whether it's sorted by CSP or CDP.

        Args:
            file_path (Path): The path to the seismic data file.
            indexes4read (List[int]): A list of trace indexes to read from the
                                        file.

        Returns:
            Union[List[List[Union[str, float]], Any], List[List[Union[str, float]]]]:
            Returns the processed data and associated metadata depending on the processing function.
            - If self.type_data == '2d': Returns [log_table, count]
            - If self.sort_3d_order == 'csp': Returns [log_sources, log_ffid, log_ns, count_seism]
            - Else (CDP): Returns [log_cdp, log_cdp_y, log_cdp_x, count_seism]

        Notes:
            The behavior of this function depends heavily on the attributes of `self`,
            including `self.type_data`, `self.sort_3d_order`, and the functions
            `_partdata_spectral_2d`, `_partdata_spectral_3d_csp`, and
            `_partdata_spectral_3d_cdp`.
        """
        sorted_traces, sorted_attributes, dt = read_segy(
            file_path,
            name_headers,
            indexes4read=indexes4read,
            type_data=self.type_data,
            sort_3d_order=self.sort_3d_order,
            endian=self.endian,
            bin_size=(self.bin_size_x, self.bin_size_y),
        )
        # Perform data division and preprocessing based on the sorting order (CSP or CDP)
        if self.type_data == '2d':
            log_table, count, curve_count = self._partdata_spectral_2d(file_path, sorted_traces, sorted_attributes, dt)
            return [log_table, count, curve_count]
        if self.sort_3d_order == 'csp':
            # process the seismic data for csp
            log_sources, log_ffid, log_ns, count_seism, curve_count = self._partdata_spectral_3d_csp(file_path, sorted_traces, sorted_attributes, dt)
            return [log_sources, log_ffid, log_ns, count_seism, curve_count]  # return the process log data
        else:
            # process the seismic data for cdp
            log_cdp, log_cdp_y, log_cdp_y, count_seism, curve_count = self._partdata_spectral_3d_cdp(file_path, sorted_traces, sorted_attributes, dt)
            return [log_cdp, log_cdp_y, log_cdp_y, count_seism, curve_count]  # return the process log data

    def _partdata_spectral_2d(self, file_path: Path, traces: np.ndarray, headers: np.ndarray, dt: float) -> tuple[
        list[list[str | int]], int, int]:
        """
        Processes 2D seismic data to apply spectral analysis and generate logs.

        This method divides the input seismic data into segments based on unique source coordinates so and
         then applies spectral processing to each segment. It generates logs with
         information about the processing results.

        Args:
            file_path (Path): The path to the seismic data file.
            traces (np.ndarray): NumPy array of seismic traces.
            headers (np.ndarray): NumPy array of seismic headers.
            dt (float): The time sampling interval.

        Returns:
            Tuple[List[List[Union[str, float]]], int]: A tuple containing:
                - log_table (List[List[Union[str, float]]]): A list containing the
                  logs with processing results. Each element is a list.
                - count (int): The number of data segments that were processed.
        """
        count = 0  # segment counter
        curves_count = 0

        # Define directories for right flank spectral analysis
        table, table_cmp_x, table_cmp_y, table_flank = [], [], [], [] # to store logs

        for curr_traces, curr_headers in utils.data_partition(traces, headers, sort_3d_order=""):

            # Determine the primary coordinate direction
            primary_key_idx, secondary_key_idx = utils.define_direction(curr_headers, HEADER_REC_X_IND, HEADER_REC_Y_IND)

            # Calculate offsets based on the primary and secondary coordinates
            offset = curr_headers[secondary_key_idx] - curr_headers[primary_key_idx]

            # Process the both sides (left and right) of shot gathers

            for flank_id, sign_offset in enumerate([-1, 1]):

                cut_seismic, cut_headers = utils.get_part_data(
                    primary_key_idx,
                    curr_traces,
                    curr_headers,
                    offset * sign_offset,
                    self.offset_min * sign_offset,
                    self.offset_max * sign_offset
                )

                if cut_seismic is not None:
                    count += 1 # collect amount of the processed shot gathers

                    # get signal-to-noise ratio for current seismogram
                    snr = utils.get_snr(cut_seismic, dt, cut_headers[HEADER_OFFSET_IND], self.vmin, self.vmax)

                    if snr >= self.user_snr:
                        valid_modes, cut_headers = utils.apply_spectral_processing(
                            self,
                            file_path,
                            cut_seismic,
                            cut_headers,
                            dt,
                            flank_id=flank_id,
                        )
                    else:
                        valid_modes = True

                    log_data = log_one_flank(cut_headers, flank_id, snr)  # add one log for data of one flank
                    table.append(log_data[2])  # append the log data
                    if not valid_modes:
                        table[-1] = table[-1][:-2] + " Warning: no found valid modes\n"
                    elif snr < self.user_snr:
                        table[-1] = table[-1][:-2] + " SNR is too low\n"
                    else:
                        curves_count += 1
                    table_cmp_x.append(log_data[0])  # log the cmp_x
                    table_cmp_y.append(log_data[1])  # log the cmp_y
                    table_flank.append(flank_id)  # log the flank id

        log_table = [table, table_cmp_x, table_cmp_y, table_flank]  # List Table

        return log_table, count, curves_count  # return values

    def _partdata_spectral_3d_csp(self, file_path: Path, sorted_traces: np.ndarray, sorted_attributes: np.ndarray, dt: float) \
            -> tuple[list[str], list[int], list[int], int, int]:
        """
        Performs spectral processing on 3D seismic data sorted in CSP order.

        This method divides the input seismic data into segments based on unique source coordinates and divide them
        into sectors based on receivers coordinates, applies spectral analysis to each sector, and generates logs.

        Args:
            file_path (Path): The path to the seismic data file.
            sorted_traces (np.ndarray): NumPy array of sorted seismic traces.
            sorted_attributes (np.ndarray): NumPy array of sorted seismic headers/attributes.
            dt (float): The time sampling interval.

        Returns:
            Tuple[List[str], List[int], List[int], int]: A tuple containing:
                - log_sources (List[str]): A list of strings containing source logs.
                - log_ffid (List[int]): A list of integers containing Field File IDs.
                - log_ns (List[int]): A list of integers containing sector numbers.
                - all_count (int): The total number of processed sectors.
        """
        all_count = 0  # Initialize total segment counter
        curves_count = 0  # counter for valid curves
        log_sources = []  # add Log type
        log_ffid = []  # add Log type
        log_ns = []  # add Log type
        # Iterate over partitions of seismic data using tqdm for progress bar
        for curr_traces, curr_headers in utils.data_partition(sorted_traces, sorted_attributes, sort_3d_order=self.sort_3d_order):

            curr_traces, curr_headers = utils.seismogram_without_large_values(curr_traces,
                                                                              curr_headers,
                                                                              self.offset_min,
                                                                              self.offset_max)


            if len(curr_headers[HEADER_OFFSET_IND]) < self.num_sectors:  # Reject if less than 10 traces
                continue

            alpha = utils.transform_to_polar(  # Transform to polar coordinates
                curr_headers[HEADER_REC_X_IND:HEADER_REC_Y_IND + 1],
                curr_headers[HEADER_SOU_X_IND:HEADER_SOU_Y_IND + 1]
            )

            labels = utils.weighted_clustering(  # Cluster receivers into sectors
                alpha, curr_headers[HEADER_OFFSET_IND], angle_weight=1.0, distance_weight=0.0, n_clusters=self.num_sectors
            )

            for item in range(self.num_sectors):  # Loop for each sector of num sector parameter

                cut_seismic = curr_traces[:, labels == item]  # select only seismic where labels == item
                cut_headers = curr_headers[:, labels == item]  # select only headers where labels == item

                if len(cut_headers[HEADER_OFFSET_IND]) < 10:  # Reject if less than 10 traces
                    continue

                # Sort traces and headers by offset
                sort_indexes = np.argsort(cut_headers[HEADER_OFFSET_IND])
                cut_seismic = cut_seismic[:, sort_indexes]
                cut_headers = cut_headers[:, sort_indexes]

                # Compute common midpoint (CMP) coordinates and write them to headers
                centers = [
                    np.round(cut_headers[HEADER_REC_X_IND][0] + (cut_headers[HEADER_REC_X_IND][-1] - cut_headers[HEADER_REC_X_IND][0]) / 2),
                    np.round(cut_headers[HEADER_REC_Y_IND][0] + (cut_headers[HEADER_REC_Y_IND][-1] - cut_headers[HEADER_REC_Y_IND][0]) / 2)
                ]
                cut_headers[HEADER_CDP_X_IND:HEADER_CDP_Y_IND + 1, :] = np.repeat(
                    np.array(centers)[:, np.newaxis],
                    len(sort_indexes),
                    axis=1
                )

                # Отбраковка сейсмограмм по S / N
                snr = utils.get_snr(cut_seismic, dt, cut_headers[HEADER_OFFSET_IND], self.vmin, self.vmax)

                if snr >= self.user_snr:
                    # Spectral processing
                    valid_modes, cut_headers = utils.apply_spectral_processing(
                        self,
                        file_path,
                        cut_seismic,
                        cut_headers,
                        dt,
                        num_sector=item
                    )
                else:
                    valid_modes = True

                # Add segment information to the log
                ffid_log = create_table_preprocessing_3d_csp(item, cut_headers, snr)
                log_ffid.append(ffid_log[0])  # Add the File ID into object
                log_ns.append(ffid_log[1])  # Add the sectors number into object
                log_sources.append(ffid_log[2])  # Add log sources into object
                if not valid_modes:
                    log_sources[-1] = log_sources[-1][:-2] + " Warning: no found valid modes\n"
                elif snr < self.user_snr:
                    log_sources[-1] = log_sources[-1][:-2] + " SNR is too low\n"
                else:
                    curves_count += 1

                all_count += 1

        return log_sources, log_ffid, log_ns, all_count, curves_count  # Return segment log and total count

    def _partdata_spectral_3d_cdp(self, file_path: Path, sorted_traces: np.ndarray, sorted_attributes: np.ndarray, dt: float)\
            -> tuple[list[str], list[float], list[float], int, int]:
        """
        Performs spectral processing on 3D seismic data sorted in CDP order.

        This method divides input seismic data int segments based on unique Common depth point coordinates,
        applies spectral processing, and generating logs.

        Args:
            file_path (Path): The path to the seismic data file.
            sorted_traces (np.ndarray): NumPy array of sorted seismic traces.
            sorted_attributes (np.ndarray): NumPy array of sorted seismic headers/attributes.
            dt (float): The time sampling interval.

        Returns:
            Tuple[List[str], List[float], List[float], int]: A tuple containing:
                - log_cdp (List[str]): A list of strings containing CDP logs.
                - log_cdp_x (List[float]): A list of floats containing CDP X coordinates.
                - log_cdp_y (List[float]): A list of floats containing CDP Y coordinates.
                - count (int): The total number of processed CDPs.
        """
        count = 0  # Initialize segment counter
        curves_count = 0  # counter for valid curves
        log_cdp = []  # string list to log
        log_cdp_x = []  # float List the log
        log_cdp_y = []  # float List to log
        # Unpack spectral directories from the object
        # Iterate over partitions of seismic data using tqdm for progress bar
        for curr_traces, curr_headers in utils.data_partition(sorted_traces, sorted_attributes, sort_3d_order=self.sort_3d_order):

            # Reject the partition if the number of traces is less than 10
            if len(curr_headers[HEADER_OFFSET_IND]) < 10:
                continue

            # Filter traces and headers based on the offset range [offset_min, offset_max]
            curr_traces, curr_headers = utils.seismogram_without_large_values(curr_traces, curr_headers,
                                                                        self.offset_min, self.offset_max)

            # Reject the partition if the number of traces is less than 10 after offset filtering
            if len(curr_headers[HEADER_OFFSET_IND]) < 10:
                continue

            # Average traces with the same offset (stacking) to reduce noise
            curr_traces, curr_headers = utils.mean_traces_with_equal_offsets(curr_traces, curr_headers)

            # Reject the partition if the number of traces is less than 10 after stacking
            if len(curr_headers[HEADER_OFFSET_IND]) < 10:
                continue

            snr = utils.get_snr(curr_traces, dt, curr_headers[HEADER_OFFSET_IND], self.vmin, self.vmax)

            if snr >= self.user_snr:
                valid_modes = utils.apply_spectral_processing(
                    self,
                    file_path,
                    curr_traces,
                    curr_headers,
                    dt,
                )
            else:
                valid_modes = True

            # Add segment information to the log
            cdp_log = create_table_preprocessing_3d_cdp(curr_headers, snr)
            log_cdp_x.append(cdp_log[0])  # Append coordinate X to log CDP
            log_cdp_y.append(cdp_log[1])  # append coordinate y to log CDP
            log_cdp.append(cdp_log[2])  # Append the cdp log data
            if not valid_modes:
                log_cdp[-1] = log_cdp[-1][:-2] + " Warning: no found valid modes\n"
            elif snr < self.user_snr:
                log_cdp[-1] = log_cdp[-1][:-2] + " SNR is too low\n"
            else:
                curves_count += 1
            count += 1

        return log_cdp, log_cdp_x, log_cdp_y, count, curves_count  # Return segment log and total count

    def _update_log_data(self, file_path: Path) -> tuple[dict, Message]:
        """
        Updates the log data based on the data type.

        This method updates the log data by calling either
        `reformating_log_data_2d` or `reformating_log_data_3d` based on the
        value of `self.type_data`.

        Args:
            file_path (Path): The path to the file being processed.

        Returns:
            Tuple[dict, Message]: A tuple containing the updated log data
                                    (dictionary) and the error/message object.
        """
        if self.type_data == '2d':
            self.log_data, self.error = reformating_log_data_2d(self, file_path)
            return self.log_data, self.error
        else:
            self.log_data, self.error = reformating_log_data_3d(self, file_path)
            return self.log_data, self.error

    def _prepare_log_data(self):
        """
        Prepares the log data after seismic processing.

        This method finalizes the log data by removing irrelevant information
        based on the processing type, checking for errors, and adding
        information about where the processed data and images are stored. It
        also creates a log file and raises an exception if an error occurred
        during processing.

        Raises:
            InvalidConfigurationParameters: If no seismograms were found
                and the error flag is set.
        """
        if self.type_data == '2d':
            self.log_data.pop("Sorting type")
        if self.log_data['Preprocessed seismograms'] == 0:  # Verify to data if 0 data in seismic pre process
            self.error.is_error = True  # add log type
            if self.sort_3d_order == 'csp':  # Verify the data type with CSP
                self.error.message = (
                    f"Not found seismograms in offset interval "
                    f"[{self.offset_min}, {self.offset_max}] and NS {self.num_sectors}"
                )  # Add a message with the error
            else:
                self.error.message = (
                    f"Not found seismograms in offset interval "
                    f"[{self.offset_min}, {self.offset_max}] for all CDP seismograms"
                )  # # Add a message with the error
        elif self.log_data["Saved dispersion curves"] == 0:
            self.error.is_error = True
            self.error.message = "No dispersion curves have been saved, check logs"
        else:
            if self.qc_preprocessing:  # Add log type
                self.log_data[
                    "Preprocessed seismograms stored in"] = f"\n{self.save_dir_preprocessing[0]}"  # add type to save
            if self.qc_spectral:  # Verify t the qualiti control its enable
                # add an Image value to log type
                self.log_data["Images V-f spectra stored in"] = f"\n{self.save_dir_spectral[0]}"
                # add a segy values to the log
                self.log_data["Segy V-f spectra stored in"] = f"\n{self.save_dir_spectral[2]}"
            # Add the curve discretion to the log
            self.log_data["Dispersion curves stored in"] = f"\n{self.save_dir_spectral[1]}"

        create_log(self.log_data, self.save_dir_preprocessing[0].parents[2],
                   'preprocessing_and_spectral_analysing', self.error)  # create log to report and sent to next function
        if self.error.is_error:  # Verify that it's a log
            raise InvalidConfigurationParameters(self.error.message)  # send an exception
        print('Data segmentation done. Details in the log.' + '\n')

    def run(self) -> None:
        """
        Executes the seismic data processing pipeline.

        This method iterates through seismic data files in a specified directory,
        reads the data based on provided indexing parameters, performs parallel
        preprocessing, updates the log data, and prepares the final log.

        It relies on helper functions for: getting filenames, generating indexes for
        reading specific data portions from a file, parallelized preprocessing,
        updating log data, and preparing a final log output.

        Notes:
            This function assumes all seismic files can fit in memory,
            as all traces are loaded in to local memory during processing.
        """
        for file_path in get_filenames(self.data_dir):
            indexes4read = get_indexes4read(
                self.path4ffid_file,
                file_path,
                ranges=(self.ffid_start, self.ffid_stop, self.ffid_increment),
                type_data=self.type_data,
                sort_3d_order=self.sort_3d_order,
                bin_size=(self.bin_size_x, self.bin_size_y),
                num_sources_on_cpu=self.num_sources_on_cpu
            )

            self.log_data_preprocessing = Parallel(n_jobs=-1)(
                delayed(self._parallel_preprocessing_section)(file_path, indexes4read[i])
                for i in tqdm(range(len(indexes4read)))
            )

            self.log_data, self.error = self._update_log_data(file_path)  # Update log
        self._prepare_log_data()