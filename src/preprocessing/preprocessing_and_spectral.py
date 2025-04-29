# import python libraries
from collections.abc import Generator  # For type hinting generators
from typing import Optional  # For optional type hints
from mpi4py import MPI
from tqdm import tqdm
import numpy as np  # For numerical operations
# import constants
from src import *

# import modules for logging
from src.logs.Message import Message
from src.logs.utils import (
    create_table_preprocessing_3d_csp,
    create_table_preprocessing_3d_cdp,
    log_one_flank,
    setup_loger,
    close_logger
)
from src.config_reader.Checker.exceptions import InvalidConfigurationParameters

# import models of parameters
from src.config_reader.models import PreprocessingModel, SpectralModel

# import modules for preprocessing
from src.preprocessing.config import name_headers
import src.preprocessing.utils as utils
from src.files_processor.readers import get_filenames, read_segy, seismic_trace_generator

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
        self.count_seismogram_files = 0
        self.count_preprocessed_seismograms = 0
        self.count_dispersion_curves = 0

        self.current_count_preprocessed_seismograms = 0
        self.current_count_dispersion_curves = 0

        # Assign preprocessing configurations to attributes
        self.ffid_start = preprocessing.ffid_start
        self.ffid_stop = preprocessing.ffid_stop
        self.ffid_increment = preprocessing.ffid_increment
        self.scaler_to_elevation = preprocessing.scaler_to_elevation
        self.scaler_to_coordinates = preprocessing.scaler_to_coordinates
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

        # MPI parallel options
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.status = MPI.Status()


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

        # # Remove all files in the specified directories
        # for dir_ in save_dirs['preprocessing']:
        #     for item in dir_.glob("*"):
        #         item.unlink()  # Delete each file
        #
        # for dir_ in save_dirs["spectral_analysis"]:
        #     for item in dir_.glob("*"):
        #         item.unlink()  # Delete each file

        return instance

    def _partdata_spectral_2d(self, file_path: Path, curr_traces: np.ndarray, curr_headers: np.ndarray, dt: float) \
            -> tuple[list[str], int, int]:
        """
        Processes 2D seismic data to apply spectral analysis and generate logs.

        This method divides the input seismic data into flanks based on receivers coordinates so and
         then applies spectral processing to each flank. It generates logs with
         information about the processing results.

        Args:
            file_path (Path): The path to the seismic data file.
            curr_traces (np.ndarray): NumPy array of seismic traces.
            curr_headers (np.ndarray): NumPy array of seismic headers.
            dt (float): The time sampling interval.

        Returns:
            Tuple[List[str], int, int]: A tuple containing:
                - table_flanks (List[str]): A list containing the information about each flank.
                - seism_count (int): The number of flanks that were processed.
                - curves_count (int): The number of curves that were extracted.
        """
        seism_count = 0  # segment counter
        curves_count = 0

        # Define directories for right flank spectral analysis
        table_flanks = [] # to store logs

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

            if cut_seismic is not None and len(cut_headers[HEADER_OFFSET_IND]) > 10:
                seism_count += 1 # collect amount of the processed shot gathers

                valid_modes, snr, cut_headers = utils.apply_spectral_processing(
                    self,
                    file_path,
                    cut_seismic,
                    cut_headers,
                    dt,
                    flank_id=flank_id,
                )

                table_flanks.append(log_one_flank(cut_headers, flank_id, snr))  # add one log for data of one flank
                if not valid_modes:
                    table_flanks[-1] = table_flanks[-1] + " Warning: no found valid modes"
                elif snr < self.user_snr:
                    table_flanks[-1] = table_flanks[-1] + " SNR is too low"
                else:
                    curves_count += 1 # List Table

        return table_flanks, seism_count, curves_count  # return values

    def _partdata_spectral_3d_csp(self, file_path: Path, curr_traces: np.ndarray, curr_headers: np.ndarray, dt: float) \
            -> tuple[list[str], int, int]:
        """
        Performs spectral processing on 3D seismic data sorted in CSP order.

        This method divides the input seismic data sectors based on receivers coordinates,
        applies spectral analysis to each sector, and generates logs.

        Args:
            file_path (Path): The path to the seismic data file.
            curr_traces (np.ndarray): NumPy array of seismic traces.
            curr_headers (np.ndarray): NumPy array of seismic headers.
            dt (float): The time sampling interval.

        Returns:
            Tuple[List[str], int, int]: A tuple containing:
                - log_sectors (List[str]): A list of strings containing segments logs.
                - seism_count (int): Counts of valid sectors.
                - curves_count (int): Counts of extracted desperation curves.
        """
        seism_count = 0  # Initialize total segment counter
        curves_count = 0  # counter for valid curves
        log_sectors = []
        # Iterate over partitions of seismic data using tqdm for progress bar
        curr_traces, curr_headers = utils.seismogram_without_large_values(curr_traces,
                                                                          curr_headers,
                                                                          self.offset_min,
                                                                          self.offset_max)

        if len(curr_headers[HEADER_OFFSET_IND]) < self.num_sectors:  # Reject if less than 10 traces
            return log_sectors, seism_count, curves_count

        sort_indexes = np.argsort(curr_headers[HEADER_OFFSET_IND])
        curr_traces, curr_headers = curr_traces[:, sort_indexes], curr_headers[:, sort_indexes]

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

            # Spectral processing
            valid_modes, snr, cut_headers = utils.apply_spectral_processing(
                self,
                file_path,
                cut_seismic,
                cut_headers,
                dt,
                num_sector=item
            )

            # Add segment information to the log
            log_sectors.append(create_table_preprocessing_3d_csp(item, cut_headers, snr))
            if not valid_modes:
                log_sectors[-1] = log_sectors[-1] + " Warning: no found valid modes"
            elif snr < self.user_snr:
                log_sectors[-1] = log_sectors[-1] + " Warning: SNR is too low"
            else:
                curves_count += 1

            seism_count += 1
        return log_sectors, seism_count, curves_count  # Return segment log and total count

    def _partdata_spectral_3d_cdp(self, file_path: Path, curr_traces: np.ndarray, curr_headers: np.ndarray, dt: float)\
            -> tuple[list[str], int, int]:
        """
        Performs spectral processing on 3D seismic data sorted in CDP order.

        This method applies spectral processing, and generating logs for CDP seismogram.

        Args:
            file_path (Path): The path to the seismic data file.
            curr_traces (np.ndarray): NumPy array of sorted seismic traces.
            curr_headers (np.ndarray): NumPy array of sorted seismic headers/attributes.
            dt (float): The time sampling interval.

        Returns:
            Tuple[List[str], int, int]: A tuple containing:
                - log_cdp (List[str]): A list of strings information about cdp seismogram.
                - seism_count (int): The total number of processed CDPs.
                - curves_count (int): The total number of extracted desperation curves.
        """
        seism_count = 0  # Initialize segment counter
        curves_count = 0  # counter for valid curves
        log_cdp = []  # string list to log
        # Unpack spectral directories from the object
        # Iterate over partitions of seismic data using tqdm for progress bar

        # Reject the partition if the number of traces is less than 10
        if len(curr_headers[HEADER_OFFSET_IND]) < 10:
            return log_cdp, seism_count, curves_count  # Return segment log and total count

        # Filter traces and headers based on the offset range [offset_min, offset_max]
        curr_traces, curr_headers = utils.seismogram_without_large_values(curr_traces, curr_headers,
                                                                    self.offset_min, self.offset_max)

        # Reject the partition if the number of traces is less than 10 after offset filtering
        if len(curr_headers[HEADER_OFFSET_IND]) < 10:
            return log_cdp, seism_count, curves_count  # Return segment log and total count

        # Average traces with the same offset (stacking) to reduce noise
        curr_traces, curr_headers = utils.mean_traces_with_equal_offsets(curr_traces, curr_headers)

        # Reject the partition if the number of traces is less than 10 after stacking
        if len(curr_headers[HEADER_OFFSET_IND]) < 10:
            return log_cdp, seism_count, curves_count  # Return segment log and total count

        sorted_indexes = np.argsort(curr_headers[HEADER_OFFSET_IND])
        curr_traces, curr_headers = curr_traces[:, sorted_indexes], curr_headers[:, sorted_indexes]

        valid_modes, snr, curr_headers = utils.apply_spectral_processing(
                self,
                file_path,
                curr_traces,
                curr_headers,
                dt,
            )

        # Add segment information to the log
        cdp_log = create_table_preprocessing_3d_cdp(curr_headers, snr)
        log_cdp.append(cdp_log)  # Append the cdp log data
        if not valid_modes:
            log_cdp[-1] = log_cdp[-1] + " Warning: no found valid modes"
        elif snr < self.user_snr:
            log_cdp[-1] = log_cdp[-1] + " SNR is too low"
        else:
            curves_count += 1
        seism_count += 1

        return log_cdp, seism_count, curves_count  # Return segment log and total count

    def _master(self):
        """
        Master process responsible for coordinating the preprocessing of seismogram files
        using multiple worker processes in an MPI environment.

        This method performs the following tasks:
        - Initializes logging.
        - Iterates through seismogram files in the data directory.
        - Distributes data to worker processes using a generator.
        - Collects results from workers and handles their completion.
        - Tracks and logs preprocessing statistics for each file.
        - Finalizes worker processes once all data is processed.
        """

        # Set up the logger and logging handler
        self.logger, self.handler = setup_loger(self)

        # Log data type and sorting order (if 3D data)
        self.logger.info(f"Data type: {self.type_data}")
        if self.type_data == "3d":
            self.logger.info(f"Sorting type: {self.sort_3d_order}")

        # Iterate over each file in the data directory
        for file_path in get_filenames(self.data_dir):
            print(f"Processing file: {file_path.name}", flush=True)
            self.logger.info(f"Processing file: {file_path.name}")
            self.count_seismogram_files += 1
            self.current_count_preprocessed_seismograms = 0
            self.current_count_dispersion_curves = 0

            # Track worker status (active/inactive)
            worker_status = {rank: False for rank in range(1, self.size)}

            # Initialize the generator for seismic trace reading
            indexes4read = seismic_trace_generator(
                self.path4ffid_file,
                file_path,
                ranges=(self.ffid_start, self.ffid_stop, self.ffid_increment)
            )

            # Distribute initial data chunks to all available workers
            for rank in range(1, self.size):
                try:
                    seismic_data = utils.step_on_generator(self, indexes4read, file_path)
                    self.comm.send(seismic_data, dest=rank, tag=1)
                    worker_status[rank] = True
                except StopIteration:
                    break  # No more data to distribute

            # Count how many workers are currently active
            active_workers = sum(worker_status.values())

            # Receive results from workers and continue distribution until all are done
            while active_workers > 0:
                log_res = self.comm.recv(source=MPI.ANY_SOURCE, tag=2, status=self.status)
                worker_rank = self.status.Get_source()

                # Log information returned by the worker
                for seismic_data in log_res[0]:
                    self.logger.info(str(seismic_data))

                # Update processing counts
                self.current_count_preprocessed_seismograms += log_res[1]
                self.current_count_dispersion_curves += log_res[2]

                try:
                    # Try to send more data to the worker, or mark them as done
                    seismic_data = utils.step_on_generator(self, indexes4read, file_path)
                    self.comm.send(seismic_data, dest=worker_rank, tag=1)
                except StopIteration:
                    active_workers -= 1  # No more data, this worker is done

            # Log warnings if no valid data was processed from the file
            if self.current_count_preprocessed_seismograms == 0:
                self.logger.warning(f"No found valid seismograms in file: {file_path.name}")
            elif self.current_count_dispersion_curves == 0:
                self.logger.warning(f"No one dispersion curves was saved from file: {file_path.name}")

            # Accumulate the overall processed data counts
            self.count_preprocessed_seismograms += self.current_count_preprocessed_seismograms
            self.count_dispersion_curves += self.current_count_dispersion_curves

        # Signal all workers to finalize
        self._finalize_workers()

        # Close the logger after processing
        close_logger(self)

    def _worker(self):
        """
        Worker process that receives seismic data from the master process,
        processes it based on the data type and sorting order,
        and sends the results back to the master.

        This method runs an infinite loop to:
        - Wait for data from the master.
        - Exit when a termination signal is received (tag 0).
        - Process data based on 2D or 3D configuration.
        - Send the processing results back to the master (tag 2).
        """
        while True:
            # Receive a task from the master
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)
            tag = self.status.Get_tag()

            if tag == 0:
                # Termination signal received
                break

            # Unpack the task: traces, headers, time step, and file path
            traces, headers, dt, file_path = task

            # Process data based on type and sorting method
            if self.type_data == "2d":
                log_res = self._partdata_spectral_2d(file_path, traces, headers, dt)
            elif self.sort_3d_order == "csp":
                log_res = self._partdata_spectral_3d_csp(file_path, traces, headers, dt)
            else:
                log_res = self._partdata_spectral_3d_cdp(file_path, traces, headers, dt)

            # Send the result back to the master
            self.comm.send(log_res, dest=0, tag=2)

    def _finalize_workers(self):
        """
        Sends termination signal (tag 0) to all worker processes.

        This is used in MPI-based processing to notify each worker that
        no more tasks will be sent, and they should shut down gracefully.
        """
        for rank in range(1, self.size):
            self.comm.send(None, dest=rank, tag=0)

    def _single_process_mode(self):
        """
        Processes all seismogram files in a single-process (non-MPI) mode.

        This method is useful for debugging, small datasets, or environments
        where MPI is not available. It reads seismic data, processes each
        chunk, and logs the results. Only the 3D CSP sorting method is used
        in this implementation.
        """

        # Set up logger
        self.logger, self.handler = setup_loger(self)

        # Log the type of data and sorting method
        self.logger.info(f"Data type: {self.type_data}")
        if self.type_data == "3d":
            self.logger.info(f"Sorting type: {self.sort_3d_order}")

        # Process each seismogram file in the directory
        for file_path in get_filenames(self.data_dir):
            print(f"Processing file: {file_path.name}", flush=True)
            self.count_seismogram_files += 1
            self.logger.info(f"Processing file :{file_path.name}")
            self.current_count_preprocessed_seismograms = 0
            self.current_count_dispersion_curves = 0

            # Generate index ranges for reading seismic traces
            indexes4read = seismic_trace_generator(
                self.path4ffid_file,
                file_path,
                ranges = (self.ffid_start, self.ffid_stop, self.ffid_increment)
            )

            # Read and process each set of traces
            for current_indexes in tqdm(indexes4read):
                traces, headers, dt = read_segy(
                    file_path,
                    name_headers,
                    indexes4read=current_indexes,
                    endian=self.endian,
                    elevation_scaler=self.scaler_to_elevation,
                    coordinates_scaler=self.scaler_to_coordinates,
                )

                # Process data based on type and sorting method
                if self.type_data == "2d":
                    log_res = self._partdata_spectral_2d(file_path, traces, headers, dt)
                elif self.sort_3d_order == "csp":
                    log_res = self._partdata_spectral_3d_csp(file_path, traces, headers, dt)
                else:
                    log_res = self._partdata_spectral_3d_cdp(file_path, traces, headers, dt)

                # Log information returned by the worker
                for seismic_data in log_res[0]:
                    self.logger.info(str(seismic_data))

                # Update processing counts
                self.current_count_preprocessed_seismograms += log_res[1]
                self.current_count_dispersion_curves += log_res[2]

            # Log warnings if no valid data was processed from the file
            if self.current_count_preprocessed_seismograms == 0:
                self.logger.warning(f"No found valid seismograms in file: {file_path.name}")
            elif self.current_count_dispersion_curves == 0:
                self.logger.warning(f"No one dispersion curves was saved from file: {file_path.name}")

            # Accumulate the overall processed data counts
            self.count_preprocessed_seismograms += self.current_count_preprocessed_seismograms
            self.count_dispersion_curves += self.current_count_dispersion_curves

        # Close the logger after processing
        close_logger(self)


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
        if self.size == 1:
            self._single_process_mode()
        elif self.rank == 0:
             self._master()
        else:
            self._worker()

        self.comm.Barrier()

        if self.rank == 0:
            if self.count_preprocessed_seismograms == 0 or self.count_dispersion_curves == 0:
                message = "No output data in module, check logs"
                raise InvalidConfigurationParameters(message)


