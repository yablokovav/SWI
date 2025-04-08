from collections.abc import Generator
from typing import Union

import numpy as np
import segyio
from src.preprocessing.utils import define_direction
from src import *  ### import globvar (indexes for headers)
from src.config_reader.models import PWaveVelocityModel, Ranges, DispersionCurve

MULTIPLIER_FOR_DT = 1e-6 # multiplier for time sampling step


def get_filenames(data_dir: Path, suffix: Union[tuple[str], str] = (".sgy", ".segy")) -> Generator[Path, None, None]:
    """
    Recursively retrieves filenames with specified suffixes from a directory.

    If `data_dir` is a file and its suffix is in `suffix`, it yields the file.
    If `data_dir` is a directory, it iterates through the files within and
    yields those with suffixes present in `suffix`.

    Args:
        data_dir (Path): Path to a file or directory containing files.
        suffix (Union[tuple[str, ...], str]): Suffix or suffixes to filter: (".sgy", ".segy").

    Yields:
        Generator[Path, None, None]: A generator of Path objects, each
                                     representing a file with the desired suffix.
    """
    if data_dir.is_file() and data_dir.suffix in suffix:
        yield data_dir
    elif data_dir.is_dir():
        # If it's a directory, iterate through the files inside.
        for file_path in data_dir.iterdir():
            if file_path.suffix in suffix:
                yield file_path

def get_ffid_header(header_file_path: Path, header_field: int, endian: str) -> np.ndarray:
    """
    Extracts a FFID header from a seismic data file (SEGY or ASCII).

    This function reads seismic data from a SEGY or SGY file, or reads a
    single header field from an ASCII file (specifically, an "Import Trace
    Headers to ASCII file" output from SeiSee).

    Args:
        header_file_path (Path): Path to the seismic data file.  This can be
                                  a SEGY/SGY file or an ASCII file in SeiSee
                                  "Import Trace Headers to ASCII file" format.
        header_field (int): The SEGY header field number to extract
                              (e.g., Field Record (FFID)). This is ignored if
                              the file is an ASCII file.
        endian (str): Endianness of the SEGY file ("<" for little-endian,
                      ">" for big-endian).  This is ignored if the file is
                      an ASCII file.

    Returns:
        np.ndarray: A NumPy array containing the values of the specified
                    header field for all traces in the seismic data file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If there's an issue reading the file format.
    """
    if header_file_path.suffix in (".segy", ".sgy"):
        with segyio.open(header_file_path.as_posix(), endian=endian, ignore_geometry=True) as segy_file:
            header_ffid = segy_file.attributes(header_field)[:]
    else:
        try:
            #  File format corresponds to the "Import Trace Headers to ASCII file" output from SeiSee for a single header field (Field Record (FFID) expected).
            header_ffid = np.loadtxt(header_file_path, usecols=1)
        except ValueError:
            raise ValueError("Data in file with FFID numbers has incorrect format, check it.")
    return header_ffid

def convert_cdp2ffid(header_file_path: Path,
                    header_field: list[int],
                    bin_size: tuple[float, float],
                    endian: str = 'big') -> np.ndarray:
    """
    Converts CDP (Common Depth Point) coordinates to unique ID for every CDP based on binning.

    This function reads header information from a seismic data file (SEG-Y or ASCII),
    bins the CDP coordinates based on the provided bin size, and assigns a unique
    ID to each bin.

    Args:
        header_file_path (Path): Path to the seismic data file (SEG-Y or ASCII).
        header_field (List[int]): A list of two header fields corresponding to the
                                  CDP X and CDP Y coordinates.
        bin_size (Tuple[int, int]): A tuple containing the bin size for the X and Y
                                    coordinates, respectively.
        endian (str, optional): The endianness of the SEG-Y file ('big' or 'little').
                                Defaults to 'big'.

    Returns:
        np.ndarray: A NumPy array containing the assigned FFID for each trace,
                    indexed from 1.
    """
    if header_file_path.suffix in (".segy", ".sgy"):
        with segyio.open(header_file_path.as_posix(), endian=endian, ignore_geometry=True) as segy_file:
            headers_cdp = np.array([segy_file.attributes(field)[:] for field in header_field])/100
    else:
        try:
            #  File format corresponds to the "Import Trace Headers to ASCII file" output from SeiSee for a single header field (Field Record (FFID) expected).
            headers_cdp = np.loadtxt(header_file_path, usecols=(1, 2))/100
        except ValueError:
            raise ValueError("Data in file with CDP coordinates has incorrect format, check it.")

    headers_cdp[:, 0] = np.int32(headers_cdp[:, 0]/bin_size[0]) * bin_size[0] + bin_size[0]/2
    headers_cdp[:, 1] = np.int32(headers_cdp[:, 1]/bin_size[1]) * bin_size[1] + bin_size[1]/2

    unique_cdp = np.unique(headers_cdp, axis=0)
    cdp_bin_index = np.zeros(len(headers_cdp), dtype=int)
    for cdp_ind in range(len(unique_cdp)):
        local_ind_cdp = np.where((headers_cdp[:, 0] == unique_cdp[cdp_ind, 0]) &
                                 (headers_cdp[:, 1] == unique_cdp[cdp_ind, 1]))[0]
        cdp_bin_index[local_ind_cdp] = cdp_ind+1

    return cdp_bin_index

def get_parameters4parallel(unique_ffid: np.ndarray, num_sources_on_cpu: int) -> tuple[list, list, int]:
    """
    Calculates parameters for parallel processing of seismic data based on
    unique Field Record IDs (FFIDs).

    This function determines the number of CPU cores to use for parallel
    processing and generates start and stop FFIDs for each core, aiming to
    distribute the workload evenly.

    Args:
        unique_ffid (np.ndarray): A NumPy array containing unique FFIDs
                                 (Field Record IDs).
        num_sources_on_cpu (int): The desired number of sources to process
                                  per CPU core. If 0, processing will be
                                  done on a single core.

    Returns:
        Tuple[List[int], List[int], int]: A tuple containing:
            - ffid_start (List[int]): A list of the starting FFIDs for each CPU core.
            - ffid_stop (List[int]): A list of the ending FFIDs for each CPU core.
            - num_cpu (int): The calculated number of CPU cores to use.  Will be at least 1.
    """
    print(f"Number of unique sources: {len(unique_ffid)}")
    if num_sources_on_cpu == 0:
        num_cpu = 1
    else:
        num_cpu = int(round(len(unique_ffid) / num_sources_on_cpu))
        if num_cpu == 0:
            num_cpu = 1
    print(f"Common amount of iterations by unique sources: {num_cpu}")
    ffid_split = np.array_split(unique_ffid, num_cpu)
    ffid_start = [ffid[0].tolist() for ffid in ffid_split]
    ffid_stop = [ffid[-1].tolist() for ffid in ffid_split]
    return ffid_start, ffid_stop, num_cpu

def get_indexes4read(header_file_path: Path,
                     data_file_path: Path,
                     ranges: tuple,
                     type_data: str,
                     sort_3d_order: str,
                     bin_size: tuple[float, float],
                     header_field_ffid: int = 9,
                     header_field_cdp: list[int] = list[181, 185],
                     endian: str = 'big',
                     num_sources_on_cpu: int = 1) -> list:
    """
    Determines the indices of traces within a seismic dataset to be read,
    based on specified FFID ranges and parallel processing parameters.

    This function extracts unique FFIDs (Field Record IDs) from a header file,
    selects those within a specified range, and calculates parameters for
    distributing the read operation across multiple CPU cores.

    Args:
        header_file_path (Optional[Path]): Path to the header file containing
                                         FFIDs. If None, defaults to
                                         `data_file_path`.
        data_file_path (Path): Path to the seismic data file. Used as the
                               header file path if `header_file_path` is None.
        ranges (Tuple[float, float, float]): A tuple specifying the start, stop,
                                             and increment values for selecting
                                             FFIDs within a range (start, stop,
                                             increment). Using float types to accommodate
                                             floating-point FFIDs.
        type_data (str): The data type of the seismic data.
        sort_3d_order (str): Method of sorting data in 3d seismic_data.
        header_field_ffid (int): The header field number containing the FFID.
                            Defaults to 9.
        header_field_cdp (List[int]): The header field number containing the CDP_X and CDP_Y.
                            Defaults to [181, 185].
        bin_size: (tuple[int, int]): The size of bins for CDP seismogram.
        endian (str): Endianness of the seismic data ("big" or "little").
                     Defaults to 'big'.
        num_sources_on_cpu (int): The desired number of sources to process per
                                  CPU core. Defaults to 1.

    Returns:
        List[int]: A list of start FFIDs, a list of end FFIDs and the number of cpus for the given task. It returns only the start FFIDs so the functions will work as before

    Raises:
        ValueError: If the specified FFID range and increment result in no
                    selected FFIDs.

    Notes:
        The function uses `get_ffid_header` to read the header data and
        `get_parameters4parallel` to calculate parallel processing parameters.
    """
    if header_file_path is None:
        header_file_path = data_file_path

    start, stop, increment = ranges
    if type_data == "2d" or sort_3d_order == "csp":
        header = get_ffid_header(header_file_path, header_field_ffid, endian)
    else:
        header = convert_cdp2ffid(header_file_path, header_field_cdp, bin_size, endian)


    unique_ffid, indexes, counts = np.unique(header, return_index=True, return_counts=True)
    if increment > 0:
        selected_ind_uniq = np.where((unique_ffid >= start) & (unique_ffid <= stop))[0][::increment]
        if not len(selected_ind_uniq):
            raise ValueError("Check the ffid ranges and increment or file with ffid numbers, maybe you should change it.")
        unique_ffid, indexes, counts = unique_ffid[selected_ind_uniq], indexes[selected_ind_uniq], counts[selected_ind_uniq]
    ffid_start, ffid_stop, num_cpu = get_parameters4parallel(unique_ffid, num_sources_on_cpu)

    all_selected_indexes = []
    for core_ind in range(num_cpu):
        selected_ind_core_uniq = np.where((unique_ffid >= ffid_start[core_ind]) & (unique_ffid <= ffid_stop[core_ind]))[0]
        unic_ffid_tmp = unique_ffid[selected_ind_core_uniq]
        all_selected_indexes_tmp = [np.where(header == unic_ffid)[0] for unic_ffid in unic_ffid_tmp]
        all_selected_indexes.append(np.hstack(all_selected_indexes_tmp))
    return all_selected_indexes


def read_segy(

    file_path: Path,
    name_headers: tuple,
    indexes4read: list,
    endian="big",
    sort_3d_order: str = "",
    type_data = "",
    bin_size: tuple = (),
    need_sort: bool = True,
) -> tuple[np.ndarray, np.ndarray, float] :
    """
    Reads SEGY data, extracts traces and headers, and sorts them according
    to specified criteria.

    This function opens a SEGY file, extracts specified traces and headers
    based on provided indices, and then sorts the data according to a
    specified order (for 2D or 3D data).

    Args:
        file_path (Path): Path to the SEGY file.
        name_headers (Tuple[str, ...]): Tuple of header names to extract.
        indexes4read (List[int]): List of trace indices to read from the SEGY file.
        endian (str): Endianness of the SEGY file ("big" or "little"). Defaults to "big".
        sort_3d_order (str): Sorting order for 3D data. Can be "csp"
                             (common source point) or "" (common midpoint).
                             Defaults to "".
        type_data (str): Type of seismic data. Can be "2d" or "". Defaults to "".
        bin_size (Tuple[float, float]): Bin size for CDP binning (for 3D data when
                                     sort_3d_order is not "csp"). If used, the bin sizes
                                     must be of float type. Defaults to ().
        need_sort (bool): Whether to sort the data after reading. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, int]: A tuple containing:
            - traces (np.ndarray): A 2D NumPy array of seismic traces (time x trace).
            - headers (np.ndarray): A 2D NumPy array of header values (header x trace).
            - dt (float): The sample interval (in seconds).
            - count_unique_seismic (int): The number of unique seismic sources or CDPs.
    """
    with segyio.open(file_path.as_posix(), endian=endian, ignore_geometry=True) as segy_file:
        segy_file.mmap()

        ### Read traces and headers for given indices
        traces = np.array([segy_file.trace[idx] for idx in indexes4read])
        headers = np.array([[segy_file.attributes(key)[idx] for idx in indexes4read] for key in name_headers]).squeeze()
        dt = segyio.dt(segy_file) * MULTIPLIER_FOR_DT

        if type_data == "2d":
            ### Case 2D data, the direction along which the profile 2D is directed is determined
            ### Data are sorting like [SourceX, GroupX] or [SourceY, GroupY]

            header_sou_ind_cur, header_rec_ind_cur = define_direction(headers, HEADER_SOU_X_IND, HEADER_SOU_Y_IND)

            if not need_sort:
                return traces.T, headers, dt
            sorted_indices = np.lexsort([headers[header_rec_ind_cur], headers[header_sou_ind_cur]])

        elif sort_3d_order == "csp":
            ### Case 3D data, Data are sorting like [SourceX, SourceY]
            headers[HEADER_SOU_X_IND:HEADER_REC_Y_IND+1] = headers[HEADER_SOU_X_IND:HEADER_REC_Y_IND+1] / 100
            sorted_indices = np.lexsort([headers[HEADER_OFFSET_IND], headers[HEADER_SOU_Y_IND], headers[HEADER_SOU_X_IND]])

        else:
            ## Case 3D data, Data are sorting like [CDP_X, CDP_Y]
            ### Binning CDP_X and writing to headers
            headers[HEADER_SOU_X_IND:HEADER_REC_Y_IND + 1] = headers[HEADER_SOU_X_IND:HEADER_REC_Y_IND + 1] / 100
            headers[HEADER_CDP_X_IND:HEADER_CDP_Y_IND + 1] = headers[HEADER_CDP_X_IND:HEADER_CDP_Y_IND + 1] / 100
            headers[HEADER_CDP_X_IND] = np.int32(headers[HEADER_CDP_X_IND] / bin_size[0]) * bin_size[0] + bin_size[0]//2
            headers[HEADER_CDP_Y_IND] = np.int32(headers[HEADER_CDP_Y_IND] / bin_size[1]) * bin_size[1] + bin_size[1]//2
            headers[HEADER_OFFSET_IND] = np.sqrt((headers[HEADER_REC_X_IND] - headers[HEADER_CDP_X_IND])**2 +
                                                 (headers[HEADER_REC_Y_IND] - headers[HEADER_CDP_Y_IND])**2)


            sorted_indices = np.lexsort(
                [headers[HEADER_OFFSET_IND], headers[HEADER_CDP_Y_IND], headers[HEADER_CDP_X_IND]]
            )

        return traces[sorted_indices].T, headers[:, sorted_indices], dt


def read_vp_model_segy(
    file_path: Path,
    name_headers: tuple,
    endian="big",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Reads a velocity model from a SEGY file.

    This function reads seismic data from a SEGY file, extracts all traces
    and specified headers, and returns them along with the sampling interval.
    It's designed for reading velocity models stored in SEGY format.

    Args:
        file_path (Path): Path to the SEGY file containing the velocity model.
        name_headers (Tuple[str, ...]): Tuple of header names to extract from
                                       the SEGY file.
        endian (str): Endianness of the SEGY file ("big" or "little").
                     Defaults to "big".

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: A tuple containing:
            - traces (np.ndarray): A 2D NumPy array of seismic traces
                                   representing the velocity model (time x trace).
            - headers (np.ndarray): A 2D NumPy array of header values (header x trace).
            - dt (float): The sample interval (in seconds).
    """


    with segyio.open(file_path.as_posix(), endian=endian, ignore_geometry=True) as segy_file:
        segy_file.mmap()

        ### Read traces and headers for given indices
        traces = segy_file.trace.raw[:]
        headers = np.array([segy_file.attributes(key)[:]  for key in name_headers]).squeeze()
        dt = segyio.dt(segy_file) * 1e-6
        return traces.T, headers, dt

def get_ranges_from_file(file_path: Path) -> list["Ranges"]:  # Assuming Ranges is defined elsewhere
    """
    Reads ranges (velocity shear and thicknesses) from a file.

    Args:
        file_path (Path): Path to the file containing the ranges data (min_vs, max_vs, min_thk, max_thk). The file is expected to be delimited by ';'.

    Returns:
        list[Ranges]: A list containing Ranges object(s) read from the file.
    """
    tmp: np.ndarray = np.genfromtxt(file_path, delimiter=";")  # Read data from file
    return [Ranges(velocity_shear_range=tmp[1:, :2], thicknesses_range=tmp[1:-1, 2:4])] # Create and return a Ranges object

def load_ranges_from_file(path: Path, disp_curves: list[DispersionCurve]) -> list[Ranges]:
    """
    Loads velocity and thickness ranges from a file and creates a Ranges object for each dispersion curve.

    This function reads velocity shear and thicknesses ranges from a file, and then creates a list of Ranges
    objects. Each Ranges object contains the *same* velocity shear and thicknesses ranges, and is created
    for each item in the disp_curves list.  It's important to understand that this function does *not*
    associate different ranges with different dispersion curves; it simply replicates the same ranges for
    each curve.

    Args:
        path (Path): Path to the file containing the velocity shear and thicknesses ranges data.  The
                     file is expected to contain two arrays: one for velocity shear ranges and one for
                     thicknesses ranges. The format should be compatible with `get_ranges_from_file`.
        disp_curves (List[np.ndarray]): A list of dispersion curve arrays. The length of this list
                                        determines how many Ranges objects will be created. The content
                                        of the dispersion curves is not used in this function.

    Returns:
        List[Ranges]: A list of Ranges objects. Each object contains the velocity shear and thicknesses
                      ranges loaded from the file. The list has the same length as the `disp_curves` list.
    """
    velocity_shear_range: np.ndarray
    thicknesses_range: np.ndarray
    velocity_shear_range, thicknesses_range = get_ranges_from_file(path)[0]  # Load ranges from file

    # Create a Ranges object for each dispersion curve, using the same data
    ranges_list: list[Ranges] = [
        Ranges(velocity_shear_range=velocity_shear_range, thicknesses_range=thicknesses_range)
        for _ in disp_curves
    ]

    return ranges_list

def get_vp_model_from_file(file_path: Path) -> list["PWaveVelocityModel"]:  # Assuming VpModel is defined elsewhere
    """
    Reads a Vp model from a file.

    Args:
        file_path (Path): Path to the file containing the Vp model data (depth, Vp, vp2vs). The file is expected to be delimited by ';'.

    Returns:
        list[VpModel]: A list containing VpModel object(s) read from the file.
    """
    tmp: np.ndarray = np.genfromtxt(file_path, delimiter=";")  # Read data from file
    return [PWaveVelocityModel(depth=tmp[1:, 0], vp=tmp[1:, 1], vp2vs=tmp[1:, 2])] # Create and return a VpModel object
