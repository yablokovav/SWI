import numpy as np
from typing import Set
import os
import segyio
from pathlib import Path
from src.config_reader.load_params import ConfigReader
from src.config_reader.models import SWIConfigModel


def get_params(params_dir: Path, swi_dir: Path):
    preprocessing, spectral, inversion, postprocessing = ConfigReader.read(params_dir,
                                                                           SWIConfigModel,
                                                                           swi_dir,
                                                                           show=False)
    return preprocessing, spectral, inversion, postprocessing


def are_npz_files_equal(file1_path: str, file2_path: str) -> bool:
    """
    Compares two npz files with data of dispersion curves.

    Args:
        file1_path (str): Path to the first npz file.
        file2_path (str): Path to the second npz file.

    Returns:
        bool: True if the npz files are identical, False otherwise.
    """
    try:
        data1 = np.load(file1_path, allow_pickle=True)
        data2 = np.load(file2_path, allow_pickle=True)

        if set(data1.keys()) != set(data2.keys()):
            print("No match keys")
            return False

        for key in data1.keys():
            value1 = data1[key]
            value2 = data2[key]

            if (key == 'frequency') or (key == 'velocity_phase'):
                try:
                    value1 = list(value1.item().values())
                    value2 = list(value2.item().values())
                except:
                    print("Information about dispersion curves has incorrect format")
                    return False
                if len(value1) != len(value2):
                    print("Incorrect number founded modes")
                    return False
                for mode1, mode2 in zip(value1, value2):
                    if len(mode1) != len(mode2):
                        print("Incorrect counts of frequencies or velocities in founded modes")
                        return False
                    if not np.allclose(mode1, mode2):
                        print("Incorrect values of frequencies or velocities in curve")
                        return False
            else:
                if value1 != value2:
                    print("Incorrect values of attributes")
                    return False
        return True

    except FileNotFoundError:
        print("File not found")
        return False
    except Exception as e:
        print("Data in npz file has incorrect format, unable to check")
        return False


def compare_npz_folders(folder1_path: str, folder2_path: str) -> bool:
    """
    Compares two directories for identical .npz files (down to the content).

    Args:
        folder1_path (str): Path to the first directory.
        folder2_path (str): Path to the second directory.

    Returns:
        bool: True if the directories contain the same .npz files with identical
              content, False otherwise.
    """

    try:
        # Gets list sof npz files in directories
        files1: Set[str] = {f for f in os.listdir(folder1_path) if f.endswith(".npz")}
        files2: Set[str] = {f for f in os.listdir(folder2_path) if f.endswith(".npz")}

        # Compare lists oof files
        if files1 != files2:
            print(files1, flush=True)
            print(files2, flush=True)
            print("Mismatch between names of npz files>")
            return False

        # Compare files with each other
        for filename in files1:
            file1_path: str = os.path.join(folder1_path, filename)
            file2_path: str = os.path.join(folder2_path, filename)

            if not are_npz_files_equal(file1_path, file2_path):
                print(f"Nor match npz files {Path(file1_path).stem} {Path(file2_path).stem}")
                return False

        return True  # All files are identical

    except FileNotFoundError:
        print("File not found")
        return False
    except Exception as e:
        print(e)
        return False

def are_segy_files_equal(file1_path: str, file2_path: str) -> bool:
    """
    Compares two SEGY files for identical trace content and trace headers.

    Args:
        file1_path (str): Path to the first SEGY file.
        file2_path (str): Path to the second SEGY file.

    Returns:
        bool: True if the SEGY files are identical in terms of trace content and headers,
              False otherwise.
    """
    try:
        with segyio.open(file1_path, 'r', ignore_geometry=True) as segyfile1, \
             segyio.open(file2_path, 'r', ignore_geometry=True) as segyfile2:

            # Compare bin headers
            if segyfile1.bin != segyfile2.bin:
                print("Incorrect bin header")
                return False

            # Compare counts of traces
            if segyfile1.tracecount != segyfile2.tracecount:
                print("Incorrect trace count")
                return False

            num_traces: int = segyfile1.tracecount

            # Compare traces and headers between each other
            for i in range(num_traces):
                # Сравниваем заголовки трасс
                if segyfile1.header[i] != segyfile2.header[i]:
                    print("Incorrect headers")
                    return False

                # compare trace data
                trace1: np.ndarray = segyfile1.trace[i]
                trace2: np.ndarray = segyfile2.trace[i]

                if not isinstance(trace1, np.ndarray) or not isinstance(trace2, np.ndarray):
                    print("Incorrect trace type")
                    return False

                if not trace1.shape == trace2.shape:
                    print("Incorrect shape of traces")
                    return False

                if not np.allclose(trace1, trace2):
                    print("Incorrect traces values")
                    return False

            return True  # All traces and headers equally

    except FileNotFoundError:
        return False
    except Exception as e:
        print("Data in sgy file has incorrect format, unable to check")
        return False

def compare_segy_folders(folder1_path: str, folder2_path: str) -> bool:
    """
    Compares two directories for identical SEGY files (based on trace content and headers).

    Args:
        folder1_path (str): Path to the first directory.
        folder2_path (str): Path to the second directory.

    Returns:
        bool: True if the directories contain the same SEGY files with identical
              trace content and headers, False otherwise.
    """

    try:
        # gets list of segy files in directories
        files1: Set[str] = {f for f in os.listdir(folder1_path) if f.endswith(".segy") or f.endswith(".sgy")}
        files2: Set[str] = {f for f in os.listdir(folder2_path) if f.endswith(".segy") or f.endswith(".sgy")}

        # compare lists of files
        if files1 != files2:
            print("Incorrect counts of files or incorrect files's names")
            return False

        # compere file s with each other
        for filename in files1:
            file1_path: str = os.path.join(folder1_path, filename)
            file2_path: str = os.path.join(folder2_path, filename)

            if not are_segy_files_equal(file1_path, file2_path):
                print(f"Nor match npz files {Path(file1_path).stem} {Path(file2_path).stem}")
                return False

        return True  #  All files are identical

    except FileNotFoundError:
        return False
    except Exception as e:
        return False