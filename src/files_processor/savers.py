import struct
from pathlib import Path
import segyio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import io
from PIL import Image
from src.spectral_analysis.models import Spectra
from scipy.interpolate import interp1d
plt.switch_backend('Agg')

HEADER_BYTE_FOR_FREQ = 0
HEADER_BYTE_CDP_X = 71
HEADER_BYTE_CDP_Y = 72
MAX_POSSIBLE_NUM_MODS = 70
MULTIPLIER_FOR_FREQ_IN_SEGY = 1000  # Multiplier for correct saving values of frequencies in SEGY-file
MULTIPLIER_FOR_DT_IN_SEGY = 1e6  # Multiplier for correct saving dt in SEGY-file
DIVIDER_FOR_VEL_IN_SEGY = 1000  # Divider for correct saving velocity interval in SEGY-file

def save_dc_rest_image(img_name, dc_obs, dc_rest, freq, vs, thk, vs_init, maxdepth, ranges, method):
    """
   Saves a diagnostic image comparing observed and restored dispersion curves
   and the restored velocity model.

   This function generates a two-panel plot: the first panel shows the
   observed and restored dispersion curves (phase velocity vs. frequency),
   and the second panel displays the restored shear-wave velocity (Vs) model
   as a function of depth. The plot is then saved to a PNG image file.

   Args:
       img_name (Path): Path to save the output image.
       dc_obs (List[np.ndarray]): List of observed dispersion curves
                                   (each a NumPy array of phase velocities).
       dc_rest (List[np.ndarray]): List of restored dispersion curves
                                    (each a NumPy array of phase velocities).
       freq (List[np.ndarray]): List of frequency arrays corresponding to the
                                 dispersion curves.
       vs (list[float]): List of shear-wave velocities (Vs) for the
                        restored model.
       thk (list[float]): List of layer thicknesses for the restored
                         model.
       maxdepth (float): Maximum depth to display in the velocity model plot.
   """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Plot the dispersion curves
    for mode_i in range(len(dc_rest)):
        dc_rest[mode_i][dc_rest[mode_i]==0] = np.nan
        ax[0].plot(freq[mode_i], dc_obs[mode_i], color='darkgreen', label='Extracted curve')
        ax[0].plot(freq[mode_i], dc_rest[mode_i], color='darkred', label='Restored curve')
    ax[0].set_xlabel("Frequency, Hz")
    ax[0].set_ylabel("Phase velocity, m/s")

    # Create a unique legend.  This handles duplicate labels in the loop.
    handles, labels = ax[0].get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax[0].legend(handles, labels)

    # Plot the velocity model
    depth = np.r_[0, np.cumsum(thk), maxdepth]
    vs = np.r_[vs, vs[-1]]
    ax[1].step(vs, depth, color='darkred', label='Restored model')
    ax[1].set_ylim(maxdepth, 0)
    ax[1].set_xlabel("Shear-wave velocity, m/s")
    ax[1].set_ylabel("Depth, m")
    if method != 'occam':
        depth_min = np.r_[0, np.cumsum(ranges.thicknesses_range[:, 1]), maxdepth]
        depth_max = np.r_[0, np.cumsum(ranges.thicknesses_range[:, 0]), maxdepth]
        vs_min = np.r_[ranges.velocity_shear_range[:, 0], ranges.velocity_shear_range[-1, 0]]
        vs_max = np.r_[ranges.velocity_shear_range[:, 1], ranges.velocity_shear_range[-1, 1]]
        ax[1].step(vs_min, depth_min, color='gray', label = 'Model ranges')
        ax[1].step(vs_max, depth_max, color='gray')
    else:
        vs_init = np.r_[vs_init, vs_init[-1]]
        ax[1].step(vs_init, depth, color='gray', label='Start model')
    ax[1].legend(loc='upper center')

    fig.tight_layout()

    # Save the plot to a PNG image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)

    # Load the image from the buffer and save to file
    image = Image.open(buf)
    image.save(img_name, format="PNG")
    buf.close()
    plt.close()

def save_spec_image(img_name: Path,
                    spectra: Spectra,
                    frequencies: list[np.ndarray],
                    freq_limits: np.ndarray,
                    lover_v: np.ndarray,
                    upper_v: np.ndarray,
                    dcs: list[np.ndarray],
                    flags: list):
    """
    Saves a spectrogram image with overlaid dispersion curves.

    This function generates a spectrogram image, overlays extracted and
    rejected dispersion curves, and saves the resulting image to a PNG file.

    Args:
        img_name (Path): Path to save the output image.
        spectra (Spectra):  An object containing the spectrogram data
                          (vf_spectra), velocities, and frequencies. Incomplete without more
                          information about this data.
        frequencies (List[np.ndarray]): List of frequency arrays for each
                                     dispersion curve.
        freq_limits (np.ndarray): Array defining the frequency limits for the
                                 dashed lines.
        lover_v (np.ndarray): Array defining the lower velocity limits for the
                              dashed lines.
        upper_v (np.ndarray): Array defining the upper velocity limits for the
                              dashed lines.
        dcs (List[np.ndarray]): List of dispersion curves (phase velocities)
                                 corresponding to the frequencies.
        flags (list): Boolean array indicating whether each dispersion
                            curve was extracted (True) or rejected (False).
    """
    plt.ioff()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # Plot the spectrogram (using imshow)
    ax.imshow(
        spectra.vf_spectra,
        extent=(
            spectra.frequencies.min(),
            spectra.frequencies.max(),
            spectra.velocities.max(),
            spectra.velocities.min(),
        ),
        aspect="auto",
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1
    )

    ax.set_ylim(spectra.velocities.min(), spectra.velocities.max())
    ax.plot(freq_limits, lover_v, "w--")
    ax.plot(freq_limits, upper_v, "w--")

    # Overlay dispersion curves
    for ind_mode, curve in enumerate(zip(frequencies, dcs)):
        frequency = curve[0]
        dc = curve[1]
        if flags[ind_mode]:
            ax.plot(frequency, dc, "darkgreen", label="Extracted curve")
        else:
            ax.plot(frequency, dc, "--k", label="Rejected curve")
        ax.annotate(f"{ind_mode}", xy=(float(frequency[0]), float(dc[0])), xycoords='data')

    # Create a unique legend. This handles possible duplicates.
    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels)


    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("Phase velocity, m/s")

    # Save the plot to a PNG image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)

    # Load the image from the buffer and save to file
    image = Image.open(buf)
    image.save(img_name, format="PNG")
    buf.close()
    plt.close()

def save_segy(
    output_segy: Path,
    traces: np.ndarray,
    attributes: np.ndarray,
    headers: tuple,
    dt: float,
    sorting: int = 1,
    format: int = 1,
) -> None:
    """
   Saves seismic data (traces and headers) to a SEGY file.

   This function creates a SEGY file and writes the provided traces and
   header information to it. It handles the creation of the output directory
   if it doesn't exist, and sets various SEGY file specifications.

   Args:
       output_segy (Path): Path to the output SEGY file.
       traces (np.ndarray): A 2D NumPy array of seismic traces (time x trace).
       attributes (np.ndarray): A 2D NumPy array of header values (header x trace).
       headers (Tuple[int, ...]): A tuple of SEGY header keys to write.
       dt (float): The sample interval (in seconds).
       sorting (int): The sorting type for traces (e.g., 1 for unknown, see
                     segyio.Sorting for options). Defaults to 1.
       format (int): The data format code for the traces (e.g., 1 for
                     32-bit floating-point, see segyio.TraceField.DataFormat
                     for options). Defaults to 1.

   """
    # проверка и создание директории
    output_segy.parent.mkdir(parents=True, exist_ok=True)

    nt, nx = traces.shape
    spec = segyio.spec()
    spec.samples = list(np.arange(nt))
    spec.tracecount = nx
    spec.sorting = sorting
    spec.format = format
    attributes = np.int32(attributes)
    traces = np.float32(traces)

    with segyio.create(output_segy.as_posix(), spec) as dst_file:
        dst_file.bin[segyio.BinField.Interval] = int(dt * MULTIPLIER_FOR_DT_IN_SEGY)
        dst_file.trace = np.ascontiguousarray(traces.T)
        for i in range(spec.tracecount):
            for j, attr in enumerate(headers):
                dst_file.header[i][attr] = attributes[j, i]


def save_spec_segy(
        segy_name: Path,
        spectra: 'Spectra',
        frequencies: list,
        dcs: list,
        cdp: tuple
) -> None:
    """
    Saves spectrogram and dispersion curve data to a SEGY file.

    This function creates a SEGY file containing the spectrogram data and
    interpolated dispersion curves. It extracts a limited set of modes,
    interpolates the dispersion curves to match the spectrogram frequency
    range, stacks the header information, and then uses the `save_segy`
    function to write the data to a SEGY file.

    Args:
        segy_name (Path): Path to save the output SEGY file.
        spectra (Spectra): An object containing spectrogram data, including
                           the spectrogram (vf_spectra), frequencies, and
                           d_vel (velocity spacing).  The Spectra object is a forward ref and should be updated if info is available.
        frequencies (List[np.ndarray]): List of frequency arrays for each
                                     dispersion curve.
        dcs (List[np.ndarray]): List of dispersion curves (phase velocities)
                                 corresponding to the frequencies.
        cdp (Tuple[float, float]): CDP (Common Depth Point) coordinates (X, Y)
                                 for the data. These are float types because coordinates often have decimal places
        """
    num_modes = min([len(dcs), MAX_POSSIBLE_NUM_MODS])
    nf = len(spectra.frequencies)

    header_byte = sorted(segyio.tracefield.keys.values())
    freq_header_byte = [header_byte[HEADER_BYTE_FOR_FREQ]]
    cdp_x_header_byte = [header_byte[HEADER_BYTE_CDP_X]]
    cdp_y_header_byte = [header_byte[HEADER_BYTE_CDP_Y]]

    dc_header_bytes = []
    headers_dc = np.zeros((num_modes, nf))
    for mode_i  in range(num_modes):
        dc_header_bytes.append(header_byte[1 + mode_i])
        # Create an interpolation function for the dispersion curve data.
        headers_dc[mode_i] = interp1d(
            frequencies[mode_i],
            dcs[mode_i],
            axis=0,
            kind='linear',
            fill_value=(0, 0),
            bounds_error=False,
        )(spectra.frequencies)

    # Stack the header information into a single array.  Convert numeric values to int32.
    all_headers = np.vstack(
        (spectra.frequencies*MULTIPLIER_FOR_FREQ_IN_SEGY,
         headers_dc,
         np.repeat(cdp[0], repeats=nf),
         np.repeat(cdp[1], repeats=nf)
         )
    )

    # Save the dispersion curve data to a SEGY file using the save_segy function.
    save_segy(
        Path(segy_name),
        np.float32(spectra.vf_spectra),
        all_headers,
        tuple(freq_header_byte + dc_header_bytes + cdp_x_header_byte + cdp_y_header_byte),
        spectra.d_vel/DIVIDER_FOR_VEL_IN_SEGY
    )

def save_model_to_bin(
    filepath: Path,
    model: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    elevation: np.ndarray,
    projection: str,
) -> None:
    """
    Saves a velocity model and related data to a binary file using Numpy.savez function.

    This function packages a shear-wave velocity model (Vs) along with
    coordinates (x, y, z), elevation data, and projection information into a
    single `.npz` file using Numpy.savez function.

    Args:
        filepath (Path): Path to the output `.npz` file.
        model (np.ndarray): A 2D NumPy array representing the shear-wave
                            velocity model (Vs). It is transposed before saving.
        x (np.ndarray): A NumPy array of x-coordinates.
        y (np.ndarray): A NumPy array of y-coordinates.
        z (np.ndarray): A NumPy array of z-coordinates.
        elevation (np.ndarray): A NumPy array of elevation values.
        projection (str): A string describing the profile projection ('xz' or 'yz').
    """

    np.savez(
        filepath,
        vs=model.T,  # Transpose the velocity model before saving
        z=z,
        x=x,
        y=y,
        elevation=elevation,
        projection=projection,
        allow_pickle=True,  # Allows saving of object arrays (e.g. strings). Consider if needed to ensure safe saving of code values
    )

def save_max_deviation_hist(max_deviation: list[float], file_path_name: Path, hits_title: str) -> None:
    """
    Saves a histogram of maximum percentage deviation values to a file.

    This function generates a histogram of maximum percentage deviation values,
    formats the y-axis as percentages, sets the title and axis labels, and
    saves the plot to a specified file.

    Args:
        max_deviation (List[float]): A list of maximum deviation values
                                    (as decimals, e.g., 0.1 for 10%).
        file_path_name (Path): Path to save the histogram image.
        hits_title (str): Title for the histogram plot.
    """
    bins = np.linspace(0, 100, 21)
    plt.hist(np.array(max_deviation)*100,
             bins=bins,
             weights=np.ones(len(max_deviation)) / len(max_deviation),
             histtype='bar',
             rwidth=0.8)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title(hits_title)
    plt.xticks(bins)
    plt.ylabel("Percentage of the total number of curves")
    plt.xlabel("Max_deviation, %")
    plt.xticks(bins)
    plt.savefig(file_path_name)
    plt.close()


def write_fdm(filename: str, velocity_model: np.ndarray, x_size: int, y_size: int, z_size: int,
              x_step: float, y_step: float, z_step: float, cfst: float = 1.0, sfst: float = 1.0,
              cinc: float = 1.0, sinc: float = 1.0, dist_unit: float = 1.0, angle_unit: float = 2.0,
              north_angle: float = 0.0, rot_angle: float = 0.0, utm_x: float = 0, utm_y: float = 0) -> None:
    """Writes a 3D velocity model to an FDM format file according to seismic processing specifications.

    The FDM file format consists of a 512-byte header (128 float32 values) followed by a 3D array
    of float32 values ordered by crosslines (X), inlines (Y), and depths/time (Z), with depth
    varying most rapidly.

    Args:
        filename: Output file path for the FDM file
        velocity_model: 3D numpy array (z, y, x) containing velocity values in meters/second
        x_size: Number of bins along crossline (X) dimension
        y_size: Number of bins along inline (Y) dimension
        z_size: Number of bins along depth/time (Z) dimension
        x_step: Bin size in meters along X dimension (must be positive)
        y_step: Bin size in meters along Y dimension (must be positive)
        z_step: Bin size in meters along Z dimension (must be positive)
        cfst: First crossline number (typically starts at 1.0)
        sfst: First inline number (typically starts at 1.0)
        cinc: Crossline number increment (typically 1.0)
        sinc: Inline number increment (typically 1.0)
        dist_unit: Distance units indicator (1=meters, 2=feet, etc.)
        angle_unit: Angle units indicator (1=degrees, 2=radians, etc.)
        north_angle: Unused parameter (assume equal PI=3.141592...)
        rot_angle: Coordinate system rotation angle in radians
        utm_x: UTM X coordinate (easting) of bin (1,1) center
        utm_y: UTM Y coordinate (northing) of bin (1,1) center

    Raises:
        AssertionError: If velocity_model dimensions don't match specified x/y/z sizes

    Notes:
        - The coordinate system is assumed to be local (already rotated and translated)
        - All header fields are written as 32-bit floats unless otherwise noted
        - UTM coordinates are stored as integer + fractional parts at byte offset 96
        - The 3D array is written in Z,Y,X order with depth varying fastest
    """
    # Validate input dimensions
    assert velocity_model.shape == (x_size, y_size, z_size), (
        f"Velocity model shape {velocity_model.shape} doesn't match "
        f"specified dimensions (x={x_size}, y={y_size}, z={z_size})"
    )

    # Initialize header with 128 float32 values (512 bytes total)
    header = np.zeros(128, dtype=np.float32)

    # Populate mandatory header fields
    # Byte offsets are calculated as index * 4 (since float32 = 4 bytes)
    header[0] = 0.0  # Xor1: X coordinate of bin (1,1) (typically 0)
    header[1] = x_size  # Xsize: Number of crossline bins
    header[2] = x_step  # Xstep: Crossline bin size in meters
    header[3] = 0.0  # Yor1: Y coordinate of bin (1,1) (typically 0)

    header[4] = y_size  # Ysize: Number of inline bins
    header[5] = y_step  # Ystep: Inline bin size in meters
    header[6] = 0.0  # Zor1: Starting depth/time (typically 0)
    header[7] = z_size  # Zsize: Number of depth/time bins

    header[8] = z_step  # Zstep: Depth/time bin size in meters
    # Indices 9-11: Reserved (0.0)

    header[12] = cfst  # Cfst: First crossline number
    header[13] = sfst  # Sfst: First inline number

    header[16] = cinc  # Cinc: Crossline number increment
    header[17] = sinc  # Sinc: Inline number increment
    # Indices 18-19: Reserved (0.0)

    header[20] = dist_unit  # DistUnit: Distance units indicator
    header[21] = angle_unit  # AngleUnit: Angle units indicator
    header[22] = north_angle  # NorthAngle: Reserved for future use
    header[23] = rot_angle  # RotAngle: Coordinate system rotation in radians

    # Convert header to byte array for direct manipulation
    header_bytes = header.view(np.uint8)

    # Store UTM coordinates at byte offset 96 (24 floats * 4 bytes)
    # Format: [int32 X, float32 X_frac, int32 Y, float32 Y_frac]
    utm_x_int = int(utm_x)
    utm_x_frac = utm_x - utm_x_int
    utm_y_int = int(utm_y)
    utm_y_frac = utm_y - utm_y_int

    # Write UTM coordinates using struct packing
    pos = 96  # Byte position for UTM coordinates
    struct.pack_into('i', header_bytes, pos, utm_x_int)  # X integer
    struct.pack_into('f', header_bytes, pos + 4, utm_x_frac)  # X fraction
    struct.pack_into('i', header_bytes, pos + 8, utm_y_int)  # Y integer
    struct.pack_into('f', header_bytes, pos + 12, utm_y_frac)  # Y fraction

    # Write to file
    with open(filename, 'wb') as f:
        f.write(header_bytes)  # 512-byte header

        # Write 3D array in Z,Y,X order (depth-fastest)
        f.write(np.ascontiguousarray(velocity_model, dtype=np.float32).tobytes())