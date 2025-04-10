from typing import Dict, Callable, Any
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from segyio import TraceField
from tqdm import tqdm

from src.files_processor.savers import save_max_deviation_hist
from src.config_reader.models import DispersionCurve, InversionModel, ModelVCR, Ranges, PWaveVelocityModel
from src.inversion.getranges import define_model_ranges
from src.files_processor.readers import get_filenames, read_vp_model_segy, get_vp_model_from_file
from src.files_processor.savers import save_dc_rest_image
from src.inversion.optimizers import ssa, gwo, occam
from src.logs.utils import create_log
from src.inversion.utils import get_misfit, cluster_dispersion_curves


def select_optimizer(method: str) -> Callable:
    """
    Selects and returns an optimizer class based on the inversion method.

    This function acts as a factory for optimizer classes.  It takes a string argument
    specifying the desired inversion method and returns the corresponding optimizer class.

    Args:
        method (str): The name of the inversion method (e.g., "ssa", "gwo", "occam").

    Returns:
        Callable: The optimizer class (e.g., ssa.SSA, gwo.GWO, occam.Occam).  This is a
                  class, not an instance of the class.

    Raises:
        ValueError: If the provided method is not supported.
    """
    if method == "ssa":
        return ssa.SSA
    if method == "gwo":
        return gwo.GWO
    if method == "occam":
        return occam.Occam
    raise ValueError(f"Unsupported inversion method: {method}")


def inversion_processor(module_dirs: dict[str, list[Path]], inv_model: InversionModel) -> None:  #Added ""
    """
    Processes dispersion curves and performs velocity model inversion.

    This function orchestrates the inversion process, which involves:
        1. Cleaning the inversion output directories.
        2. Preparing dispersion curves and Vp models.
        3. Defining model ranges based on the chosen method.
        4. Running the inversion in parallel using the specified optimizer.
        5. Saving the inversion results (velocity models and QC images).
        6. Generating a log file summarizing the inversion process.

    Args:
        module_dirs (Dict[str, List[Path]]): A dictionary containing the paths to the spectral analysis and
                                            inversion directories. The expected keys are "spectral_analysis"
                                            and "inversion", with values being lists of Path objects.
                                            Specifically, module_dirs["spectral_analysis"] should contain a list
                                            of directories containing dispersion curve files, and
                                            module_dirs["inversion"] should contain pairs of directories for
                                            saving the inverted models (binary files) and QC images. The
                                            "spectral_analysis" dirs are expect to be 3 times longer than the "inversion"
        inv_model (InversionModel): An InversionModel object containing the inversion parameters,
                                    such as the inversion method, number of iterations, and global search settings.

    """

    # Remove all files in directories
    [[item.unlink() for item in dir_.glob("*")] for dir_ in module_dirs["inversion"]]  # Clean output directories

    log_inversion: dict[str, Any] = {"Inverted dispersion curves": 0, "Restored Vs-models stored in": "",
                                      "Images Vs-models stored in": ""}  # Initialize inversion log

    # Determine the filename for the QC image based on the inversion method
    if inv_model.inversion_method != 'occam':
        hits_name: str = f"MAPE_{inv_model.inversion_method.value}_{inv_model.niter}_{inv_model.global_search.xi}.png"  # Filename
        if not inv_model.global_search.path4vs_limits:
            log_inversion["Auto-defined ranges"] = ""  # Assign value to log
        else:
            log_inversion["Ranges from path4limits"] = ""  # Assign value to log
    else:
        hits_name = f"MAPE occam_{inv_model.niter}.png"  # Assign value to log

    # Select the appropriate optimizer based on the inversion method
    optimizer: Callable = select_optimizer(inv_model.inversion_method)

    for idx, spec_dc_dir in enumerate(module_dirs["spectral_analysis"][::3]):  # Iterate over spectral analysis directories, skip every 3 as we have 3 times more spectral dirs

        path4save_bins_dir: Path
        path4save_image_dir: Path
        path4save_bins_dir, path4save_image_dir = module_dirs["inversion"][idx * 2: (idx + 1) * 2]  # Get output directories

        # Prepare dispersion curves
        dispersion_curves: list[DispersionCurve] = [DispersionCurve.load(path_dc) for path_dc in
                                                       get_filenames(data_dir=spec_dc_dir, suffix=".npz")]  # Load from files # added "" to be future reference to type

        # Prepare Vp models
        vp_model: list[Any] = prepare_vp_models(inv_model, dispersion_curves) #Changed List[VpModel] to Any

        # Calculate the maximum depth of the velocity model, estimated from all dispersion curves
        max_depth: float = float(np.median(np.array([curve.velocity_phase[0][0] / curve.frequency[0][0] / 3 for curve in
                                                dispersion_curves])))

        ranges, classes = np.arange(len(dispersion_curves)), np.arange(len(dispersion_curves)) #Define some default values for no if statement

        n_lay: list[int] = []
        if inv_model.inversion_method != 'occam':
            # Calculate model ranges
            ranges = define_model_ranges(inv_model.global_search, dispersion_curves)
            for i in ranges:
                n_lay.append(len(i.velocity_shear_range)) # Append num layers to n_lay

            if (len(ranges) == 1) and (len(dispersion_curves) != 1): # If one range and multiple curves
                ranges *= len(dispersion_curves) # Multiply list

            elif (len(ranges) > 1) and (len(dispersion_curves) != len(ranges)): # Added condition to check number of the dispersion curve and ranges to prevent errors
                _, classes = cluster_dispersion_curves(dispersion_curves)  # Cluster dispersion curves

        print("Start Inversion in parallel option")
        # Run inversion in parallel
        mape: list[float] = Parallel(n_jobs=-1, backend='loky')(
            delayed(curve_inversion)(optimizer, inv_model, dispersion_curves[i], vp_model[i], ranges[classes[i]],
                                     max_depth,
                                     path4save_bins_dir, path4save_image_dir, inv_model.qc_inversion)
            for i in tqdm(range(len(dispersion_curves)))
        ) # added List

        hits_title: str = f"MAPE for {inv_model.inversion_method.value}"  # String Title

        save_max_deviation_hist(mape, module_dirs["inversion"][0].parent / hits_name,
                       hits_title)  # Save Hits images to the data

        if inv_model.inversion_method != 'occam':
            if not inv_model.global_search.path4vs_limits:
                log_inversion["Auto-defined ranges"] += f"\n{spec_dc_dir} {np.unique(n_lay, return_counts=False)}"
            else:
                log_inversion["Ranges from path4limits"] += f"\n{spec_dc_dir} {np.unique(n_lay, return_counts=False)}"

        log_inversion["Inverted dispersion curves"] += len(dispersion_curves)  # Update log
        log_inversion["Restored Vs-models stored in"] += "\n " + str(path4save_bins_dir)  # Update log
        log_inversion["Images Vs-models stored in"] += "\n " + str(path4save_image_dir)  # Update log

    # More readable
    log_inversion["Restored Vs-models stored in"] += "\n "
    log_inversion["Images Vs-models stored in"] += "\n "
    log_inversion["Inverted dispersion curves"] = str(log_inversion["Inverted dispersion curves"]) + "\n"

    create_log(log_inversion, module_dirs["spectral_analysis"][0::2][0].parents[4], "inversion")  # Create log file


def curve_inversion(
        optimizer,
        inv_model: InversionModel,
        dispersion_curves: DispersionCurve,
        vp_model: PWaveVelocityModel,
        ranges: [list, Ranges],
        max_depth: float,
        path4save_bins_dir: Path,
        path4save_image_dir: Path,
        qc_inversion: bool,
) -> float:
    vs_tmp, thk_tmp, dc_rest = optimizer(inv_model, dispersion_curves, ranges, vp_model, max_depth).run()

    error = float(np.mean([get_misfit("mape", obs, rest) for obs, rest in zip(dispersion_curves.velocity_phase, dc_rest)]))
    save_model(vs_tmp, thk_tmp, dc_rest, dispersion_curves, max_depth, ranges, inv_model.inversion_method, path4save_bins_dir, path4save_image_dir, qc_inversion, error)

    return error


def save_model(
        velocity_shear: list[float],
        thickness: list[float],
        dc_rest: list[float],
        disp_curve: "DispersionCurve",
        max_depth: float,
        ranges: [list, Ranges],
        method: str,
        path4save_bins_dir: Path,
        path4save_image_dir: Path,
        qc_inversion: bool,
        mape: float,
) -> None:
    """
    Saves the inverted velocity model and, optionally, a QC image of the dispersion curve fit.

    This function saves the inverted velocity model (shear wave velocity and layer thicknesses) to a binary
    file using the `ModelVCR` class. If quality control (QC) is enabled, it also saves an image showing the
    fit between the observed dispersion curve and the computed dispersion curve, along with the velocity model.

    Args:
        velocity_shear (List[float]): List of shear wave velocities for each layer in the inverted model (m/s).
        thickness (List[float]): List of layer thicknesses in the inverted model (m).
        dc_rest (List[float]): List of values representing the rest of the computed dispersion curve data.
        disp_curve (DispersionCurve): The DispersionCurve object containing information about the observed
                                       dispersion curve, such as CMP location, relief, spectral name, and
                                       the observed velocity and frequency values.
        max_depth (float): The maximum depth of the velocity model for plotting purposes (m).
        path4save_bins_dir (Path): The directory to save the binary file containing the inverted velocity model.
        path4save_image_dir (Path): The directory to save the QC image, if `qc_inversion` is True.
        qc_inversion (bool): A flag indicating whether to save the QC image. If True, the image is saved; otherwise, it is skipped.
        mape (float): The Mean Absolute Percentage Error (MAPE) between the observed and computed dispersion curves.
                      This value is saved as part of the ModelVCR data.
    """
    ModelVCR(
        thickness=thickness,
        velocity_shear=velocity_shear,
        cmp_x=disp_curve.cmp_x,
        cmp_y=disp_curve.cmp_y,
        relief=disp_curve.relief,
        error_dc=mape,

    ).save(path4save_bins_dir / disp_curve.spec_name)  # Save the inverted model to a binary file

    if qc_inversion:  # Save QC image if qc_inversion is True
        save_dc_rest_image(
            path4save_image_dir / f"{Path(disp_curve.spec_name).stem}.png", # Construct the path for the QC image
            disp_curve.velocity_phase,  # Observed phase velocity
            dc_rest,  # Computed dispersion curve
            disp_curve.frequency,  # Frequencies
            velocity_shear,  # Shear wave velocity
            thickness,  # Layer thicknesses
            max_depth,  # Maximum depth for plotting
            ranges,
            method,
        )

def prepare_vp_models(inv_model: InversionModel, disp_curves: list[DispersionCurve]) -> list[PWaveVelocityModel]:
    """
    Prepares P-wave velocity (Vp) models for each dispersion curve.

    This function loads Vp models and related parameters (Vp/Vs ratio and depths) from a file and associates
    them with each dispersion curve. The method of loading depends on the file type specified in the
    `inv_model`. It handles both CSV files (representing 1D models) and SEGY/SGY files (potentially representing 3D models).

    Args:
        inv_model (InversionModel): An InversionModel object containing the inversion parameters, including
                                    the path to the Vp model file (`path4vp_model`) and the specification of the
                                    Vp model itself (`vp_model`).
        disp_curves (List[DispersionCurve]): A list of DispersionCurve objects. Each curve will be associated with
                                            a Vp model.

    Returns:
        List[PWaveVelocityModel]: A list of PWaveVelocityModel objects, one for each dispersion curve. The
                                  Vp model parameters are loaded from the specified file and associated with
                                  the corresponding dispersion curve based on location (for SEGY/SGY files).

    Raises:
        ValueError: If the Vp/Vs values in the SEGY file are too large (greater than 100) when `inv_model.vp_model` is "vp2vs".
    """

    path: Path = inv_model.path4vp_model  # Get the path to the Vp model file

    if path.suffix == ".csv":  # Load 1D Vp model from CSV file
        depth: np.ndarray
        vp: np.ndarray
        vp2vs: np.ndarray
        depth, vp, vp2vs = get_vp_model_from_file(path)[0]  # Load depth, Vp, and Vp/Vs from file
        # Create a PWaveVelocityModel object for each dispersion curve, using the same 1D model
        return [PWaveVelocityModel(depth=depth[1], vp=vp[1], vp2vs=vp2vs[1]) for _ in disp_curves] # Removed indices [1] as depth, vp and vp2vs are already arrays

    elif path.suffix == ".segy" or path.suffix == ".sgy":  # Load Vp model from SEGY/SGY file
        traces: np.ndarray
        headers: np.ndarray
        dt: float
        traces, headers, dt = read_vp_model_segy(path, (TraceField.CDP_X, TraceField.CDP_Y))  # Read traces, headers, and dt

        if inv_model.vp_model == "vp2vs" and np.mean(traces) > 100:  # Check for excessively large Vp/Vs values
            raise ValueError("Use segy-file of Vp/Vs. Vp/Vs has too large values (greater than 100)")

        depth: np.ndarray = np.arange(0, traces.shape[0] * dt * 1000, dt * 1000)  # Create depth array
        vp_all: list[PWaveVelocityModel] = []
        for curve in disp_curves:  # Iterate over dispersion curves
            index_min: int = int(np.argmin((curve.cmp_x - headers[0, :]) ** 2 + (curve.cmp_y - headers[1, :]) ** 2))  # Find the closest trace
            vp_all.append(PWaveVelocityModel(depth=depth, vp=traces[:, index_min], vp2vs=traces[:, index_min]))  # Assign Vp model to each curve
        return vp_all #Return Vp Models

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}") # added expection


