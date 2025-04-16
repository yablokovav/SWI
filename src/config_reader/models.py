from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, model_validator, ConfigDict
from typing_extensions import Self

from src.config_reader.enums import (
    ExtractDCMethod,
    GetNumLayers,
    InversionDcMethod,
    Sort3dOrder,
    SpectralMethod,
    TypeData,
    VelocityType,
    WaveType,
    VpModel,
    Interp_dim
)





class Parameters3DModel(BaseModel):
    sort_3d_order: Sort3dOrder
    num_sectors: int


class PreprocessingModel(BaseModel):
    ffid_start: int
    ffid_stop: int
    ffid_increment: int
    scaler_to_elevation: float
    scaler_to_coordinates: float
    num_sources_on_cpu: int
    path4ffid_file: Optional[Path]
    data_dir: Path
    offset_min: float
    offset_max: float
    type_data: TypeData
    snr: float
    qc_preprocessing: bool
    parameters_3d: Optional[Parameters3DModel]

    @model_validator(mode="after")
    def validate_data_dir(self) -> Self:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")
        return self


# ___spectral___############
class SpectralAdvancedModel(BaseModel):
    desired_nt: int
    desired_nx: int
    smooth_data: bool
    width: float
    peak_fraction: float
    cutoff_fraction: float
    dc_error_thr: float


class SpectralModel(BaseModel):   
    spectral_method: SpectralMethod        
    fmin: float
    fmax: float
    vmin: float
    vmax: float
    extract_dc_method: ExtractDCMethod
    path4dc_limits: Optional[Path]        
    advanced: SpectralAdvancedModel
    qc_spectral: bool


class GlobalSearchModel(BaseModel):
    test_count: int
    get_num_layers: GetNumLayers
    xi: float
    path4vs_limits: Optional[Path]


class LocalSearchModel(BaseModel):
    nlay: int

class InversionModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    inversion_method: InversionDcMethod
    niter: int
    max_num_modes: int
    wavetype: WaveType
    veltype: VelocityType
    global_search: GlobalSearchModel
    local_search: LocalSearchModel
    vp_model: VpModel
    lock_vp: bool
    qc_inversion: bool
    path4vp_model: Optional[Path]


class Postprocessing2dModel(BaseModel):
    interp_dim: Interp_dim


class Postprocessing3dModel(BaseModel):
    num_xslices_3d: int
    num_yslices_3d: int
    num_zslices_3d: int


class PostprocessingModel(BaseModel):
    d_x: int
    d_y: int
    d_z: int
    smooth_factor: Optional[float]
    max_depth: float
    remove_outliers_smoothing: bool
    vmin_in_model: float
    vmax_in_model: float
    save_segy: bool
    save_fdm: bool
    error_thr: float
    parameters_2d: Optional[Postprocessing2dModel]
    parameters_3d: Optional[Postprocessing3dModel]


# ___SWIConfigModel___############
class SWIConfigModel(BaseModel):
    preprocessing: PreprocessingModel
    spectral: SpectralModel
    inversion: InversionModel
    postprocessing: PostprocessingModel

    def extract_core_params(self):
        return (
            self.preprocessing,
            self.spectral,
            self.inversion,
            self.postprocessing
        )


class Ranges(BaseModel):
    velocity_shear_range: Any
    thicknesses_range: Any


class PWaveVelocityModel(BaseModel):
    depth: Any
    vp: Any
    vp2vs: Any


class DispersionCurve(BaseModel):
    frequency: Any
    velocity_phase: Any
    cmp_x: float
    cmp_y: float
    relief: float
    spec_name: str
    num_modes: int

    def save(self, filename: Path) -> None:
        np.savez(
            filename,
            frequency=self.frequency,
            velocity_phase=self.velocity_phase,
            cmp_x=self.cmp_x,
            cmp_y=self.cmp_y,
            relief=self.relief,
            spec_name=self.spec_name,
            num_modes=self.num_modes
        )

    @classmethod
    def load(cls, filename: Path) -> "DispersionCurve":
        data = np.load(filename, allow_pickle=True)
        return cls(
            frequency=list(data["frequency"].item().values()),
            velocity_phase=list(data["velocity_phase"].item().values()),
            cmp_x=float(data["cmp_x"]),
            cmp_y=float(data["cmp_y"]),
            relief=float(data["relief"]),
            spec_name=str(data["spec_name"]),
            num_modes = float(data["num_modes"]),
        )


class ModelVCR(BaseModel):
    thickness: list[float]
    velocity_shear: list[float]
    cmp_x: float
    cmp_y: float
    relief: float
    error_dc: float

    def save(self, filename: Any) -> None:
        np.savez(
            filename,
            thickness=self.thickness,
            velocity_shear=self.velocity_shear,
            cmp_x=self.cmp_x,
            cmp_y=self.cmp_y,
            relief=self.relief,
            error_dc=self.error_dc,
        )

    @classmethod
    def load(cls, filename: Path) -> "ModelVCR":
        data = np.load(filename)
        return cls(
            thickness=data["thickness"],
            velocity_shear=data["velocity_shear"],
            cmp_x=float(data["cmp_x"]),
            cmp_y=float(data["cmp_y"]),
            relief=float(data["relief"]),
            error_dc=float(data["error_dc"]),
        )
