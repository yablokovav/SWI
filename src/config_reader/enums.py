from enum import Enum


class TypeData(str, Enum):
    data_2d = "2d"
    data_3d = "3d"


class ExtractDCMethod(str, Enum):
    max = "max"
    ae = "ae"
    dbscan = "dbscan"


class Sort3dOrder(str, Enum):
    csp = "csp"
    cdp = "cdp"


class SpectralMethod(str, Enum):
    sfk = "sfk"
    fkt = "fkt"


class InversionDcMethod(str, Enum):
    ssa = "ssa"
    gwo = "gwo"
    fcnn = "fcnn"
    occam = "occam"


class WaveType(str, Enum):
    rayleigh = "rayleigh"
    love = "love"


class VelocityType(str, Enum):
    phase = "phase"
    group = "group"


class GetNumLayers(str, Enum):
    mean = "mean"
    classes = "classes"
    every = "every"


class VpModel(str, Enum):
    vp = "vp"
    vp2vs = "vp2vs"


class Interp_dim(str, Enum):
    d_1 = "1d"
    d_2 = "2d"
