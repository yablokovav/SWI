from pathlib import Path

### Global variables for indexing headers within a processing
HEADER_FFID_IND = 0
HEADER_SOU_X_IND = 1
HEADER_SOU_Y_IND = 2
HEADER_REC_X_IND = 3
HEADER_REC_Y_IND = 4
HEADER_ELEV_IND = 5
HEADER_CDP_X_IND = 6
HEADER_CDP_Y_IND = 7
HEADER_OFFSET_IND = 8

COLL_SIZE = 25  # - number of characters for alignment in logging
WIDTH_FOR_RP = 12  # - number of characters for alignment RP in logging
WIDTH_FOR_NS = 8  # - number of characters for alignment Ns in logging

### Global variables for creating directories
root_dir = Path(__file__).resolve().parent.parent
runs_dir = root_dir / "runs/"
prep_dir = runs_dir / "preprocessing/"
spec_dir = runs_dir / "spectral_analysis/"
inv_dir = runs_dir / "inversion/"
log_dir = runs_dir / "logs/"
