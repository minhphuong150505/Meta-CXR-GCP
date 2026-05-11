from pathlib import Path
from omegaconf import OmegaConf

_CONFIG_PATH = Path(__file__).parent / "configs" / "env_config.yaml"

if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"env_config.yaml not found at {_CONFIG_PATH}. "
        "Copy configs/env_config.yaml.example to configs/env_config.yaml and fill in your paths."
    )

_cfg = OmegaConf.to_container(OmegaConf.load(str(_CONFIG_PATH)), resolve=True)

PATH_TO_MIMIC_CXR = _cfg["paths"]["data_root"]
VIS_ROOT          = _cfg["paths"]["mimic_cxr_jpg_root"]
SPLIT_CSV         = _cfg["paths"]["split_csv"]
REPORTS_CSV       = _cfg["paths"]["reports_csv"]
CHEXPERT_CSV      = _cfg["paths"]["chexpert_csv"]
METADATA_CSV      = _cfg["paths"]["metadata_csv"]
OUTPUT_DIR        = _cfg["paths"]["output_dir"]

JAVA_HOME         = _cfg["java"]["home"]
JAVA_PATH         = _cfg["java"]["path"]

WANDB_ENTITY      = _cfg["wandb"]["entity"]
WANDB_PROJECT     = _cfg["wandb"]["project"]
