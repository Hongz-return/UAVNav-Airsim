"""Shared training method settings and checkpoint input contract."""

from configparser import ConfigParser
from pathlib import Path


METHOD_CONFIGS = {
    "sac": Path("configs/config_sac_framestack4.ini"),
    "coordatt": Path("configs/config_coordatt_sac.ini"),
    "lstm": Path("configs/config_lstm.ini"),
    "mamba": Path("configs/config_mamba.ini"),
    "ma": Path("configs/config_mamba_coordatt.ini"),
    "transformer": Path("configs/config_transformer.ini"),
    "ma_motion": Path("configs/config_ma_motion.ini"),
    "ma_aux": Path("configs/config_ma_aux.ini"),
    "ma_motion_aux": Path("configs/config_ma_motion_aux.ini"),
}
DEFAULT_METHODS = ["sac", "coordatt", "lstm", "mamba", "ma", "transformer"]

CH4_OBSERVATION_CONTRACT = {
    ("options", "navigation_3d"): "True",
    ("options", "using_velocity_state"): "True",
    ("options", "perception"): "depth_state",
    ("environment", "max_depth_meters"): "25",
    ("environment", "screen_height"): "60",
    ("environment", "screen_width"): "90",
    ("environment", "distance_norm_scale"): "200.0",
    ("environment", "max_vertical_difference"): "2.0",
    ("mamba", "n_stack"): "1",
    ("mamba", "inference_seq_len"): "8",
}


def read_ini(path: Path) -> ConfigParser:
    config = ConfigParser()
    with path.open(encoding="utf-8-sig") as handle:
        config.read_file(handle)
    return config


def assert_observation_contract(cfg: ConfigParser) -> None:
    for key, expected in CH4_OBSERVATION_CONTRACT.items():
        value = cfg.get(*key)
        if value != expected:
            raise RuntimeError(f"Unexpected checkpoint input contract at {key}: {value}")
