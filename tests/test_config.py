from pathlib import Path

import pytest

from timeseries_zarr.config import Config, load_config
from timeseries_zarr.types import WriteOpts


def test_two_positionals_populate_paths():
    cfg = load_config({}, ["in.nwb", "out_bundle"])
    assert isinstance(cfg, Config)
    assert cfg.nwb_path == Path("in.nwb")
    assert cfg.final_dir == Path("out_bundle")


def test_unset_settings_use_writeopts_defaults():
    cfg = load_config({}, ["in.nwb", "out_bundle"])
    assert cfg.opts == WriteOpts()


def test_unset_staging_derives_from_final_dir():
    cfg = load_config({}, ["in.nwb", "out_bundle"])
    assert cfg.staging_dir != cfg.final_dir
    assert cfg.staging_dir.parent == cfg.final_dir.parent


def test_unset_properties_file_uses_the_default_name():
    cfg = load_config({}, ["in.nwb", "out/session.zarr"])
    assert cfg.properties_path == Path("out/asset-properties.json")


def test_env_properties_file_honored():
    cfg = load_config(
        {"ASSET_PROPERTIES_FILE": "declared.json"},
        ["in.nwb", "out/session.zarr"],
    )
    assert cfg.properties_path == Path("out/declared.json")


def test_properties_file_is_a_sibling_of_the_bundle():
    cfg = load_config({}, ["in.nwb", "out/session.zarr"])
    assert cfg.properties_path.parent == cfg.final_dir.parent


def test_env_staging_dir_honored():
    cfg = load_config(
        {"ZARR_WRITER_STAGING_DIR": "scratch/stage"},
        ["in.nwb", "out_bundle"],
    )
    assert cfg.staging_dir == Path("scratch/stage")


def test_env_settings_override_defaults():
    cfg = load_config(
        {
            "ZARR_WRITER_ZSTD_LEVEL": "9",
            "ZARR_WRITER_MAX_LEVELS": "4",
            "ZARR_WRITER_MIN_BINS": "512",
            "ZARR_WRITER_INNER_LEN": "1024",
            "ZARR_WRITER_TARGET_SHARD_BYTES": "2048",
        },
        ["in.nwb", "out_bundle"],
    )
    assert cfg.opts == WriteOpts(
        zstd_level=9,
        max_levels=4,
        min_bins=512,
        inner_len=1024,
        target_shard_bytes=2048,
    )


def test_max_levels_above_the_format_cap_raises():
    with pytest.raises(ValueError):
        load_config({"ZARR_WRITER_MAX_LEVELS": "9"}, ["in.nwb", "out_bundle"])


def test_max_levels_below_one_raises():
    with pytest.raises(ValueError):
        load_config({"ZARR_WRITER_MAX_LEVELS": "0"}, ["in.nwb", "out_bundle"])


def test_max_levels_at_the_cap_is_accepted():
    cfg = load_config({"ZARR_WRITER_MAX_LEVELS": "7"}, ["in.nwb", "out_bundle"])
    assert cfg.opts.max_levels == 7


def test_zero_positionals_raises():
    with pytest.raises(ValueError):
        load_config({}, [])


def test_one_positional_raises():
    with pytest.raises(ValueError):
        load_config({}, ["in.nwb"])


def test_zero_positionals_without_dirs_raises():
    with pytest.raises(ValueError):
        load_config({"INPUT_DIR": "/data/input"}, [])


def test_input_dir_scan_finds_sole_nwb(tmp_path):
    nwb = tmp_path / "session.nwb"
    nwb.write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    out = tmp_path / "out"
    cfg = load_config({"INPUT_DIR": str(tmp_path), "OUTPUT_DIR": str(out)}, [])
    assert cfg.nwb_path == nwb
    assert cfg.final_dir == out / "session.zarr"


def test_input_dir_with_no_nwb_raises(tmp_path):
    with pytest.raises(ValueError):
        load_config(
            {"INPUT_DIR": str(tmp_path), "OUTPUT_DIR": str(tmp_path / "out")},
            [],
        )


def test_input_dir_with_multiple_nwb_raises(tmp_path):
    (tmp_path / "a.nwb").write_bytes(b"")
    (tmp_path / "b.nwb").write_bytes(b"")
    with pytest.raises(ValueError):
        load_config(
            {"INPUT_DIR": str(tmp_path), "OUTPUT_DIR": str(tmp_path / "out")},
            [],
        )


def test_non_integer_setting_raises():
    with pytest.raises(ValueError):
        load_config(
            {"ZARR_WRITER_ZSTD_LEVEL": "high"}, ["in.nwb", "out_bundle"]
        )
