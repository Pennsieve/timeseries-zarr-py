"""Command-line entry point: NWB file in, published bundle out."""

import logging
import os
import sys
from collections.abc import Sequence

from pynwb import NWBHDF5IO

from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.config import load_config
from timeseries_zarr.nwb_reader import build_sources_from_nwb
from timeseries_zarr.properties import write_properties

logger = logging.getLogger(__name__)


def main(argv: Sequence[str]) -> int:
    """Run one bundle build from the command line and return an exit code.

    Resolves a Config from the process environment and argv, opens the NWB file
    it names, discovers the channel sources, then writes and publishes the
    viewer bundle followed by the output properties naming it. Returns 0 on
    success, 2 on a bad invocation, 1 on a failed build.
    """
    try:
        cfg = load_config(os.environ, argv)
    except ValueError:
        logger.exception("invalid invocation")
        return 2

    try:
        # The sources stream from the open HDF5 file, so the whole write stays
        # inside the context manager.
        with NWBHDF5IO(str(cfg.nwb_path), mode="r") as io:
            nwbfile = io.read()
            sources = build_sources_from_nwb(nwbfile)
            logger.info(
                "writing %d continuous + %d event channels to %s",
                len(sources.continuous),
                len(sources.events),
                cfg.final_dir,
            )
            write_bundle(
                sources.continuous,
                sources.events,
                staging_dir=cfg.staging_dir,
                final_dir=cfg.final_dir,
                opts=cfg.opts,
                recording=sources.recording,
            )
        write_properties(cfg.properties_path, cfg.final_dir)
    except (OSError, ValueError):
        logger.exception("failed to build bundle from %s", cfg.nwb_path)
        return 1

    logger.info("published bundle to %s", cfg.final_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main(sys.argv[1:]))
