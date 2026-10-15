# Releasing

The version of a build comes from the git tag through setuptools-scm.

No file holds a version number.

A build from a commit with no tag gets a development version
such as `0.1.dev3+g1a2b3c4`, which PyPI rejects.

## Procedure

1. Merge the work into `main` and wait for the CI workflow to pass.
2. Move the `Unreleased` entries in [CHANGELOG.md](../CHANGELOG.md) under a heading for
   the new version, and commit the change to `main`. Skip this step for a pre-release.
3. Build the package locally and check it:

   ```bash
   python -m pip install build twine
   python -m build
   python -m twine check --strict dist/*
   ```

   The wheel holds `timeseries_zarr/py.typed` and no `tests/` directory.
4. Create a GitHub release with a tag named `v` followed by the version, for example
   `v0.1.0`, on the tip of `main`.
5. Publishing the release starts the Release workflow. It builds the package, checks
   that the built version equals the tag, installs the wheel into a clean environment,
   and uploads to PyPI through trusted publishing.

PyPI never accepts a second upload of a version, even after a deletion. A release that
fails after the upload needs a new version.

## PyPI setup

The Release workflow uploads with a trusted publisher. Register one on PyPI for the
project `timeseries-zarr` with owner `Pennsieve`, repository `timeseries-zarr-py`,
workflow `release.yml`, and environment `pypi`. Create the matching `pypi` environment
in the repository settings. Add required reviewers to it to add a manual verification
step for each release.

## Version numbers

The version follows Semantic Versioning. While the major version is 0, a minor release
can change the public names that `timeseries_zarr.__all__` lists.

Note: The TS-Zarr specification has its own version separate from this library.
