"""Check license metadata and contents of built distribution archives."""

from email.parser import BytesParser
from email.policy import default
from pathlib import Path
import tarfile
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PACKAGE_PREFIX = "optiprofiler_pycutest/"
FORBIDDEN_SUFFIXES = (
    ".a",
    ".c",
    ".cc",
    ".cpp",
    ".dll",
    ".dylib",
    ".f",
    ".f90",
    ".h",
    ".hpp",
    ".lib",
    ".o",
    ".obj",
    ".pyd",
    ".sif",
    ".so",
)


def _find_one(pattern):
    matches = list(DIST.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"Expected one {pattern!r} archive, found {matches}")
    return matches[0]


def _require_suffixes(names, suffixes):
    for suffix in suffixes:
        if not any(name.endswith(suffix) for name in names):
            raise AssertionError(f"Distribution is missing {suffix}")


def _reject_upstream_payload(names, archive_name):
    normalized = [name.lower() for name in names]
    if any(name.endswith(FORBIDDEN_SUFFIXES) for name in normalized):
        raise AssertionError(
            f"{archive_name} unexpectedly contains upstream source or binaries"
        )
    if any("pycutest" in Path(name).parts for name in normalized):
        raise AssertionError(
            f"{archive_name} unexpectedly contains the upstream PyCUTEst package"
        )


def main():
    wheel = _find_one("*.whl")
    sdist = _find_one("*.tar.gz")

    with ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        metadata_name = next(
            name for name in wheel_names if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_name))

    _require_suffixes(
        wheel_names,
        [
            f"{PACKAGE_PREFIX}LICENSE",
            f"{PACKAGE_PREFIX}THIRD_PARTY_NOTICES.md",
        ],
    )
    license_text = metadata.get("License", "")
    if "Redistribution and use in source and binary forms" not in license_text:
        raise AssertionError("Wheel metadata does not contain the BSD license")
    if "License :: OSI Approved :: BSD License" not in metadata.get_all(
        "Classifier", []
    ):
        raise AssertionError("Wheel metadata is missing the BSD classifier")

    _reject_upstream_payload(wheel_names, "Wheel")

    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = archive.getnames()
    _require_suffixes(sdist_names, ["/LICENSE", "/THIRD_PARTY_NOTICES.md"])
    _reject_upstream_payload(sdist_names, "Sdist")


if __name__ == "__main__":
    main()
