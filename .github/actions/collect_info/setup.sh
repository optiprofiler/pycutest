#!/usr/bin/env bash
# setup.sh - Set up CUTEst environment for GitHub Actions
#
# This script installs CUTEst and pycutest dependencies required
# for collecting problem information.

set -e
set -x

source .github/actions/collect_info/runtime-versions.env

clone_pinned() {
    local url="$1"
    local version="$2"
    local expected_ref="$3"
    local destination="$4"
    git clone --depth 1 --branch "$version" "$url" "$destination"
    local actual_ref
    actual_ref="$(git -C "$destination" rev-parse HEAD)"
    if [ "$actual_ref" != "$expected_ref" ]; then
        echo "Pinned tag mismatch for $url: expected $expected_ref, got $actual_ref" >&2
        exit 1
    fi
}

# Install Python dependencies
python -m pip install --upgrade pip
python -m pip install -r .github/actions/collect_info/requirements.txt

# Download CUTEst and its dependencies
# Install to $HOME/cutest to match workflow environment variable settings
mkdir -p "$HOME/cutest"
clone_pinned https://github.com/ralna/ARCHDefs.git \
    "$ARCHDEFS_VERSION" "$ARCHDEFS_REF" "$HOME/cutest/archdefs"
clone_pinned https://github.com/ralna/SIFDecode.git \
    "$SIFDECODE_VERSION" "$SIFDECODE_REF" "$HOME/cutest/sifdecode"
clone_pinned https://github.com/ralna/CUTEst.git \
    "$CUTEST_VERSION" "$CUTEST_REF" "$HOME/cutest/cutest"
clone_pinned https://bitbucket.org/optrove/sif.git \
    "$MASTSIF_VERSION" "$MASTSIF_REF" "$HOME/cutest/mastsif"

# Set the environment variables for CUTEst
export ARCHDEFS="$HOME/cutest/archdefs"
export SIFDECODE="$HOME/cutest/sifdecode"
export CUTEST="$HOME/cutest/cutest"
export MASTSIF="$HOME/cutest/mastsif"
export MYARCH=pc64.lnx.gfo
{
  echo "ARCHDEFS=$ARCHDEFS"
  echo "SIFDECODE=$SIFDECODE"
  echo "CUTEST=$CUTEST"
  echo "MASTSIF=$MASTSIF"
  echo "MYARCH=$MYARCH"
} >> "$GITHUB_ENV"

# Build and install CUTEst using the official installation script
/bin/bash -c "$(curl -fsSL "https://raw.githubusercontent.com/jfowkes/pycutest/${PYCUTEST_REF}/.install_cutest.sh")"

# Install pycutest
python -m pip install "pycutest==${PYCUTEST_VERSION}"

echo "CUTEst and pycutest setup completed successfully."
