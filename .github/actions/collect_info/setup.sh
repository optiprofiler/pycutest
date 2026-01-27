#!/usr/bin/env bash
# setup.sh - Set up CUTEst environment for GitHub Actions
#
# This script installs CUTEst and pycutest dependencies required
# for collecting problem information.

set -e
set -x

# Install Python dependencies
python -m pip install --upgrade pip
python -m pip install -r .github/actions/collect_info/requirements.txt

# Download CUTEst and its dependencies
# Install to $HOME/cutest to match workflow environment variable settings
mkdir -p "$HOME/cutest"
git clone --depth 1 --branch v2.2.3 https://github.com/ralna/ARCHDefs.git "$HOME/cutest/archdefs"
git clone --depth 1 --branch v2.1.3 https://github.com/ralna/SIFDecode.git "$HOME/cutest/sifdecode"
git clone --depth 1 --branch v2.0.42 https://github.com/ralna/CUTEst.git "$HOME/cutest/cutest"
git clone --depth 1 --branch v0.5 https://bitbucket.org/optrove/sif.git "$HOME/cutest/mastsif"

# Set the environment variables for CUTEst
export ARCHDEFS="$HOME/cutest/archdefs"
export SIFDECODE="$HOME/cutest/sifdecode"
export CUTEST="$HOME/cutest/cutest"
export MASTSIF="$HOME/cutest/mastsif/sif"
export MYARCH=pc64.lnx.gfo
{
  echo "ARCHDEFS=$ARCHDEFS"
  echo "SIFDECODE=$SIFDECODE"
  echo "CUTEST=$CUTEST"
  echo "MASTSIF=$MASTSIF"
  echo "MYARCH=$MYARCH"
} >> "$GITHUB_ENV"

# Build and install CUTEst using the official installation script
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/jfowkes/pycutest/master/.install_cutest.sh)"

# Install pycutest
python -m pip install pycutest

echo "CUTEst and pycutest setup completed successfully."
