#!/bin/bash
set -e

# Where to install LHAPDF
INSTALL_DIR=$HOME/local/lhapdf

# LHAPDF version
VER=6.5.4

# Create dirs
mkdir -p $INSTALL_DIR
mkdir -p $HOME/src
cd $HOME/src

# Download LHAPDF
wget https://lhapdf.hepforge.org/downloads/?f=LHAPDF-$VER.tar.gz -O LHAPDF-$VER.tar.gz
tar xf LHAPDF-$VER.tar.gz
cd LHAPDF-$VER

# Configure and build
./configure --prefix=$INSTALL_DIR
make -j$(nproc)
make install

# Environment variables
echo "Add these lines to your ~/.bashrc:"
echo "export PATH=$INSTALL_DIR/bin:\$PATH"
echo "export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
echo "export PYTHONPATH=$INSTALL_DIR/lib/python3.\$(python3 -V | cut -d' ' -f2 | cut -d. -f1-2)/site-packages:\$PYTHONPATH"
