#!/usr/bin/env bash
# Setup for FlowDAgger pi0.5 + MetaWorld assembly (paper Table 1, Assembly row).
# Needs: Linux, Python 3.11, CUDA 12 GPU (paper reports ~8 GB VRAM for pi0.5).
set -euo pipefail

git clone https://github.com/microsoft/FlowDAgger.git
cd FlowDAgger
conda create -n flowdagger python=3.11 -y
conda activate flowdagger

git submodule update --init flowdagger_pi05/openpi
pip install -e flowdagger_pi05/openpi
# Must come AFTER openpi so these pins win.
pip install -r flowdagger_pi05/requirements.txt
echo "Setup done. The checkpoint downloads from Hugging Face on first run."