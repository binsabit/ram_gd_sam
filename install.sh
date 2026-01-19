#!/bin/bash
set -e  # Exit immediately if any command fails

# --- Configuration ---
ENV_NAME="env312"
PYTHON_VER="3.12"

echo "====================================================="
echo "   Setting up Conda Environment: $ENV_NAME"
echo "====================================================="

# 1. Create the environment if it doesn't exist
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists."
else
    echo "Creating environment $ENV_NAME with Python $PYTHON_VER..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VER"
fi

# 2. Activate the environment strictly for this script
# (This is the trick to make 'conda activate' work inside a shell script)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "--- Active Environment: $(which python) ---"

# 3. Install core build tools (Fixes 'No module named pip' errors)
echo "Installing pip and build tools..."
conda install -y pip
pip install setuptools wheel ninja

# 4. Install PyTorch FIRST (Critical for GroundingDINO/SAM-2)
echo "Installing PyTorch..."
pip install "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" --index-url https://download.pytorch.org/whl/cu121

# 5. Install GroundingDINO & SAM-2 manually (No Build Isolation)
# We disable build isolation so they can find the PyTorch we just installed.
echo "Compiling GroundingDINO..."
pip install --no-build-isolation "git+https://github.com/IDEA-Research/GroundingDINO.git@856dde20aee659246248e20734ef9ba5214f5e44"

echo "Compiling SAM-2..."
pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

# 6. Install the rest of requirements.txt
# We filter out 'groundingdino' and 'SAM-2' to prevent pip from trying to rebuild them and crashing.
echo "Installing remaining requirements..."
grep -vE "groundingdino|SAM-2" requirements.txt | pip install -r /dev/stdin

echo "====================================================="
echo "   INSTALLATION COMPLETE"
echo "====================================================="
echo "To start using the environment, run:"
echo "conda activate $ENV_NAME"