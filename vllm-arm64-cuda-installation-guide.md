# vLLM + PyTorch + CUDA 12.8 Installation Guide for ARM64

**Target Architecture:** ARM64/aarch64
**CUDA Version:** 12.8
**Date Created:** 2025-10-25

---

## Overview

This guide provides step-by-step instructions to install vLLM with CUDA support on ARM64 systems. The standard PyTorch wheels don't include CUDA support for ARM64, so we use PyTorch nightly builds and compile vLLM from source.

---

## Prerequisites

### System Requirements
- **Architecture:** ARM64/aarch64
- **OS:** Ubuntu 20.04+ (tested on Ubuntu 24.04 LTS)
- **GPU:** NVIDIA GPU with ARM64 driver support
- **CUDA Driver:** Compatible with CUDA 12.x (check with `nvidia-smi`)
- **Python:** 3.12 (or 3.10+)

### Check Your System
```bash
# Verify architecture
uname -m
# Should show: aarch64

# Verify NVIDIA driver
nvidia-smi
# Should show CUDA Version 12.x or higher

# Verify Python version
python3 --version
# Should show Python 3.10 or higher
```

---

## Installation Steps

### Step 1: Install System Dependencies

```bash
# Update package list
sudo apt-get update

# Install Python development headers (CRITICAL!)
sudo apt-get install -y python3-dev python3.12-dev

# Install build tools
sudo apt-get install -y build-essential git

# Verify Python.h is installed
ls -la /usr/include/python3.12/Python.h
# Should show the file exists
```

**⚠️ Important:** Without `python3-dev`, the build will fail with CMake errors about missing Python include directories.

---

### Step 2: Create Python Virtual Environment

```bash
# Navigate to your project directory
cd /path/to/your/project

# Create virtual environment
python3 -m venv testenv

# Activate the environment
source testenv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

### Step 3: Install PyTorch Nightly with CUDA Support

**⚠️ Important:** Standard PyTorch wheels for ARM64 are CPU-only. You MUST use the nightly builds for CUDA support.

```bash
# Activate environment (if not already active)
source testenv/bin/activate

# Install PyTorch nightly with CUDA 12.8
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
PyTorch: 2.10.0.devYYYYMMDD+cu128
CUDA available: True
```

If CUDA shows as `False`, the installation failed. Try the nightly build again.

---

### Step 4: Clone vLLM Repository

```bash
# Clone the vLLM repository
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

---

### Step 5: Configure vLLM to Use Existing PyTorch

This script removes PyTorch from vLLM's dependency files so it uses the version we installed:

```bash
# Activate environment (if not already active)
source testenv/bin/activate

# Run the configuration script
python use_existing_torch.py
```

This will clean up all PyTorch references from vLLM's requirements files.

---

### Step 6: Install Build Dependencies

```bash
# Activate environment (if not already active)
source testenv/bin/activate

# Install build requirements
pip install -r requirements/build.txt

# Install additional build tools
pip install wheel packaging ninja
```

---

### Step 7: Build and Install vLLM

**⚠️ Warning:** This step takes 10-30 minutes on ARM64 as it compiles CUDA kernels.

```bash
# Activate environment (if not already active)
source testenv/bin/activate

# Build and install vLLM (editable mode)
pip install --no-build-isolation -e .
```

**What happens during build:**
1. CMake configures the build (finds Python headers, CUDA, etc.)
2. Ninja compiles C++ and CUDA code in parallel
3. Flash Attention kernels are compiled (most time-consuming)
4. vLLM Python package is installed

**Progress indicators:**
- You'll see "Building editable for vllm (pyproject.toml): still running..." multiple times
- This is normal! The build is actively compiling CUDA kernels
- You can check compilation status: `ps aux | grep nvcc | wc -l`

---

### Step 8: Verify Installation

```bash
# Activate environment (if not already active)
source testenv/bin/activate

# Check vLLM version
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected Output:**
```
vLLM version: 0.11.1rc4.dev6+...
PyTorch: 2.10.0.dev...+cu128
CUDA: True
GPU: NVIDIA <YOUR_GPU_NAME>
```

---

## Usage

### Activating the Environment

Always activate the environment before using vLLM:

```bash
source /path/to/testenv/bin/activate
```

### Running vLLM

Example - Start OpenAI-compatible API server:

```bash
# Activate environment
source testenv/bin/activate

# Start API server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000
```

Example - Use vLLM in Python:

```python
from vllm import LLM, SamplingParams

# Create LLM instance
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Generate text
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

---

## Troubleshooting

### Common Issues

#### 1. CMake Error: "Unable to find python matching"

**Cause:** Missing Python development headers

**Solution:**
```bash
sudo apt-get install -y python3-dev python3.12-dev
# Verify installation
ls -la /usr/include/python3.12/Python.h
```

#### 2. CUDA Not Available in PyTorch

**Cause:** Installed CPU-only PyTorch instead of nightly build

**Solution:**
```bash
# Uninstall current PyTorch
pip uninstall -y torch torchvision torchaudio

# Reinstall PyTorch nightly with CUDA
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 3. Build Fails with "No matching distribution found for torch"

**Cause:** Trying to use standard CUDA wheels on ARM64

**Solution:** ARM64 doesn't have standard CUDA wheels. Use nightly builds as shown in Step 3.

#### 4. Build Takes Forever / Appears Stuck

**Cause:** Compiling CUDA kernels is slow on ARM64

**Solution:**
- Be patient! Build takes 10-30 minutes
- Check if it's actually compiling: `ps aux | grep nvcc`
- If you see nvcc processes, it's working

#### 5. GPU Capability Warning

**Warning Message:**
```
Found GPU0 <NAME> which is of cuda capability X.Y.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

**Solution:** This is just a warning. PyTorch will still use your GPU. You can ignore it unless you encounter actual runtime errors.

---

## Alternative Installation Options

### Option A: Build PyTorch from Source

If nightly builds don't work, you can build PyTorch from source:

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export USE_CUDA=1
export USE_CUDNN=1
python setup.py install
```

**Note:** This takes 1-2 hours on ARM64.

### Option B: Use NVIDIA NGC Containers

NVIDIA provides pre-built containers with PyTorch for ARM64:

```bash
docker pull nvcr.io/nvidia/pytorch:24.10-py3-arm64
docker run --gpus all -it nvcr.io/nvidia/pytorch:24.10-py3-arm64
```

Then build vLLM inside the container.

---

## Environment Details

### What Was Installed

#### Python Packages (Key Dependencies)
- `torch==2.10.0.dev20251024+cu128` - PyTorch with CUDA 12.8
- `vllm==0.11.1rc4.dev6+...` - vLLM (built from source)
- `transformers==4.57.1` - Hugging Face transformers
- `ray==2.50.1` - Distributed computing
- `flashinfer-python==0.4.1` - Flash attention implementation
- `triton_kernels==1.0.0` - Triton kernels
- Plus 100+ additional dependencies

#### CUDA Libraries (Bundled with PyTorch)
- `nvidia-cuda-runtime-cu12==12.8.90`
- `nvidia-cudnn-cu12==9.10.2.21`
- `nvidia-cublas-cu12==12.8.4.1`
- `nvidia-nccl-cu12==2.27.5`
- And other CUDA libraries

---

## Quick Reference Commands

```bash
# Activate environment
source /path/to/testenv/bin/activate

# Update vLLM (rebuild from source)
cd /path/to/vllm
git pull
pip install --no-build-isolation -e .

# Check versions
python -c "import vllm, torch; print(f'vLLM: {vllm.__version__}, PyTorch: {torch.__version__}')"

# Test CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU
nvidia-smi
```

---

## Notes

1. **PyTorch Nightly Builds**: The nightly builds are updated daily. If you reinstall weeks later, you might get a newer version.

2. **vLLM Updates**: vLLM is under active development. You can update by running `git pull` and rebuilding.

3. **CUDA Version**: Ensure your NVIDIA driver supports CUDA 12.8 or higher. Check with `nvidia-smi`.

4. **Disk Space**: The build requires ~10GB of temporary disk space during compilation.

5. **Memory**: Compiling vLLM uses significant RAM. Ensure you have at least 8GB available.

---

## Complete Installation Script

Here's a complete script you can run (review and customize as needed):

```bash
#!/bin/bash
set -e  # Exit on error

echo "=== vLLM ARM64 CUDA Installation Script ==="

# Configuration
INSTALL_DIR="$HOME/vllm-workspace"
VENV_NAME="testenv"
PYTHON_VERSION="python3.12"

# Step 1: Install system dependencies
echo "Step 1: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-dev python3.12-dev build-essential git

# Step 2: Create project directory
echo "Step 2: Creating project directory..."
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Step 3: Create virtual environment
echo "Step 3: Creating virtual environment..."
$PYTHON_VERSION -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# Step 4: Upgrade pip
echo "Step 4: Upgrading pip..."
pip install --upgrade pip

# Step 5: Install PyTorch nightly
echo "Step 5: Installing PyTorch nightly with CUDA 12.8..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Step 6: Verify PyTorch CUDA
echo "Step 6: Verifying PyTorch CUDA support..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('✓ CUDA is available')"

# Step 7: Clone vLLM
echo "Step 7: Cloning vLLM repository..."
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Step 8: Configure vLLM
echo "Step 8: Configuring vLLM to use existing PyTorch..."
python use_existing_torch.py

# Step 9: Install build dependencies
echo "Step 9: Installing build dependencies..."
pip install -r requirements/build.txt
pip install wheel packaging ninja

# Step 10: Build vLLM
echo "Step 10: Building vLLM (this will take 10-30 minutes)..."
pip install --no-build-isolation -e .

# Step 11: Verify installation
echo "Step 11: Verifying installation..."
python -c "import vllm; print(f'✓ vLLM {vllm.__version__} installed successfully')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "=== Installation Complete! ==="
echo ""
echo "To activate the environment, run:"
echo "  source $INSTALL_DIR/$VENV_NAME/bin/activate"
echo ""
echo "To test vLLM, run:"
echo "  python -c 'import vllm; print(vllm.__version__)'"
echo ""
```

Save this script as `install_vllm_arm64.sh` and run:

```bash
chmod +x install_vllm_arm64.sh
./install_vllm_arm64.sh
```

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Ensure Python development headers are installed
4. Check vLLM GitHub issues: https://github.com/vllm-project/vllm/issues
5. Check PyTorch forums: https://discuss.pytorch.org/

---

**Last Updated:** 2025-10-25
**Tested On:** Ubuntu 24.04 LTS ARM64 with NVIDIA GB10 GPU
