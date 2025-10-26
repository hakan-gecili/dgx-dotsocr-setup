# DotsOCR Installation Guide for ARM64

**Target Architecture:** ARM64/aarch64
**Date Created:** 2025-10-25

---

## Overview

This guide provides step-by-step instructions to install DotsOCR (multilingual document layout parsing) on ARM64 systems. DotsOCR requires vLLM and PyTorch with CUDA support, so this guide assumes you have already completed the vLLM installation.

**Repository:** https://github.com/rednote-hilab/dots.ocr

---

## Prerequisites

### System Requirements
- **Architecture:** ARM64/aarch64
- **OS:** Ubuntu 20.04+ (tested on Ubuntu 24.04 LTS)
- **GPU:** NVIDIA GPU with ARM64 driver support
- **CUDA Driver:** Compatible with CUDA 12.x (check with `nvidia-smi`)
- **Python:** 3.12 (or 3.10+)
- **vLLM:** Already installed (see vllm-arm64-cuda-installation-guide.md)

### Check Your System

```bash
# Verify vLLM installation
source /path/to/testenv/bin/activate
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Installation Steps

### Step 1: Clone DotsOCR Repository

```bash
# Navigate to your working directory
cd ~/ml_work  # or your preferred location

# Clone the repository (avoid periods in directory name)
git clone https://github.com/rednote-hilab/dots.ocr.git DotsOCR
cd DotsOCR
```

**Note:** We use `DotsOCR` instead of `dots.ocr` to avoid potential path issues with periods in directory names.

---

### Step 2: Activate Virtual Environment

```bash
# Activate the testenv environment where vLLM is installed
source ~/ml_work/testenv/bin/activate
```

---

### Step 3: Install DotsOCR Dependencies

DotsOCR requires several packages. We'll install them manually to avoid flash-attn rebuild issues.

```bash
# Install core dependencies (skip flash-attn as it's already in vLLM)
pip install gradio gradio_image_annotation PyMuPDF openai qwen_vl_utils \
    transformers==4.51.3 huggingface_hub modelscope accelerate
```

**Key Dependencies:**
- `gradio` - Web UI framework
- `gradio_image_annotation` - Image annotation component
- `PyMuPDF` - PDF processing library
- `openai` - OpenAI API client
- `qwen_vl_utils` - Vision-Language utilities for Qwen models
- `transformers==4.51.3` - HuggingFace transformers (specific version required)
- `huggingface_hub` - Model hub client
- `modelscope` - Model management library
- `accelerate` - Distributed computing library

**Note:** flash-attn is already installed via vLLM, so we skip it to avoid compilation errors.

---

### Step 4: Handle Pydantic Version Conflict

vLLM requires pydantic>=2.12, but gradio prefers pydantic<2.12. We prioritize vLLM's requirement:

```bash
# Upgrade pydantic to satisfy vLLM
pip install 'pydantic>=2.12.0'
```

This may generate warnings from gradio, but both packages will work correctly.

---

### Step 5: Install DotsOCR Package

```bash
# Install DotsOCR in editable mode without rebuilding dependencies
pip install -e . --no-deps
```

The `--no-deps` flag prevents pip from trying to reinstall dependencies (especially flash-attn).

---

### Step 6: Verify Installation

```bash
# Check if DotsOCR can be imported
python -c "import dots_ocr; print('✓ DotsOCR successfully imported')"

# Check all key dependencies
python -c "
import dots_ocr
import gradio
import qwen_vl_utils
import transformers
import accelerate
print('✓ All dependencies imported successfully')
print(f'Transformers version: {transformers.__version__}')
print(f'Gradio version: {gradio.__version__}')
"
```

**Expected Output:**
```
✓ DotsOCR successfully imported
✓ All dependencies imported successfully
Transformers version: 4.51.3
Gradio version: 5.49.1
```

---

### Step 7: Download Model Weights

DotsOCR requires pre-trained model weights. Download them using the provided tool:

```bash
# Activate environment (if not already active)
source ~/ml_work/testenv/bin/activate

# Navigate to DotsOCR directory
cd ~/ml_work/DotsOCR

# Download model weights
python3 tools/download_model.py
```

This will download the Qwen2-VL model weights needed for document parsing.

---

## Usage

### Activating the Environment

Always activate the environment before using DotsOCR:

```bash
source ~/ml_work/testenv/bin/activate
```

### Running DotsOCR

Refer to the repository's README for usage examples. Typical workflows include:

1. **Command Line Interface:**
   ```bash
   python inference.py --image path/to/document.pdf
   ```

2. **Gradio Web UI:**
   ```bash
   python app.py
   ```

3. **Python API:**
   ```python
   from dots_ocr import DotsOCR

   model = DotsOCR()
   results = model.parse_document("path/to/document.pdf")
   print(results)
   ```

---

## Troubleshooting

### Common Issues

#### 1. Flash-attn Build Errors

**Cause:** DotsOCR tries to build flash-attn from source

**Solution:** Install dependencies manually (as shown in Step 3) and use `--no-deps` when installing DotsOCR:
```bash
pip install -e . --no-deps
```

#### 2. Pydantic Version Warnings

**Warning Message:**
```
WARNING: gradio 5.49.1 requires pydantic<2.12, but you have pydantic 2.12.3
```

**Solution:** This is expected. vLLM requires pydantic>=2.12, which takes precedence. Both packages work despite the warning.

#### 3. Transformers Version Conflict

**Cause:** vLLM installs transformers 4.57.1, but DotsOCR needs 4.51.3

**Solution:** The installation process downgrades to 4.51.3 automatically. Verify:
```bash
python -c "import transformers; print(transformers.__version__)"
# Should show: 4.51.3
```

#### 4. Import Errors for dots_ocr

**Cause:** DotsOCR package not installed

**Solution:**
```bash
cd ~/ml_work/DotsOCR
source ~/ml_work/testenv/bin/activate
pip install -e . --no-deps
```

#### 5. Missing Model Weights

**Error Message:**
```
Model checkpoint not found
```

**Solution:**
```bash
cd ~/ml_work/DotsOCR
python3 tools/download_model.py
```

---

## Package Versions

### Key Installed Packages

```
dots_ocr==1.0
gradio==5.49.1
gradio_image_annotation==0.7.0
PyMuPDF==1.26.5
qwen_vl_utils==0.1.2
transformers==4.51.3
modelscope==1.24.7
accelerate==1.11.0
flash-attn==2.8.0.post2 (from vLLM)
pydantic==2.12.3
```

### Inherited from vLLM

These are already installed via vLLM and don't need reinstallation:
- `torch==2.10.0.dev20251024+cu128`
- `vllm==0.11.1rc4.dev6`
- `flash-attn==2.8.0.post2`
- `ray==2.50.1`

---

## Complete Installation Script

Here's a complete script you can run (review and customize as needed):

```bash
#!/bin/bash
set -e  # Exit on error

echo "=== DotsOCR ARM64 Installation Script ==="

# Configuration
WORK_DIR="~/ml_work"
VENV_PATH="$WORK_DIR/testenv"
DOTSOCR_DIR="$WORK_DIR/DotsOCR"

# Step 1: Verify vLLM installation
echo "Step 1: Verifying vLLM installation..."
source "$VENV_PATH/bin/activate"
python -c "import vllm; print(f'✓ vLLM {vllm.__version__} found')" || {
    echo "ERROR: vLLM not installed. Please install vLLM first."
    exit 1
}

# Step 2: Clone DotsOCR repository
echo "Step 2: Cloning DotsOCR repository..."
cd "$WORK_DIR"
if [ -d "$DOTSOCR_DIR" ]; then
    echo "DotsOCR directory already exists, skipping clone"
else
    git clone https://github.com/rednote-hilab/dots.ocr.git DotsOCR
fi
cd "$DOTSOCR_DIR"

# Step 3: Install dependencies
echo "Step 3: Installing dependencies..."
pip install gradio gradio_image_annotation PyMuPDF openai qwen_vl_utils \
    transformers==4.51.3 huggingface_hub modelscope accelerate

# Step 4: Fix pydantic version for vLLM
echo "Step 4: Upgrading pydantic for vLLM compatibility..."
pip install 'pydantic>=2.12.0'

# Step 5: Install DotsOCR package
echo "Step 5: Installing DotsOCR package..."
pip install -e . --no-deps

# Step 6: Verify installation
echo "Step 6: Verifying installation..."
python -c "import dots_ocr; print('✓ DotsOCR installed successfully')"
python -c "import gradio, qwen_vl_utils, transformers; print(f'✓ Dependencies OK (transformers {transformers.__version__})')"

# Step 7: Download models
echo "Step 7: Downloading model weights..."
echo "NOTE: This may take a while depending on your internet connection"
python3 tools/download_model.py || {
    echo "WARNING: Model download failed. You may need to run this manually later:"
    echo "  cd $DOTSOCR_DIR && python3 tools/download_model.py"
}

echo ""
echo "=== Installation Complete! ==="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To use DotsOCR, run:"
echo "  cd $DOTSOCR_DIR"
echo "  python app.py  # For web UI"
echo "  # or"
echo "  python inference.py --image <path>  # For CLI"
echo ""
```

Save this script as `install_dotsocr_arm64.sh` and run:

```bash
chmod +x install_dotsocr_arm64.sh
./install_dotsocr_arm64.sh
```

---

## Environment Layout

```
~/ml_work/
├── testenv/                    # Python virtual environment
│   ├── bin/
│   │   └── python3            # Python 3.12
│   └── lib/
│       └── python3.12/
│           └── site-packages/ # All packages installed here
│               ├── vllm/
│               ├── torch/
│               ├── flash_attn/
│               ├── gradio/
│               ├── dots_ocr/
│               └── ...
├── vllm/                      # vLLM source code
├── DotsOCR/                   # DotsOCR source code
│   ├── dots_ocr/             # Main package
│   ├── tools/                 # Utility scripts
│   │   └── download_model.py
│   ├── requirements.txt
│   ├── setup.py
│   └── README.md
└── vllm-arm64-cuda-installation-guide.md
```

---

## Quick Reference Commands

```bash
# Activate environment
source ~/ml_work/testenv/bin/activate

# Check installations
python -c "import vllm, dots_ocr, torch; print('All OK')"

# Check versions
python -c "import vllm, dots_ocr, transformers; print(f'vLLM: {vllm.__version__}'); print(f'DotsOCR: 1.0'); print(f'Transformers: {transformers.__version__}')"

# Run DotsOCR web UI
cd ~/ml_work/DotsOCR
python app.py

# Download models (if not done during installation)
cd ~/ml_work/DotsOCR
python3 tools/download_model.py
```

---

## Notes

1. **Environment Sharing:** DotsOCR and vLLM share the same virtual environment (`testenv`) to avoid duplication.

2. **Version Pinning:**
   - Transformers is pinned to 4.51.3 (DotsOCR requirement)
   - This is older than vLLM's preferred version (4.57.1), but works for both

3. **Flash Attention:**
   - Already installed via vLLM (version 2.8.0.post2)
   - DotsOCR's requirements.txt lists it, but we skip reinstallation to avoid ARM64 build issues

4. **Pydantic Conflict:**
   - vLLM needs >=2.12, Gradio prefers <2.12
   - We use 2.12.3 (vLLM's requirement takes priority)
   - Both packages function correctly despite version warning

5. **Model Storage:**
   - Downloaded models typically stored in `~/.cache/huggingface/`
   - Large models (several GB) - ensure sufficient disk space

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Ensure vLLM is properly installed first
4. Check DotsOCR GitHub issues: https://github.com/rednote-hilab/dots.ocr/issues
5. For vLLM issues: https://github.com/vllm-project/vllm/issues

---

**Last Updated:** 2025-10-25
**Tested On:** Ubuntu 24.04 LTS ARM64 with NVIDIA GB10 GPU
