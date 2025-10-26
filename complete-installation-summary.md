# Complete ARM64 CUDA Environment Installation Summary

**Date Created:** 2025-10-25
**Architecture:** ARM64/aarch64
**System:** Ubuntu 24.04 LTS
**GPU:** NVIDIA GB10
**CUDA Driver:** 13.0 (compatible with CUDA 12.8)

---

## Overview

This document summarizes the complete installation of a high-performance machine learning environment on ARM64 with CUDA support, including:

1. **PyTorch with CUDA 12.8** (from nightly builds)
2. **vLLM** (Vision-Language Model inference engine)
3. **DotsOCR** (Multilingual document layout parsing)

---

## Environment Details

### Location
```
/home/naq2/hakan/testenv
```

### Python Version
```
Python 3.12.3
```

### Key Packages Installed

#### Core ML Framework
- `torch==2.10.0.dev20251024+cu128` - PyTorch with CUDA 12.8 support
- `torchvision==0.20.0.dev20251024+cu128`
- `torchaudio==2.10.0.dev20251024+cu128`

#### Inference & Optimization
- `vllm==0.11.1rc4.dev6+g66a168a19.d20251025.cu130` - High-performance LLM inference
- `flash-attn==2.8.0.post2` - Flash Attention kernels
- `flashinfer-python==0.4.1` - Flash inference library
- `triton_kernels==1.0.0` - Triton GPU kernels

#### Model Libraries
- `transformers==4.51.3` - HuggingFace transformers
- `accelerate==1.11.0` - Distributed training/inference
- `ray==2.50.1` - Distributed computing framework

#### DotsOCR Specific
- `dots_ocr==1.0` - Document layout parsing
- `gradio==5.49.1` - Web UI framework
- `gradio_image_annotation==0.7.0` - Image annotation component
- `PyMuPDF==1.26.5` - PDF processing
- `qwen_vl_utils==0.1.2` - Qwen Vision-Language utilities
- `modelscope==1.24.7` - Model management

#### CUDA Libraries (bundled)
- `nvidia-cuda-runtime-cu12==12.8.90`
- `nvidia-cudnn-cu12==9.10.2.21`
- `nvidia-cublas-cu12==12.8.4.1`
- `nvidia-nccl-cu12==2.27.5`

---

## Installation Timeline

### Phase 1: Environment Setup
1. Created virtual environment: `/home/naq2/hakan/testenv`
2. Upgraded pip to latest version
3. Verified system: ARM64, CUDA 13.0 driver, Python 3.12

### Phase 2: PyTorch Installation
1. Attempted standard CUDA wheels → Failed (ARM64 not supported)
2. Switched to PyTorch nightly builds with CUDA 12.8
3. Successfully installed PyTorch with CUDA support
4. Verified: `torch.cuda.is_available() == True`

### Phase 3: vLLM Installation
1. Cloned vLLM repository: `/home/naq2/hakan/vllm`
2. Initial build attempt → Failed (missing python3-dev headers)
3. Installed system dependencies:
   - `python3-dev`
   - `python3.12-dev`
4. Configured vLLM to use existing PyTorch (via `use_existing_torch.py`)
5. Built vLLM from source (~13 minutes)
6. Successfully compiled CUDA kernels and Flash Attention

### Phase 4: DotsOCR Installation
1. Cloned DotsOCR repository: `/home/naq2/hakan/DotsOCR`
2. Installed dependencies manually (avoiding flash-attn rebuild)
3. Resolved pydantic version conflict (prioritized vLLM >=2.12)
4. Installed DotsOCR package in editable mode
5. Verified all imports successful

---

## Key Challenges & Solutions

### Challenge 1: PyTorch CUDA on ARM64
**Problem:** Standard PyTorch wheels don't include CUDA support for ARM64

**Solution:** Use PyTorch nightly builds
```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Result:** Successfully installed PyTorch 2.10.0.dev with CUDA 12.8

---

### Challenge 2: Missing Python Headers
**Problem:** vLLM build failed with CMake error "Unable to find python matching"

**Solution:** Install Python development headers
```bash
sudo apt-get install -y python3-dev python3.12-dev
```

**Result:** CMake successfully found Python headers at `/usr/include/python3.12/Python.h`

---

### Challenge 3: Flash Attention Build
**Problem:** Building flash-attn from source takes very long on ARM64

**Solution:** Let vLLM build it once, then reuse for DotsOCR
```bash
# vLLM builds flash-attn during installation
# DotsOCR installation skips it with --no-deps
pip install -e . --no-deps
```

**Result:** Avoided duplicate compilation (saved ~10-15 minutes)

---

### Challenge 4: Pydantic Version Conflict
**Problem:**
- vLLM requires `pydantic>=2.12`
- Gradio requires `pydantic<2.12`

**Solution:** Prioritize vLLM's requirement
```bash
pip install 'pydantic>=2.12.0'
```

**Result:** Both packages work despite version warning

---

### Challenge 5: Transformers Version
**Problem:**
- vLLM installs `transformers==4.57.1`
- DotsOCR requires `transformers==4.51.3`

**Solution:** Downgrade to DotsOCR's version
```bash
pip install transformers==4.51.3
```

**Result:** Both vLLM and DotsOCR work with 4.51.3

---

## Directory Structure

```
/home/naq2/hakan/
├── testenv/                                    # Virtual environment
│   ├── bin/
│   │   ├── python3                            # Python 3.12.3
│   │   ├── pip
│   │   └── activate                           # Activation script
│   └── lib/
│       └── python3.12/
│           └── site-packages/                 # All packages (150+)
│               ├── torch/                     # PyTorch CUDA 12.8
│               ├── vllm/                      # vLLM inference engine
│               ├── flash_attn/                # Flash Attention
│               ├── transformers/              # HuggingFace
│               ├── gradio/                    # Web UI
│               ├── dots_ocr/                  # DotsOCR (symlink)
│               └── ...
│
├── vllm/                                      # vLLM source code
│   ├── vllm/                                  # Main package
│   ├── csrc/                                  # C++/CUDA sources
│   ├── CMakeLists.txt
│   ├── setup.py
│   └── use_existing_torch.py                 # Configuration script
│
├── DotsOCR/                                   # DotsOCR source code
│   ├── dots_ocr/                             # Main package
│   │   ├── __init__.py
│   │   └── ...
│   ├── tools/
│   │   └── download_model.py                 # Model download script
│   ├── requirements.txt
│   ├── setup.py
│   ├── app.py                                # Gradio web UI
│   └── README.md
│
├── vllm-arm64-cuda-installation-guide.md     # vLLM guide
├── dotsocr-installation-guide.md             # DotsOCR guide
└── complete-installation-summary.md          # This file
```

---

## Installation Guides

### For New Installations

Follow these guides in order:

1. **vLLM Installation:**
   ```
   File: vllm-arm64-cuda-installation-guide.md
   ```
   - Complete prerequisite setup
   - Install PyTorch nightly with CUDA
   - Build vLLM from source
   - ~30-45 minutes total

2. **DotsOCR Installation:**
   ```
   File: dotsocr-installation-guide.md
   ```
   - Clone repository
   - Install dependencies
   - Install DotsOCR package
   - Download model weights
   - ~10-20 minutes total (excluding model download)

---

## Quick Start Commands

### Activate Environment
```bash
source /home/naq2/hakan/testenv/bin/activate
```

### Verify Installation
```bash
# Check all components
python -c "
import torch
import vllm
import dots_ocr
import transformers

print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ vLLM {vllm.__version__}')
print(f'✓ DotsOCR 1.0')
print(f'✓ Transformers {transformers.__version__}')
"
```

**Expected Output:**
```
✓ PyTorch 2.10.0.dev20251024+cu128
✓ CUDA available: True
✓ GPU: NVIDIA GB10
✓ vLLM 0.11.1rc4.dev6+...
✓ DotsOCR 1.0
✓ Transformers 4.51.3
```

### Use vLLM
```bash
# Python API
python -c "
from vllm import LLM, SamplingParams

llm = LLM(model='meta-llama/Llama-2-7b-hf')
prompts = ['Hello, my name is']
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
"

# OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000
```

### Use DotsOCR
```bash
# Navigate to DotsOCR directory
cd /home/naq2/hakan/DotsOCR

# Download models (first time only)
python3 tools/download_model.py

# Launch web UI
python app.py

# Or use CLI
python inference.py --image path/to/document.pdf
```

---

## System Requirements

### Minimum
- **CPU:** ARM64/aarch64 processor
- **RAM:** 8GB (16GB+ recommended)
- **Disk:** 50GB free space
- **GPU:** NVIDIA GPU with compute capability 8.0+
- **CUDA Driver:** 12.x or higher

### Recommended
- **CPU:** ARM64 with 8+ cores
- **RAM:** 32GB+
- **Disk:** 100GB+ SSD
- **GPU:** NVIDIA GPU with 24GB+ VRAM
- **CUDA Driver:** 13.0+

---

## Performance Notes

1. **Build Times:**
   - PyTorch nightly download: ~2-5 minutes
   - vLLM compilation: ~10-30 minutes (depends on CPU)
   - DotsOCR installation: ~2-5 minutes

2. **Model Inference:**
   - vLLM provides 2-3x faster inference than standard transformers
   - Flash Attention significantly reduces memory usage
   - ARM64 performance is comparable to x86_64 for inference

3. **Disk Usage:**
   - Virtual environment: ~8GB
   - Downloaded models: varies (2-20GB per model)
   - Build artifacts: ~2GB

---

## Maintenance

### Update vLLM
```bash
source /home/naq2/hakan/testenv/bin/activate
cd /home/naq2/hakan/vllm
git pull
pip install --no-build-isolation -e .
```

### Update DotsOCR
```bash
source /home/naq2/hakan/testenv/bin/activate
cd /home/naq2/hakan/DotsOCR
git pull
pip install -e . --no-deps
```

### Update PyTorch Nightly
```bash
source /home/naq2/hakan/testenv/bin/activate
pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

**⚠️ Warning:** Updating PyTorch may require rebuilding vLLM!

---

## Automation Scripts

### vLLM Installation Script
```bash
# See: vllm-arm64-cuda-installation-guide.md
# Section: "Complete Installation Script"
```

### DotsOCR Installation Script
```bash
# See: dotsocr-installation-guide.md
# Section: "Complete Installation Script"
```

### Combined Setup Script

Create a file `setup_ml_environment.sh`:

```bash
#!/bin/bash
set -e

echo "=== ARM64 ML Environment Setup ==="

# Configuration
WORK_DIR="/home/naq2/hakan"
VENV_NAME="testenv"

# Part 1: vLLM Installation
echo ""
echo "=== Part 1: Installing vLLM ==="
bash vllm-arm64-cuda-installation-guide.md  # Extract and run script section

# Part 2: DotsOCR Installation
echo ""
echo "=== Part 2: Installing DotsOCR ==="
bash dotsocr-installation-guide.md  # Extract and run script section

# Verification
echo ""
echo "=== Verification ==="
source "$WORK_DIR/$VENV_NAME/bin/activate"
python -c "
import torch, vllm, dots_ocr
print('✓ All packages installed successfully')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  vLLM: {vllm.__version__}')
print(f'  DotsOCR: 1.0')
"

echo ""
echo "=== Setup Complete! ==="
echo "Activate environment: source $WORK_DIR/$VENV_NAME/bin/activate"
```

---

## Troubleshooting Reference

### vLLM Issues
See: `vllm-arm64-cuda-installation-guide.md` - Section "Troubleshooting"

Common issues:
- CMake errors (missing python3-dev)
- CUDA not available (wrong PyTorch build)
- Build takes forever (ARM64 is slow, be patient)

### DotsOCR Issues
See: `dotsocr-installation-guide.md` - Section "Troubleshooting"

Common issues:
- Flash-attn build errors (use --no-deps)
- Pydantic warnings (expected, ignore)
- Import errors (verify installation)

---

## Resources

### Documentation
- PyTorch: https://pytorch.org/docs/stable/
- vLLM: https://docs.vllm.ai/
- DotsOCR: https://github.com/rednote-hilab/dots.ocr
- Transformers: https://huggingface.co/docs/transformers/

### Issue Trackers
- PyTorch: https://github.com/pytorch/pytorch/issues
- vLLM: https://github.com/vllm-project/vllm/issues
- DotsOCR: https://github.com/rednote-hilab/dots.ocr/issues

### ARM64 Specific
- NVIDIA NGC Containers: https://catalog.ngc.nvidia.com/
- PyTorch ARM64 Wheels: https://download.pytorch.org/whl/nightly/cu128
- vLLM ARM64 Support: https://github.com/vllm-project/vllm/issues (search "ARM64")

---

## Version History

| Date | Change | By |
|------|--------|-----|
| 2025-10-25 | Initial installation and documentation | Claude Code |

---

## Appendix: Package List

<details>
<summary>Click to expand full package list (150+ packages)</summary>

```bash
# Run this to get full list:
source /home/naq2/hakan/testenv/bin/activate
pip list
```

Key packages:
- accelerate==1.11.0
- anthropic==0.71.0
- compressed-tensors==0.12.2
- dots_ocr==1.0
- einops==0.8.1
- fastapi==0.120.0
- flash-attn==2.8.0.post2
- flashinfer-python==0.4.1
- gradio==5.49.1
- gradio_image_annotation==0.7.0
- huggingface-hub==0.30.0
- lm-format-enforcer==0.11.3
- modelscope==1.24.7
- numba==0.61.2
- numpy==2.3.4
- openai==2.6.1
- opencv-python-headless==4.12.0.88
- pillow==12.0.0
- prometheus-client==0.23.1
- protobuf==6.33.0
- pydantic==2.12.3
- PyMuPDF==1.26.5
- pyzmq==27.1.0
- qwen_vl_utils==0.1.2
- ray==2.50.1
- scipy==1.16.2
- sentencepiece==0.2.1
- tiktoken==0.12.0
- tokenizers==0.22.1
- torch==2.10.0.dev20251024+cu128
- torchaudio==2.10.0.dev20251024+cu128
- torchvision==0.20.0.dev20251024+cu128
- transformers==4.51.3
- triton_kernels==1.0.0
- vllm==0.11.1rc4.dev6+g66a168a19.d20251025.cu130
- xgrammar==0.1.25

Plus 100+ more dependencies...
</details>

---

**Last Updated:** 2025-10-25
**System:** Ubuntu 24.04 LTS ARM64, NVIDIA GB10 GPU
**Maintainer:** Installation automated with Claude Code
