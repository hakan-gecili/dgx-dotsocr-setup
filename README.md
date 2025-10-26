# DGX Spark DotsOCR Installation & Usage Guide (ARM64 CUDA 12.8)

This repository documents how I brought together **vLLM** and **DotsOCR** on an ARM64 DGX machine with CUDA 12.8.  The standard PyTorch packages don’t support CUDA on ARM64 and both vLLM and DotsOCR have tricky build and version requirements.  I captured the pitfalls I hit and the solutions that worked so that you don’t have to troubleshoot from scratch.

## What’s the problem?

Setting up DotsOCR on a modern DGX (ARM64) isn’t as simple as `pip install dots_ocr`.  You need GPU‑enabled PyTorch, you have to compile vLLM from source, and some of the dependencies (such as **pydantic** and **transformers**) conflict with one another.  Flash‑Attention kernels are optional on ARM64 and will fall back to a slower *eager* implementation if your GPU doesn’t support them.

## High‑level solution

1. **Prepare your environment** – create a virtualenv and install development headers.
2. **Install GPU‑enabled PyTorch** – use the PyTorch *nightly* wheels built with CUDA 12.8 for ARM64.
3. **Build vLLM from source** – clone the repository, remove PyTorch from its dependencies via `use_existing_torch.py`, install build tools, then compile.
4. **Install DotsOCR** – clone the repository, install its dependencies manually (skipping flash‑attention), resolve version conflicts, and install in editable mode.
5. **Download the model weights** – run the provided script to fetch Qwen 2–VL weights.
6. **Run DotsOCR** – use the CLI, the Gradio web UI or the Python API to extract text and layout from documents.

The detailed guides for each component are included in this repository.

## Repository contents

* `complete-installation-summary.md` – A summary of my environment, including installed package versions, timeline and lessons learned.
* `vllm-arm64-cuda-installation-guide.md` – Step‑by‑step instructions to install **vLLM** and CUDA‑enabled **PyTorch** on ARM64.
* `dotsocr-installation-guide.md` – Instructions to install **DotsOCR** on top of vLLM, including how to resolve dependency conflicts.
* `dotsocr-usage-guide.md` – Examples and tips for running DotsOCR for text and layout extraction.

## Quickstart

### 1. Create and activate a virtual environment

```
# replace ~/ml_work with your preferred directory
mkdir -p ~/ml_work && cd ~/ml_work
python3 -m venv testenv
source testenv/bin/activate
pip install --upgrade pip
```

### 2. Install PyTorch nightly with CUDA 12.8

PyTorch’s standard ARM64 wheels are CPU‑only.  Use the nightly index for CUDA support:

```
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The second command should print a CUDA‑enabled version (e.g., `2.10.0.dev20251024+cu128`) and `True`.

### 3. Build vLLM

```
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Remove PyTorch from vLLM’s dependency list so it uses the version you just installed
python use_existing_torch.py

# Install build tools
pip install -r requirements/build.txt
pip install wheel packaging ninja

# Build and install vLLM (this can take 10–30 minutes on ARM64)
pip install --no-build-isolation -e .
```

### 4. Install DotsOCR

```
git clone https://github.com/rednote-hilab/dots.ocr.git DotsOCR
cd DotsOCR

# Install dependencies manually to avoid rebuilding flash‑attention
pip install gradio gradio_image_annotation PyMuPDF openai qwen_vl_utils \
    transformers==4.51.3 huggingface_hub modelscope accelerate

# vLLM requires pydantic ≥ 2.12, but gradio warns about it.  This upgrade is intentional.
pip install 'pydantic>=2.12.0'

# Install DotsOCR in editable mode without reinstalling dependencies
pip install -e . --no-deps
```

### 5. Download the model weights

```
cd /path/to/DotsOCR
python3 tools/download_model.py
```

The script will download the Qwen 2–VL weights into the `weights/` directory.  They are several gigabytes, so ensure you have enough disk space.

### 6. Run DotsOCR

Use one of the following methods:

* **Command line** – for quick OCR of a PDF or image:

  ```
  python inference.py --image /path/to/document.pdf
  ```

* **Gradio web UI** – launch a local web interface in your browser:

  ```
  python app.py
  ```

* **Python API** – integrate OCR into your own scripts:

  ```python
  from dots_ocr import DotsOCR

  model = DotsOCR()
  results = model.parse_document("/path/to/document.pdf")
  print(results)
  ```

See `dotsocr-usage-guide.md` for more examples, including how to customise prompts (full layout, plain text, specific regions, etc.).

## Troubleshooting and notes

* If the vLLM build fails with a CMake error such as *“Unable to find python matching”*, install the Python development headers: `sudo apt-get install python3-dev python3.12-dev`.
* If PyTorch reports `CUDA available: False`, you likely installed the CPU‑only version.  Uninstall torch/torchvision/torchaudio and reinstall them using the nightly channel.
* DotsOCR may display warnings about pydantic versions.  vLLM’s requirement (≥2.12) takes precedence and both packages will still function correctly.
* If Flash‑Attention is not available on your GPU, the model will automatically fall back to the slower *eager* attention implementation (you might see a log message like “flash attention not available! fallback to eager implementation”).  You can also set `attn_implementation="eager"` when loading the model.

For more detailed troubleshooting (including package version conflicts, disk usage and performance notes), refer to the accompanying guides.

## Contributing

Feel free to open issues or pull requests if you replicate this on another architecture or improve these instructions.  I hope this saves you time getting DotsOCR up and running on ARM64!
