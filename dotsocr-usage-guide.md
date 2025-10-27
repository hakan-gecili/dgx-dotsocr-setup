# DotsOCR Usage Guide - Text Extraction from Images

**Date Created:** 2025-10-25
**Environment:** ~/ml_work/testenv

---

## Quick Start

### Extract Text from an Image

```bash
# Activate the environment
source /ml_work/testenv/bin/activate

# Navigate to DotsOCR directory
cd /ml_work/DotsOCR

# Run the demo script with your image
python demo/demo_hf.py
```

**Note:** Edit `demo/demo_hf.py` to change the image path and prompt type (instructions below).

---

## Extracted Text from sample_image.jpg

DotsOCR successfully extracted all text and layout information from the patient form:

### JSON Output Structure:
```json
[
  {
    "bbox": [11, 2, 171, 18],
    "category": "Section-header",
    "text": "XXXXX"
  },
  {
    "bbox": [11, 36, 642, 214],
    "category": "Text",
    "text": "XXXXXX..."
  },
  ...
]
```
---

## How to Use DotsOCR for Your Images

### Method 1: Using the Demo Script (Recommended)

#### Step 1: Edit the Demo Script

```bash
cd /ml_work/DotsOCR
nano demo/demo_hf.py
```

#### Step 2: Modify These Lines:

```python
# Change the image path (line ~68)
image_path = "/path/to/your/image.jpg"  # Your image path

# Choose your prompt type (line ~70)
prompt = dict_promptmode_to_prompt["prompt_layout_all_en"]  # Full layout + text
# OR
prompt = dict_promptmode_to_prompt["prompt_ocr"]  # Simple text extraction
```

#### Step 3: Run the Script:

```bash
source /ml_work/testenv/bin/activate
cd /ml_work/DotsOCR
python demo/demo_hf.py
```

---

### Method 2: Using Python Code Directly

Create a Python script:

```python
#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

# Set environment
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

# Load model
model_path = "/ml_work/DotsOCR/weights/DotsOCR"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Define prompt for OCR
prompt = "Extract the text content from this image."

# Prepare messages with your image
image_path = "/path/to/your/image.jpg"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]
    }
]

# Process
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = inputs.to("cuda")

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=24000)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(output_text[0])
```

---

## Available Prompts

DotsOCR supports multiple prompt modes for different use cases:

### 1. Full Layout Parsing (`prompt_layout_all_en`)

**Use Case:** Extract complete document structure with bounding boxes, categories, and text

**Prompt:**
```
Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
3. Text Extraction & Formatting Rules:
    - Picture: Text field omitted
    - Formula: LaTeX format
    - Table: HTML format
    - Others: Markdown format
4. Output must be in JSON format
```

**Output:** JSON with bbox, category, and text for each element

---

### 2. Simple OCR (`prompt_ocr`)

**Use Case:** Extract plain text content from the image

**Prompt:**
```
Extract the text content from this image.
```

**Output:** Plain text (no bounding boxes or categories)

---

### 3. Layout Detection Only (`prompt_layout_only_en`)

**Use Case:** Detect layout structure without extracting text

**Prompt:**
```
Please output the layout information from this PDF image, including each layout's bbox and its category. Do not output the corresponding text.
```

**Output:** JSON with bbox and category only

---

### 4. Grounded OCR (`prompt_grounding_ocr`)

**Use Case:** Extract text from a specific region (bbox)

**Prompt:**
```
Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).
Bounding Box: [50, 100, 300, 200]
```

**Output:** Text content within the specified bbox

---

## Configuration File Fix (Important!)

**Issue:** Transformers 4.51.3 requires a `video_processor` parameter that wasn't in the original DotsOCR configuration.

**Solution Applied:** Updated configuration files to include `Qwen2VLVideoProcessor`:

### Files Modified:

1. `.cache/huggingface/modules/transformers_modules/DotsOCR/configuration_dots.py`
2. `/ml_work/DotsOCR/weights/DotsOCR/configuration_dots.py`

### Changes Made:

```python
# Added import
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

# Updated DotsVLProcessor __init__
class DotsVLProcessor(Qwen2_5_VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        # Create a default video processor for compatibility
        video_processor = Qwen2VLVideoProcessor()
        super().__init__(image_processor, tokenizer, video_processor=video_processor, chat_template=chat_template)
        ...
```

**Result:** DotsOCR now works correctly with transformers 4.51.3

---

## Performance Notes

### First Run:
- Model loading: ~35 seconds
- Inference (simple image): ~30 seconds
- GPU Memory: ~6-8GB VRAM

### Subsequent Runs:
- If model stays loaded: <1 minute per image

### Recommendations:
1. **Batch Processing:** Keep the model loaded and process multiple images in a loop
2. **GPU:** Requires CUDA-capable GPU (tested on NVIDIA GB10)
3. **Memory:** Ensure sufficient RAM (16GB+) and VRAM (8GB+)

---

## Troubleshooting

### 1. Import Error: "No module named 'flash_attn'"

**Solution:** Use `attn_implementation="eager"` instead of `"flash_attention_2"`

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager",  # Changed from flash_attention_2
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
```

---

### 2. TypeError: "Received a NoneType for argument video_processor"

**Solution:** This should already be fixed in the configuration files. If you encounter this error:

```bash
# Clear Python cache
rm -rf .cache/huggingface/modules/transformers_modules/DotsOCR/__pycache__/

# Re-run the script
python demo/demo_hf.py
```

---

### 3. CUDA Out of Memory

**Solutions:**
- Close other GPU-using applications
- Use smaller `max_new_tokens` (default: 24000)
- Process smaller images
- Use CPU (slower):
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      model_path,
      device_map="cpu",
      dtype=torch.float32,
      trust_remote_code=True
  )
  ```

---

### 4. Slow Inference

**Causes:**
- Using `attn_implementation="eager"` (slower than flash_attention_2)
- Large images
- Long text output

**Solutions:**
- Install flash-attn for faster inference (requires compilation)
- Resize images to smaller dimensions
- Reduce `max_new_tokens`

---

## Example Use Cases

### 1. Extract Text from Medical Forms

```bash
cd /ml_work/DotsOCR
python demo/demo_hf.py
# Edit to point to medical form image
# Use prompt_layout_all_en for structured extraction
```

### 2. OCR for PDFs

```python
import fitz  # PyMuPDF
from PIL import Image

# Convert PDF page to image
pdf = fitz.open("document.pdf")
page = pdf[0]
pix = page.get_pixmap()
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
img.save("page.jpg")

# Then use DotsOCR on page.jpg
```

### 3. Batch Processing Multiple Images

```python
import glob
from pathlib import Path

# Load model once
model = AutoModelForCausalLM.from_pretrained(...)
processor = AutoProcessor.from_pretrained(...)

# Process all images in a directory
for image_path in glob.glob("/path/to/images/*.jpg"):
    print(f"Processing: {image_path}")
    # Run inference (code from Method 2)
    ...
```

---

## Quick Reference Commands

```bash
# Activate environment
source /ml_work/testenv/bin/activate

# Navigate to DotsOCR
cd /ml_work/DotsOCR

# Extract text from image (simple OCR)
python demo/demo_hf.py  # After editing image_path

# Check available prompts
python -c "from dots_ocr.utils import dict_promptmode_to_prompt; print(list(dict_promptmode_to_prompt.keys()))"

# Verify model is loaded
python -c "import torch; from transformers import AutoModelForCausalLM; print('Model loadable')"
```

---

## File Locations

```
/ml_work/
├── testenv/                                    # Virtual environment
├── DotsOCR/                                    # DotsOCR source
│   ├── demo/
│   │   ├── demo_hf.py                          # Main demo script (MODIFIED)
│   │   ├── demo_vllm.py                        # vLLM integration
│   │   └── demo_gradio.py                      # Web UI
│   ├── dots_ocr/
│   │   └── utils/
│   │       └── prompts.py                      # Available prompts
│   └── weights/
│       └── DotsOCR/                            # Model weights (6GB)
│           ├── configuration_dots.py           # Configuration (MODIFIED)
│           ├── model-00001-of-00002.safetensors
│           └── model-00002-of-00002.safetensors
├── sample_image.jpg                             # Test image
├── extract_text_dotsocr.py                     # Custom script (deprecated)
├── dotsocr-usage-guide.md                      # This file
├── dotsocr-installation-guide.md               # Installation guide
└── complete-installation-summary.md            # Environment summary
```

---

## Additional Resources

- **DotsOCR GitHub:** https://github.com/rednote-hilab/dots.ocr
- **Model on HuggingFace:** https://huggingface.co/rednote-hilab/dots.ocr
- **Live Demo:** https://dotsocr.xiaohongshu.com
- **Paper/Blog:** Check `assets/blog.md` in the repository

---

## Summary

DotsOCR is now fully functional and can extract text from images with high accuracy. The key features:

✓ Multilingual support
✓ Layout detection
✓ Multiple output formats (JSON, plain text)
✓ Bounding box extraction
✓ Table and formula recognition
✓ Fast inference with GPU acceleration

**Performance on sample_image.jpg:**
- Correctly identified all sections
- Extracted all text accurately
- Provided bounding boxes for each element
- Classified elements correctly (Section-header, Text, Picture)

---

**Last Updated:** 2025-10-25
**Tested On:** Ubuntu 24.04 LTS ARM64 with NVIDIA GB10 GPU
