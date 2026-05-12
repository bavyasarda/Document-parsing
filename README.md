# 🔍 OCR with Qwen2.5-VL-3B-Instruct

> Extract text from images and PDFs using Alibaba's latest vision-language model — with keyword search, structured output, and a Gradio UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Transformers](https://img.shields.io/badge/Transformers-4.52%2B-orange?style=flat-square&logo=huggingface)
![Gradio](https://img.shields.io/badge/Gradio-5.0%2B-ff6b6b?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-T4%20%2F%20Colab-76b900?style=flat-square&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

This project implements a fully working OCR pipeline on top of **Qwen2.5-VL-3B-Instruct** — Alibaba's Jan 2025 vision-language model with dedicated document understanding and table OCR training. It runs comfortably on a free **Google Colab T4 GPU** (~7 GB VRAM) and supports both image and multi-page PDF inputs.

The Gradio interface exposes four outputs for every run:

| Output | Description |
|---|---|
| 📝 Extracted Text | Raw OCR output, preserved line breaks and table structure |
| 🔎 Search Result | Keyword match with count and character positions |
| 🗂 Structured Text | Text split into labelled paragraphs |
| 📦 JSON Export | Machine-readable paragraph array with word counts |

---

## ✨ Features

- **Image OCR** — JPEG, PNG, WebP, BMP
- **PDF OCR** — multi-page, processed page by page with `── Page N ──` separators
- **Keyword Search** — case-insensitive, returns match count and all character positions
- **Structured Output** — paragraph splitting with readable labels
- **JSON Export** — structured data ready for downstream processing
- **Custom Prompts** — override the user and system prompts via the Advanced accordion for selective extraction (e.g. "List only course codes and grades")
- **Hindi + English** — multilingual document support out of the box

---

## 🆚 Why Qwen2.5-VL-3B and not 2B?

| Model | VRAM | OCR Quality | Notes |
|---|---|---|---|
| Qwen2.5-VL-2B | ~5 GB | ❌ Poor | Outputs bounding-box coordinates instead of text |
| **Qwen2.5-VL-3B** ← **this project** | ~7 GB | ✅ Good | Sweet spot — document OCR training, fits free T4 |
| Qwen2.5-VL-7B | ~15 GB | ✅✅ Better | Fits T4 but tight; use if accuracy matters more than speed |
| Qwen2.5-VL-72B | ~140 GB | 🏆 Best | Needs A100, not practical for Colab |

---

## 🚀 Quick Start

### Google Colab (recommended)

1. Open `ocr_qwen25vl_3b.ipynb` in Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Click **Runtime → Run all**
4. Open the public Gradio link printed in Cell 7

### Local machine

```bash
# 1. Clone the repo
git clone https://github.com/your-username/qwen-ocr.git
cd qwen-ocr

# 2. Install system dependency (Linux / WSL)
sudo apt-get install -y poppler-utils

# macOS
brew install poppler

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the notebook or convert to script and run
jupyter notebook ocr_qwen25vl_3b.ipynb
```

---

## 📦 Requirements

```
torch>=2.2.0
torchvision>=0.17.0
transformers>=4.52.0        # Qwen2.5-VL requires this minimum version
accelerate>=0.33.0
qwen-vl-utils>=0.0.11
gradio>=5.0.0
Pillow>=10.0.0
pdf2image>=1.17.0
```

> **System dependency:** `poppler-utils` must be installed for PDF support.
> The notebook installs it automatically via `apt-get` in Cell 1.

---

## 🗂 Project Structure

```
qwen-ocr/
├── ocr_qwen25vl_3b.ipynb   # Main notebook (7 cells, run top to bottom)
├── requirements.txt         # Python dependencies
└── README.md
```

### Notebook cells at a glance

| Cell | Purpose |
|---|---|
| 1 | Install all dependencies (poppler + pip packages) |
| 2 | Imports, device detection (CUDA / CPU), sanity checks |
| 3 | Load `Qwen2.5-VL-3B-Instruct` model and processor |
| 4 | Core functions: `extract_text`, `search_keyword`, `structure_text` |
| 5 | Pipeline: image path + PDF path → all four outputs |
| 6 | Gradio Blocks UI definition |
| 7 | `demo.launch(share=True)` |

---

## 🔧 How It Works

### The OCR bug in older Qwen2-VL code — and how it's fixed here

The original Qwen2-VL-2B implementations produced output like `(79,57),(837,149)` instead of text. Three bugs caused this:

**1. Missing system prompt**
Without an explicit instruction, Qwen defaults to "describe what I see" mode and emits spatial coordinates. Fixed with:
```python
DEFAULT_SYSTEM = (
    "You are an expert OCR engine. "
    "Extract ALL text from the image exactly as it appears — "
    "preserve line breaks, table structure, headings, and numbers. "
    "Do NOT output bounding boxes, coordinates, or JSON. "
    "Output plain text only."
)
```

**2. Wrong model class**
Qwen2.5-VL requires its own class, not the generic `AutoModelForVision2Seq`:
```python
# ❌ Old (breaks with Qwen2.5-VL)
from transformers import AutoModelForVision2Seq

# ✅ Correct
from transformers import Qwen2_5_VLForConditionalGeneration
```

**3. pixel_values not cast to float16**
On GPU, calling `.to(DEVICE)` without separately casting pixel tensors causes silent precision failure:
```python
inputs = inputs.to(DEVICE)
if "pixel_values" in inputs and DTYPE == torch.float16:
    inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)
```

---

## 🖥 Gradio UI

The interface is built with **Gradio 5 Blocks** and has two columns:

**Left — Inputs**
- Image upload (drag & drop)
- PDF upload (multi-page supported)
- Keyword search box
- Advanced accordion with editable user and system prompts

**Right — Outputs**
- Extracted text with copy button
- Search result
- Structured paragraph view with copy button
- JSON output (collapsible tree)

> If you provide both an image and a PDF, the PDF takes priority.

---

## 💡 Usage Tips

**For documents (grade cards, invoices, forms):**
Leave everything at default. The system prompt already handles tables and structured text well.

**For selective extraction:**
Open the *Advanced* accordion and change the user prompt, e.g.:
- `"List only the course codes and their grades."`
- `"Extract all dates and monetary amounts."`
- `"What is the student's CGPA?"`

**For multi-page PDFs:**
Output will be labelled `── Page 1 ──`, `── Page 2 ──`, etc. Each page is processed independently and concatenated.

**Re-searching without re-running OCR:**
Type a new keyword in the search box and press **Enter** — it re-searches the already-extracted text without calling the model again.

---

## ⚠️ Known Limitations

- **CPU is very slow** — Qwen2.5-VL-3B on CPU takes 5–10 minutes per image. Always use a T4 GPU.
- **Handwritten text** — accuracy drops significantly for cursive or informal handwriting.
- **Very small fonts** — increase `dpi` in `convert_from_path(pdf_path, dpi=250)` to 300 or 350 for tiny text.
- **Long PDFs** — each page takes ~10–15s on T4. A 20-page PDF takes ~4 minutes.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) by Alibaba Cloud
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Gradio](https://www.gradio.app/) by Hugging Face
- [pdf2image](https://github.com/Belval/pdf2image)
