"""
Log‑Summariser — pure TensorFlow
────────────────────────────────
✓ No PyTorch, no pipeline(), no from_tf flag
✓ Gradio UI: sample logs, file upload, token counter, download button
"""

from __future__ import annotations
import tempfile
from datetime import datetime
from typing import Tuple, Optional, Union
from pathlib import Path

import gradio as gr
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    TFAutoModelForSeq2SeqLM,
)

MODEL_ID = "VidhuMathur/bart-log-summarization"

# ── 1. Load model & tokenizer (once, at import time) ─────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, from_pt=False)  # pure‑TF

# ── 2. tiny demo corpus ──────────────────────────────────────────────────────
sample_logs = {
    "Test Run With Errors": """[INFO] Starting test run...
[INFO] TestLogin started
[ERROR] AssertionError: Expected True but got False
[INFO] TestLogin failed
[INFO] TestCheckout started
[INFO] TestCheckout passed
[WARNING] Deprecated API used in TestCart
[INFO] Test run complete""",
    "CI Pipeline Failure": """[INFO] Starting CI pipeline...
[INFO] Step 'Install dependencies' completed
[ERROR] Step 'Run unit tests' failed: 12 tests failed
[WARNING] Step 'Upload coverage' skipped due to failures
[INFO] Pipeline finished with errors""",
}

# ── 3. helpers ───────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Keep only lines that look interesting (ERROR / FAIL / WARNING)."""
    focus = ("ERROR", "FAIL", "WARNING")
    lines = [ln for ln in text.splitlines() if any(k in ln.upper() for k in focus)]
    return "\n".join(lines) or text


def do_summarise(raw_log: str) -> Tuple[str, str]:
    """Core model call — returns (summary, token‑count‑string)."""
    if not raw_log.strip():
        return "", "⚠️ No log text."

    cleaned = preprocess(raw_log)[:1024]  # hard cap for demo
    enc = tokenizer(cleaned, return_tensors="tf", truncation=True, max_length=1024)
    n_in = int(enc["input_ids"].shape[1])

    out_ids = model.generate(
        enc["input_ids"], max_length=80, min_length=20, do_sample=False, num_beams=4
    )
    summary = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    return summary, f"**Input tokens:** {n_in}"


def save_to_file(text: str) -> Optional[str]:
    if not text.strip():
        return None
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", mode="w", encoding="utf‑8"
    ) as fp:
        fp.write(text)
        return fp.name


def load_sample(name: str) -> str:
    return sample_logs.get(name, "")


def read_upload(file_obj) -> str:
    """
    Accept whatever Gradio passes (bytes, path‑string, dict, list, or file‑like)
    and return the text content of the uploaded file.
    """
    if not file_obj:
        return ""

    # 1️⃣  Spaces often wrap the upload in a single‑item list
    if isinstance(file_obj, list):
        file_obj = file_obj[0]

    # 2️⃣  If we already have the raw bytes (type="binary")
    if isinstance(file_obj, (bytes, bytearray)):
        return file_obj.decode("utf‑8", errors="ignore")

    # 3️⃣  If Gradio gave us a path string or pathlib.Path
    if isinstance(file_obj, (str, Path)):
        try:
            return Path(file_obj).read_text(encoding="utf‑8", errors="ignore")
        except Exception:
            return "⚠️ Could not read file."

    # 4️⃣  If it’s the new FileData dict form
    if isinstance(file_obj, dict):
        # try path first
        path = file_obj.get("path") or file_obj.get("name")
        if path and Path(path).exists():
            try:
                return Path(path).read_text(encoding="utf‑8", errors="ignore")
            except Exception:
                pass
        # fall back to base64 payload
        if "data" in file_obj:
            import base64
            try:
                decoded = base64.b64decode(file_obj["data"])
                return decoded.decode("utf‑8", errors="ignore")
            except Exception:
                pass
        return "⚠️ Unknown FileData structure."

    # 5️⃣  Old‑style file‑like object
    if hasattr(file_obj, "read"):
        try:
            return file_obj.read().decode("utf‑8", errors="ignore")
        except Exception:
            return "⚠️ Could not read stream."

    return "⚠️ Unsupported upload type."



# ── 4. Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="🧠 Log Summariser") as demo:
    gr.Markdown("# 🧠 AI Log Summariser")

    with gr.Row():
        drop = gr.Dropdown(choices=list(sample_logs), label="Sample log")
        load_btn = gr.Button("Load sample")

    #file_up = gr.File(label="Upload .log / .txt", file_types=["text", ".log"])
    file_up = gr.File(
        label="Upload .log / .txt",
        file_types=["text", ".log"],
        file_count="single",   # always a single file
        type="binary",         # send the raw bytes instead of a filepath
    )


    
    log_box = gr.Textbox(lines=15, label="Log input", placeholder="Paste log here")

    summarise_btn = gr.Button("Summarise")
    summary_box = gr.Textbox(label="Summary", lines=6)
    token_md = gr.Markdown()

    save_btn = gr.Button("Save summary")
    dl_link = gr.File(label="Download", visible=False)

    # wiring
    load_btn.click(load_sample, drop, log_box)
    #file_up.change(read_upload, file_up, log_box)
    file_up.upload(read_upload, file_up, log_box)
    summarise_btn.click(do_summarise, log_box, [summary_box, token_md])
    save_btn.click(save_to_file, summary_box, dl_link)

if __name__ == "__main__":
    demo.launch()
