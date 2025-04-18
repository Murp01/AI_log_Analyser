"""
Log‑Summariser — pure TensorFlow
────────────────────────────────
✓ No PyTorch, no pipeline(), no from_tf flag
✓ Gradio UI: sample log, file upload, token counter, download button
"""

from __future__ import annotations
import tempfile
from datetime import datetime
from typing import Tuple, Optional

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
[INFO] Test run complete"""
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

    cleaned = preprocess(raw_log)[:1024]                          # hard cap for demo
    enc = tokenizer(cleaned, return_tensors="tf", truncation=True, max_length=1024)
    n_in = int(enc["input_ids"].shape[1])

    out_ids = model.generate(enc["input_ids"], max_length=80, min_length=20,
                             do_sample=False, num_beams=4)
    summary = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    return summary, f"**Input tokens:** {n_in}"


def save_to_file(text: str) -> Optional[str]:
    if not text.strip():
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt",
                                     mode="w", encoding="utf‑8") as fp:
        fp.write(text)
        return fp.name


def load_sample(name: str) -> str:
    return sample_logs.get(name, "")


def read_upload(file_obj) -> str:
    return file_obj.read().decode("utf‑8", errors="ignore") if file_obj else ""


# ── 4. Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="🧠 Log Summariser") as demo:
    gr.Markdown("# 🧠 AI Log Summariser")

    with gr.Row():
        drop = gr.Dropdown(choices=list(sample_logs), label="Sample log")
        load_btn = gr.Button("Load sample")

    file_up = gr.File(label="Upload .log / .txt")
    log_box = gr.Textbox(lines=15, label="Log input", placeholder="Paste log here")

    summarise_btn = gr.Button("Summarise")
    summary_box = gr.Textbox(label="Summary", lines=6)
    token_md = gr.Markdown()

    save_btn = gr.Button("Save summary")
    dl_link = gr.File(label="Download", visible=False)

    # wiring
    load_btn.click(load_sample, drop, log_box)
    file_up.change(read_upload, file_up, log_box)
    summarise_btn.click(do_summarise, log_box, [summary_box, token_md])
    save_btn.click(save_to_file, summary_box, dl_link)

if __name__ == "__main__":
    demo.launch()
