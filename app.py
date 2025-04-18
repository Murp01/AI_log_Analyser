"""
Log‑Summariser — pure TensorFlow + Gradio 5
───────────────────────────────────────────
 • Summarise long logs with a fine‑tuned BART model
 • Works on CPU only (no PyTorch)
 • Supports sample logs, file upload, token counter, save‑to‑txt
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Union
import tempfile

import gradio as gr
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

MODEL_ID = "VidhuMathur/bart-log-summarization"

# ── 1 · Load model & tokenizer (once) ────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, from_pt=False)

# ── 2 · Sample logs for quick demos ─────────────────────────────────────────
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

# ── 3 · Helpers ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Strip uninteresting lines (keep ERROR / FAIL / WARNING)."""
    focus = ("ERROR", "FAIL", "WARNING")
    lines = [ln for ln in text.splitlines() if any(k in ln.upper() for k in focus)]
    return "\n".join(lines) or text


def do_summarise(raw_log: str) -> Tuple[str, str]:
    """Return (summary, token‑count string)."""
    if not raw_log.strip():
        return "", "⚠️ No log text."

    cleaned = preprocess(raw_log)[:1024]  # tiny demo cap
    enc = tokenizer(cleaned, return_tensors="tf", truncation=True, max_length=1024)
    token_in = int(enc["input_ids"].shape[1])

    out_ids = model.generate(
        enc["input_ids"], max_length=80, min_length=20, num_beams=4, do_sample=False
    )
    summary = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return summary, f"**Input tokens:** {token_in}"


def save_to_file(text: str) -> Optional[str]:
    """Write summary to a tmp file and return its path for download."""
    if not text.strip():
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf‑8") as fp:
        fp.write(text)
        return fp.name


def load_sample(name: str) -> str:
    return sample_logs.get(name, "")


def read_upload(file_obj: Union[bytes, str, Path, list, dict]) -> str:
    """Decode whatever Gradio passes for an uploaded file."""
    if not file_obj:
        return ""

    # Spaces (Gradio 5) wraps single upload in a list
    if isinstance(file_obj, list):
        file_obj = file_obj[0]

    # 1 · Raw bytes (type="binary")
    if isinstance(file_obj, (bytes, bytearray)):
        return file_obj.decode("utf‑8", errors="ignore")

    # 2 · Path‑like
    if isinstance(file_obj, (str, Path)):
        try:
            return Path(file_obj).read_text(encoding="utf‑8", errors="ignore")
        except Exception:
            return "⚠️ Could not read file."

    # 3 · File‑like object (older local Gradio)
    if hasattr(file_obj, "read"):
        try:
            return file_obj.read().decode("utf‑8", errors="ignore")
        except Exception:
            return "⚠️ Could not read stream."

    # 4 · FileData dict (edge case)
    if isinstance(file_obj, dict) and "data" in file_obj:
        import base64
        try:
            return base64.b64decode(file_obj["data"]).decode("utf‑8", errors="ignore")
        except Exception:
            return "⚠️ Could not decode FileData."

    return "⚠️ Unsupported upload type."

# ── 4 · Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Log Summariser")

    with gr.Row():
        drop = gr.Dropdown(choices=list(sample_logs), label="Sample log")
        load_btn = gr.Button("Load sample")

    file_up = gr.File(
        label="Upload .log / .txt",
        file_count="single",   # ensure only one file
        type="binary",         # send raw bytes (works in Spaces)
    )
    log_box = gr.Textbox(lines=15, label="Log input", placeholder="Paste log here")

    summarise_btn = gr.Button("Summarise")
    summary_box = gr.Textbox(label="Summary", lines=6)
    token_md = gr.Markdown()

    save_btn = gr.Button("Save summary")
    dl_link = gr.File(label="Download", visible=False)

    # Wire events
    load_btn.click(load_sample, drop, log_box)
    file_up.change(read_upload, file_up, log_box)
    summarise_btn.click(do_summarise, log_box, [summary_box, token_md])
    save_btn.click(save_to_file, summary_box, dl_link)

# ── 5 · Run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()
