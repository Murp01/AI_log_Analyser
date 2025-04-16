import gradio as gr
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_log(log_text):
    trimmed = log_text[:1024]
    summary = summarizer(trimmed, max_length=60, min_length=20, do_sample=False)
    return summary[0]['summary_text']

iface = gr.Interface(
    fn=summarize_log,
    inputs=gr.Textbox(lines=15, placeholder="Paste your log here..."),
    outputs="text",
    title="Log Analyzer",
    description="Paste test logs and get a quick summary of key events or errors."
)

if __name__ == "__main__":
    iface.launch()
