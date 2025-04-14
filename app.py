from transformers import pipeline

# Load a pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def load_log_file(path):
    with open(path, 'r') as f:
        return f.read()

def analyze_log(file_path):
    log_text = load_log_file(file_path)
    # Limit input length if needed
    log_text = log_text[:1024]  # model max input
    summary = summarizer(log_text, max_length=60, min_length=20, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    path = "logs/test_log.txt"
    print("Summarized log:\n", analyze_log(path))
