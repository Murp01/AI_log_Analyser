# Log Analyzer using Hugging Face Transformers

## What It Does
This tool reads a test log file and summarizes it using a Hugging Face pre-trained model (`facebook/bart-large-cnn`).

## How to Run

1. Create virtual environment:
    ```
    python -m venv venv
    ```

2. Activate it:
    - Windows: `venv\Scripts\activate`
    - macOS/Linux: `source hf_env/bin/activate`

3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Run the log analyzer (command line):
    ```
    python app.py
    ```

5. Or launch the GUI:
    ```
    python gui.py
    ```

## Sample Log
See `logs/test_log.txt` for an example.
