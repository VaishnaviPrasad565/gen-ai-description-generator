
# AI-Driven Product Description Generator (Gemma-2B)

This project generates SEO-friendly, tone-controlled product descriptions for e-commerce using the Gemma-2B instruction-tuned model.

## Files

- `app.py` – Streamlit web app with:
  - Product selection / manual input
  - Tone + structure + length controls
  - Evaluation dashboard:
    1. Readability & tone alignment
    2. SEO metrics (keyword coverage & density)
    3. Variation under prompt changes
    4. Fluency / hallucination check
- `requirements.txt` – Python dependencies to install.

> **Note**: The dataset file `product_dataset.xlsx` is **not** included in this zip. Upload it separately to the same folder in Colab or on your local machine.

## How to use in Google Colab

1. Upload this zip file to Google Drive.
2. In Colab, mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Unzip the project (adjust the path to your zip file):
   ```python
   !unzip "/content/drive/MyDrive/ai_product_desc_project.zip" -d /content/
   ```
4. Upload your dataset `product_dataset.xlsx` into `/content/ai_product_desc_project/` (using Colab file browser or `files.upload()`).
5. Install requirements:
   ```python
   %cd /content/ai_product_desc_project
   !pip install -r requirements.txt
   ```
6. (Optional but recommended) Login to Hugging Face if Gemma-2B requires it:
   ```python
   from huggingface_hub import login
   login()  # paste your HF token
   ```
7. Run Streamlit:
   ```python
   !streamlit run app.py --server.port 8501 &
   ```
8. Expose via ngrok if in Colab:
   ```python
   from pyngrok import ngrok
   # ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
   public_url = ngrok.connect(8501)
   public_url
   ```

Click the URL printed to open the app.

## How to run locally (on your laptop)

1. Unzip the folder.
2. Place `product_dataset.xlsx` in the same folder.
3. Create a virtual environment (optional but recommended) and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```
5. Open the local URL shown in the terminal (usually `http://localhost:8501`).

Make sure your machine has enough RAM / GPU for `google/gemma-2b-it`. If it is too heavy, you can change `MODEL_ID` in `app.py` to a smaller model like `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
