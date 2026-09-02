# Shaw Gemini PDF Search

A Streamlit interface for uploading reference PDFs and a UTF-8 text file of
search queries. The app runs each query against each PDF with Gemini File
Search, applies the sustainability persona, and provides results, brand-rank
charts, appearance percentages, and an Excel download.

## Run locally

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Add your Google API key to the new secrets file.
3. Install and run:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The model is fixed to `gemini-3.1-flash-lite`. The bundled
`twin_imp_columns.csv` is used by default; users may upload a CSV or XLSX
replacement in the app.

## Deploy on Streamlit Community Cloud

1. Push this directory to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `streamlit_app.py`.
4. In **Advanced settings → Secrets**, enter:

```toml
GOOGLE_API_KEY = "your-google-api-key"
```

Never commit `.streamlit/secrets.toml` or an API key to GitHub.
