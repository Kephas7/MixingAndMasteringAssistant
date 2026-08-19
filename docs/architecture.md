# Architecture

`app.py` coordinates the Streamlit interface. Modules in `src/` isolate audio loading, feature extraction, EQ analysis, compression analysis, mastering checks, stem separation, model training, and evaluation.

Runtime inference falls back to deterministic rules when trained artifacts are unavailable. This keeps the application usable on fresh installations while allowing trained models to refine recommendations.
