# Deployment

Deploy from a pinned Python environment with Streamlit as the application entry point. Provide enough memory for decoded audio arrays and, when enabled, source-separation models.

Keep generated model artifacts outside the image when they change frequently, but verify their version and integrity at startup. Terminate TLS at the hosting layer.
