# Security

Treat uploaded media as untrusted input. Limit upload size, keep decoders and scientific dependencies patched, and run deployments with restricted filesystem permissions.

Never load model artifacts from untrusted sources: Joblib files may execute code during deserialization. Train or obtain artifacts through a controlled pipeline.
