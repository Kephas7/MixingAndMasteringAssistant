# Troubleshooting

If an upload cannot be decoded, convert it to PCM WAV and retry. If model loading fails, confirm that the expected Joblib artifact exists under `models/`; the application should otherwise use its analytical fallback.

For separation failures, verify Demucs and FFmpeg independently. For memory pressure, shorten the input or process stems outside the interactive session.
