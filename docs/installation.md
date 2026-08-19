# Installation

Python 3.11 or 3.12 is recommended. Create an isolated environment, install `requirements.txt`, then verify the source compiles:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m compileall -q app.py src
```

Stem separation may also require FFmpeg to decode some audio formats.
