# Performance

Audio duration, sample rate, plot resolution, and stem model size are the main performance drivers. Cache immutable analysis results at the UI boundary and avoid recomputing transforms when only presentation state changes.

Profile with representative full-length material. Short synthetic clips are useful for tests but do not expose production memory requirements.
