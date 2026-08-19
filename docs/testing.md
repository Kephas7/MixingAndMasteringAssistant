# Testing

The project uses Python's built-in test framework so the suite runs without an extra test dependency:

```powershell
python -m unittest discover -s tests -v
```

Tests should use generated signals rather than committed audio fixtures. Keep assertions tolerant of small floating-point differences.
