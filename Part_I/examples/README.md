# Examples

Run these examples from the `Part_I/` directory after installing
`requirements.txt`.

```bash
python examples/01_classical_eigenvalue.py
python examples/02_query_neural_atlas.py
```

`01_classical_eigenvalue.py` runs one real Riccati-shooting reference point.
`02_query_neural_atlas.py` uses the production routing function and public
`N340` routing table without loading a neural checkpoint. It reports the
checkpoint path that would be used by the existing evaluation code.

No compact, reliable public single-command interface currently exposes a
selected production GEP eigenpair, so this directory deliberately contains no
synthetic GEP example.
