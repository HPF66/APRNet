# Teacher outputs

This is an optional output directory. Actual destinations are set by the entry points:

- CAnYA pretraining: `--output-dir` selects the run directory.
- CPAD cross-validation: `--output-root` in the prepared configuration selects the directory containing fold results and `config.json`.
- Full-data teacher fitting: `--output` selects the checkpoint file.
- Peptide prediction: `--output` selects the prediction file.

See the [teacher README](../README.md) for examples. Generated run results are not bundled.
