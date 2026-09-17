# DevAD (vendored)

Vendored copy of the DevAD anomaly-detection model zoo and CLI, imported from
the `tinyfire27_devad_cli` branch of this fork (author: Tinyfire27), commit
`1a50d575` ("Import DevAD CLI source"), relocated from `devad_import/src/DevAD`
to `sktime/libs/devad` so it is importable as a sktime subpackage.

Contents:

- `models/` — 17 registered families (windowed torch models: BeatGAN, COUTA,
  Donut, FCVAE, FITS, KAN-AD, LSTM-AD, ModernTCN, OmniAnomaly, TimesNet,
  TranAD, USAD; subsequence scikit-learn models: IForest, KMeansAD, SubLOF,
  SubOCSVM, SubPCA) behind a uniform `fit(x_train)` / `detect(x_test)` API
  (`models/Base.py`, `models/registry.py`). Individual model files note their
  upstream source and license in the module docstring
  (e.g. FITS: Apache-2.0; `utils/affiliation/` has its own MIT LICENSE).
- `services/model_service.py` — persistent train/detect/evaluate API:
  `train_model` writes `model.pt` + `manifest.json` + `training.log` under
  `model_root/model_id/`; `load_model` / `model_detect` / `model_evaluate`
  consume that directory.
- `cli/` — typer CLI (`python -m sktime.libs.devad.cli.app model ...`).
- `probes/` — pseudo-anomaly probe utilities.

Entry points inside sktime:

- `sktime/detection/adapters/devad.py` — registry-discoverable detectors.
- `playground/trainer.py` — labts `train` / `detect` commands on top of
  `services/model_service.py`.
