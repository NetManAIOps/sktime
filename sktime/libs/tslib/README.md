# TSLib vendored subset

Vendored from [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library),
commit `4e938a1767106324dd753b2a44832bf870a0252e` (main). MIT License, see `LICENSE`.

Contents:

- `layers/` — the upstream `layers/` files required by the model subset
  (`Embed`, `SelfAttention_Family`, `AutoCorrelation`, `FourierCorrelation`,
  `MultiWaveletCorrelation`, `Transformer_EncDec`, `Autoformer_EncDec`,
  `Conv_Blocks`), with intra-package imports rewritten to relative imports.
  Layer files only needed by non-vendored models (Crossformer, ETSformer,
  Mamba, Pyraformer, ...) were dropped so the package stays importable
  without optional dependencies (einops, mamba_ssm, pywt, reformer_pytorch).
- `models/` — a curated subset of upstream models: `DLinear`, `TimesNet`,
  `iTransformer`, `Autoformer`, `FEDformer`, `FreTS`.
- `utils/masking.py`, `utils/timefeatures.py` — upstream utilities required by
  the layers and by the sktime adapter.

Local modifications (nothing else was changed):

- `from layers.X import ...` → relative imports (`from ..layers.X` in
  `models/`, `from .X` in `layers/`).
- `from utils.masking import ...` → `from ..utils.masking import ...`.

Consumers: `sktime.forecasting.tslib` (sktime forecaster adapters).
All vendored modules import torch at module level; import them lazily.
