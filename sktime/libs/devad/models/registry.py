from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .AnomalyTransformer import AnomalyTransformer
from .Base import REQUIRED, BaseConfig, BaseModel
from .BeatGAN import BeatGAN
from .COUTA import COUTA
from .DAGMM import DAGMM
from .Donut import Donut
from .FCVAE import FCVAE
from .FITS import FITS
from .IForest import IForest
from .KANAD import KANAD
from .KMeansAD import KMeansAD
from .LSTMAD import LSTMAD
from .ModernTCN import ModernTCN
from .OmniAnomaly import OmniAnomaly
from .SubPCA import SubPCA
from .SubOCSVM import SubOCSVM
from .SubLOF import SubLOF
from .TimesNet import TimesNet
from .TranAD import TranAD
from .USAD import USAD


class ModelRegistry:
    _MODEL_CLASSES: dict[str, type[BaseModel]] = {
        "couta": COUTA,
        "donut": Donut,
        "lstm_ad": LSTMAD,
        "modern_tcn": ModernTCN,
        "fcvae": FCVAE,
        "fits": FITS,
        "iforest": IForest,
        "kan_ad": KANAD,
        "kmeans_ad": KMeansAD,
        "sub_pca": SubPCA,
        "sub_ocsvm": SubOCSVM,
        "sub_lof": SubLOF,
        "timesnet": TimesNet,
        "usad": USAD,
        "omni_anomaly": OmniAnomaly,
        "tranad": TranAD,
        "beatgan": BeatGAN,
        "anomaly_transformer": AnomalyTransformer,
        "dagmm": DAGMM,
    }

    @classmethod
    def list_families(cls) -> list[str]:
        return sorted(cls._MODEL_CLASSES)

    @classmethod
    def get_model_class(cls, family: str) -> type[BaseModel]:
        try:
            return cls._MODEL_CLASSES[family]
        except KeyError as exc:
            raise KeyError(f"Unknown model family: {family!r}") from exc

    @classmethod
    def get_model_info(cls, family: str) -> dict[str, Any]:
        model_class = cls.get_model_class(family)
        params = {}

        for name, default in model_class.HP.items():
            param_info = {"required": default is REQUIRED}
            if default is not REQUIRED:
                param_info["default"] = deepcopy(default)
            params[name] = param_info

        return {
            "family": family,
            "params": params,
        }

    @classmethod
    def create_model(
        cls,
        family: str,
        params: dict,
        seed: int = 2026,
        device: str = "mps",
    ) -> BaseModel: # 模型分发表
        model_class = cls.get_model_class(family)
        base_config = BaseConfig(
            seed=int(seed),
            device=device,
            params=params,
        )
        return model_class(base_config)
