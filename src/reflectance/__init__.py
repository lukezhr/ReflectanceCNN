"""
Thin Film Characterization Package

This package provides theoretical and computational tools for thin film optical analysis,
including classes for representing thin film layers and modules for optical models.
"""

__version__ = "1.0.0"

__all__ = [
    "ThinFilmClasses",
    "TaucLorentz",
    "model_eval_pkg",
    "cnn_structure",
    "metrics",
    "helper",
    "evaluation",
    "plotting",
    "data_processor"
]

# Import the submodules to expose them as part of the package
from . import ThinFilmClasses, TaucLorentz, model_eval_pkg

# Shortcut imports for common functionality
from .ThinFilmClasses import ThinFilmLayer, ThinFilmLayerTL, ThinFilmSystem
from .TaucLorentz import TL_nk, TL_nk_multi
from .model_eval_pkg import optimize_TL, construct_bi2o3_multilayer, predict, data_prep, denormalize
from .cnn_structure import ReflectanceCNN
from .metrics import mse, rmse, adjusted_r2, mase, se, ae, pe, ne
from .helper import params_ranges
from .evaluation import ModelEvaluationResults
from .plotting import Plotter
from .data_processor import process_single_data, process_dataset