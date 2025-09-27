"""Headline generation training package."""

from .data_utils import load_dataset, clean_article, HeadlineDataset, prepare_datasets
from .model_config import (
    create_quantization_config,
    create_model_kwargs,
    create_peft_config,
    setup_model_and_tokenizer,
    load_trained_model
)
from .training_utils import (
    compute_metrics,
    create_training_arguments,
    save_training_metrics,
    push_to_hub,
    generate_headlines
)
from .train_headline_model import train_headline_model, quick_train

__all__ = [
    # Data utilities
    "load_dataset",
    "clean_article",
    "HeadlineDataset",
    "prepare_datasets",

    # Model configuration
    "create_quantization_config",
    "create_model_kwargs",
    "create_peft_config",
    "setup_model_and_tokenizer",
    "load_trained_model",

    # Training utilities
    "compute_metrics",
    "create_training_arguments",
    "save_training_metrics",
    "push_to_hub",
    "generate_headlines",

    # Main training functions
    "train_headline_model",
    "quick_train"
]