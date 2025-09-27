"""Model configuration and setup utilities for headline generation training."""

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig
from typing import Tuple, Dict, Any, Optional


def create_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bfloat16"
) -> BitsAndBytesConfig:
    """Create quantization configuration for memory-efficient training.

    Args:
        load_in_4bit: Whether to load model in 4-bit precision
        bnb_4bit_quant_type: Quantization type for 4-bit
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization

    Returns:
        BitsAndBytesConfig object
    """
    # Fix the dtype reference
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype) if isinstance(bnb_4bit_compute_dtype, str) else bnb_4bit_compute_dtype

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def create_model_kwargs(
    quantization_config: Optional[BitsAndBytesConfig] = None,
    attn_implementation: str = "flash_attention_2",
    torch_dtype: str = "auto",
    use_cache: bool = False
) -> Dict[str, Any]:
    """Create model kwargs for model initialization.

    Args:
        quantization_config: Quantization configuration
        attn_implementation: Attention implementation to use
        torch_dtype: Torch dtype for model
        use_cache: Whether to use cache

    Returns:
        Dictionary of model kwargs
    """
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    return {
        "attn_implementation": attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": use_cache,
        "device_map": device_map,
        "quantization_config": quantization_config,
    }


def create_peft_config(
    r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    target_modules: Optional[list] = None
) -> LoraConfig:
    """Create PEFT (LoRA) configuration for parameter-efficient fine-tuning.

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout rate for LoRA layers
        bias: Bias configuration
        task_type: Type of task (CAUSAL_LM for generation)
        target_modules: List of modules to apply LoRA to

    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        target_modules=target_modules,
    )


def setup_model_and_tokenizer(
    model_id: str = "Helsinki-NLP/opus-mt-en-mul",
    use_quantization: bool = True,
    quantization_config: Optional[BitsAndBytesConfig] = None
) -> Tuple[AutoTokenizer, Dict[str, Any], LoraConfig]:
    """Setup tokenizer, model kwargs, and PEFT config.

    Args:
        model_id: Model identifier from Hugging Face
        use_quantization: Whether to use quantization
        quantization_config: Custom quantization config (if None, creates default)

    Returns:
        Tuple of (tokenizer, model_kwargs, peft_config)
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create quantization config if needed
    if use_quantization and quantization_config is None:
        quantization_config = create_quantization_config()

    # Create model kwargs
    model_kwargs = create_model_kwargs(
        quantization_config=quantization_config if use_quantization else None
    )

    # Create PEFT config
    peft_config = create_peft_config()

    return tokenizer, model_kwargs, peft_config


def load_trained_model(
    model_path: str,
    load_in_4bit: bool = True,
    device_map: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a trained model and tokenizer from a local path.

    Args:
        model_path: Path to the trained model
        load_in_4bit: Whether to load in 4-bit precision
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=load_in_4bit,
        device_map=device_map
    )

    return model, tokenizer