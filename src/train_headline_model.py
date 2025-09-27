"""Main training orchestration for headline generation model."""

from trl import SFTTrainer
from typing import Optional, Dict, Any, Tuple
import os

from .data_utils import prepare_datasets
from .model_config import setup_model_and_tokenizer, create_quantization_config, create_peft_config
from .training_utils import (
    compute_metrics,
    create_training_arguments,
    save_training_metrics,
    push_to_hub
)


def train_headline_model(
    model_id: str = "Helsinki-NLP/opus-mt-en-mul",
    output_dir: str = "headline-generator-model",
    hub_model_name: Optional[str] = None,

    # Data parameters
    max_length: int = 256,
    train_split: str = "final_headline_train_12000.csv",
    val_split: str = "final_headline_valid_1200.csv",

    # Training parameters
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 128,
    learning_rate: float = 2.0e-05,

    # Model parameters
    use_quantization: bool = True,
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bfloat16",

    # LoRA parameters
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[list] = None,

    # Training configuration
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    lr_scheduler_type: str = "cosine",
    evaluation_strategy: str = "epoch",
    logging_steps: int = 5,
    save_strategy: str = "no",
    seed: int = 42,

    # Hub upload
    push_to_hub_after_training: bool = False,

    # Additional arguments
    **kwargs
) -> Tuple[SFTTrainer, Dict[str, Any]]:
    """Train a headline generation model with configurable parameters.

    Args:
        model_id: Base model identifier from Hugging Face
        output_dir: Directory to save the trained model
        hub_model_name: Name for uploading to Hugging Face Hub

        max_length: Maximum sequence length for tokenization
        train_split: Training data split filename
        val_split: Validation data split filename

        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate

        use_quantization: Whether to use quantization
        load_in_4bit: Whether to load model in 4-bit
        bnb_4bit_quant_type: Quantization type
        bnb_4bit_compute_dtype: Compute dtype for quantization

        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout rate
        lora_target_modules: Target modules for LoRA

        fp16: Whether to use fp16 training
        gradient_checkpointing: Whether to use gradient checkpointing
        lr_scheduler_type: Learning rate scheduler type
        evaluation_strategy: Evaluation strategy
        logging_steps: Logging steps
        save_strategy: Save strategy
        seed: Random seed

        push_to_hub_after_training: Whether to push to hub after training
        **kwargs: Additional arguments for TrainingArguments

    Returns:
        Tuple of (trainer, metrics)
    """
    print(f"Starting headline model training with model: {model_id}")
    print(f"Output directory: {output_dir}")

    # Setup model, tokenizer, and configurations
    print("Setting up model and tokenizer...")

    # Create quantization config if needed
    quantization_config = None
    if use_quantization:
        quantization_config = create_quantization_config(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
        )

    tokenizer, model_kwargs, _ = setup_model_and_tokenizer(
        model_id=model_id,
        use_quantization=use_quantization,
        quantization_config=quantization_config
    )

    # Create PEFT config
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    peft_config = create_peft_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules
    )

    # Prepare datasets
    print("Preparing datasets...")
    from .data_utils import load_dataset
    train_df, val_df = load_dataset(train_split, val_split)

    print(f"Training dataset size: {len(train_df)}")
    print(f"Validation dataset size: {len(val_df)}")

    # Create training arguments
    training_args = create_training_arguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy=evaluation_strategy,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        seed=seed,
        **kwargs
    )

    # Create compute metrics function
    metrics_fn = compute_metrics(tokenizer)

    # Initialize trainer
    print("Initializing SFT trainer...")
    trainer = SFTTrainer(
        model=model_id,
        model_init_kwargs=model_kwargs,
        args=training_args,
        compute_metrics=metrics_fn,
        train_dataset=train_df,
        eval_dataset=val_df,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
        max_seq_length=max_length,
    )

    # Start training
    print("Starting training...")
    train_result, metrics = save_training_metrics(
        trainer=trainer,
        output_dir=output_dir,
        train_dataset=train_df,
        training_args=training_args
    )

    print("Training completed!")
    print(f"Final metrics: {metrics}")

    # Push to hub if requested
    if push_to_hub_after_training and hub_model_name:
        print(f"Pushing model to hub as: {hub_model_name}")
        # Load the trained model
        from .model_config import load_trained_model
        model, tokenizer = load_trained_model(output_dir)
        push_to_hub(model, tokenizer, hub_model_name)

    return trainer, metrics


def quick_train(
    output_dir: str = "headline-generator-quick",
    num_train_epochs: int = 1,
    gradient_accumulation_steps: int = 64,
    **kwargs
) -> Tuple[SFTTrainer, Dict[str, Any]]:
    """Quick training run with reduced parameters for testing.

    Args:
        output_dir: Output directory
        num_train_epochs: Number of epochs (default 1 for quick training)
        gradient_accumulation_steps: Reduced accumulation steps
        **kwargs: Additional arguments passed to train_headline_model

    Returns:
        Tuple of (trainer, metrics)
    """
    return train_headline_model(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    trainer, metrics = train_headline_model(
        output_dir="headline-generator-experiment",
        num_train_epochs=1,
        hub_model_name="my-headline-generator",
        push_to_hub_after_training=False
    )

    print("Training completed successfully!")
    print(f"Metrics: {metrics}")