"""Training utilities and metrics for headline generation."""

import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from transformers import TrainingArguments
from typing import Dict, Any, Tuple, Optional


def compute_metrics(tokenizer):
    """Create a metrics computation function for text generation tasks.

    Args:
        tokenizer: Tokenizer for decoding predictions and labels

    Returns:
        Function that computes ROUGE and BLEU metrics
    """
    def _compute_metrics(eval_preds) -> Dict[str, float]:
        """Custom metrics computation function for text generation tasks."""
        predictions, labels = eval_preds

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Decode labels (ignore padding tokens)
        decoded_labels = tokenizer.batch_decode(
            labels.where(labels != -100, tokenizer.pad_token_id),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Initialize scorers
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rougeL_scores = []
        bleu_scores = []

        # Calculate scores for each prediction-label pair
        for pred, label in zip(decoded_preds, decoded_labels):
            # ROUGE scores
            scores = rouge.score(label, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

            # BLEU score
            label_tokens = label.split()
            pred_tokens = pred.split()
            bleu = sentence_bleu([label_tokens], pred_tokens)
            bleu_scores.append(bleu)

        return {
            'rouge1': np.mean(rouge1_scores),
            'rougeL': np.mean(rougeL_scores),
            'bleu': np.mean(bleu_scores)
        }

    return _compute_metrics


def create_training_arguments(
    output_dir: str = 'headline-generator-model',
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 128,
    learning_rate: float = 2.0e-05,
    fp16: bool = True,
    do_eval: bool = True,
    evaluation_strategy: str = "epoch",
    logging_steps: int = 5,
    save_strategy: str = "no",
    overwrite_output_dir: bool = True,
    gradient_checkpointing: bool = True,
    lr_scheduler_type: str = "cosine",
    seed: int = 42,
    **kwargs
) -> TrainingArguments:
    """Create training arguments for the SFT trainer.

    Args:
        output_dir: Directory to save model outputs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        fp16: Whether to use fp16 precision
        do_eval: Whether to run evaluation
        evaluation_strategy: When to run evaluation
        logging_steps: Number of steps between logging
        save_strategy: When to save checkpoints
        overwrite_output_dir: Whether to overwrite output directory
        gradient_checkpointing: Whether to use gradient checkpointing
        lr_scheduler_type: Type of learning rate scheduler
        seed: Random seed
        **kwargs: Additional arguments

    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        fp16=fp16,
        do_eval=do_eval,
        evaluation_strategy=evaluation_strategy,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        log_level="info",
        logging_steps=logging_steps,
        logging_strategy="steps",
        lr_scheduler_type=lr_scheduler_type,
        max_steps=-1,
        num_train_epochs=num_train_epochs,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        save_strategy=save_strategy,
        save_total_limit=None,
        seed=seed,
        **kwargs
    )


def save_training_metrics(trainer, output_dir: str, train_dataset, training_args):
    """Save training metrics and state.

    Args:
        trainer: The SFT trainer object
        output_dir: Directory to save metrics
        train_dataset: Training dataset
        training_args: Training arguments used
    """
    train_result = trainer.train()
    metrics = train_result.metrics

    max_train_samples = (
        training_args.max_train_samples
        if hasattr(training_args, 'max_train_samples') and training_args.max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    return train_result, metrics


def push_to_hub(model, tokenizer, hub_model_name: str, commit_message: Optional[str] = None):
    """Push trained model and tokenizer to Hugging Face Hub.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        hub_model_name: Name for the model on the hub
        commit_message: Optional commit message
    """
    if commit_message is None:
        commit_message = f"Upload {hub_model_name}"

    model.push_to_hub(hub_model_name, commit_message=commit_message)
    tokenizer.push_to_hub(hub_model_name, commit_message=commit_message)

    print(f"Model and tokenizer pushed to hub as: {hub_model_name}")


def generate_headlines(model, tokenizer, articles, max_length: int = 128, num_beams: int = 5):
    """Generate headlines from articles using a trained model.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        articles: List of article texts
        max_length: Maximum length for generation
        num_beams: Number of beams for beam search

    Returns:
        List of generated headlines
    """
    import torch

    # Ensure articles is a list
    if isinstance(articles, str):
        articles = [articles]

    inputs = tokenizer(
        articles,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams
        )

    headlines = [
        tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for output in outputs
    ]

    return headlines