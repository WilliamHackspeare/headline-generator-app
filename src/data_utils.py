"""Data loading, cleaning, and preprocessing utilities for headline generation."""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


def clean_article(article: str) -> str:
    """Clean article text by removing metadata such as dates.

    Args:
        article: Raw article text

    Returns:
        Cleaned article text
    """
    return article.split(') ', 1)[-1] if ') ' in article else article


def load_dataset(train_split: str = "final_headline_train_12000.csv",
                val_split: str = "final_headline_valid_1200.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation datasets from Hugging Face.

    Args:
        train_split: Name of the training split file
        val_split: Name of the validation split file

    Returns:
        Tuple of (train_df, val_df)
    """
    splits = {'train': train_split, 'validation': val_split}

    train_df = pd.read_csv("hf://datasets/valurank/News_headlines/" + splits["train"])
    val_df = pd.read_csv("hf://datasets/valurank/News_headlines/" + splits["validation"])

    # Apply cleaning to both datasets
    train_df['article'] = train_df['article'].apply(clean_article)
    val_df['article'] = val_df['article'].apply(clean_article)

    # Create text field for SFTTrainer (article + headline format)
    train_df['text'] = train_df['article'] + " ### " + train_df['headline']
    val_df['text'] = val_df['article'] + " ### " + val_df['headline']

    return train_df, val_df


class HeadlineDataset(Dataset):
    """Custom dataset class for headline generation training."""

    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 256, target: bool = True):
        """Initialize the dataset.

        Args:
            dataframe: DataFrame containing 'article' and 'headline' columns
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            target: Whether to include target (headline) data
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target = target

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        """Get a single item from the dataset.

        Args:
            index: Index of the item to retrieve

        Returns:
            Dictionary containing tokenized inputs and optionally targets
        """
        row = self.dataframe.iloc[index]
        input_text = row['article']
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        if self.target:
            target_text = row['headline']
            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': targets['input_ids'].squeeze(),
                'target_text': target_text
            }
        else:
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze()
            }


def prepare_datasets(tokenizer, max_length: int = 256,
                    train_split: str = "final_headline_train_12000.csv",
                    val_split: str = "final_headline_valid_1200.csv") -> Tuple[HeadlineDataset, HeadlineDataset]:
    """Prepare training and validation datasets.

    Args:
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
        train_split: Name of the training split file
        val_split: Name of the validation split file

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_df, val_df = load_dataset(train_split, val_split)

    train_dataset = HeadlineDataset(train_df, tokenizer, max_length)
    val_dataset = HeadlineDataset(val_df, tokenizer, max_length)

    return train_dataset, val_dataset