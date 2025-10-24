__package__ = "__data__"

from torch.utils.data import Dataset, DataLoader
import torch

class TranslationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer # I use Xlm-roberta tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_text = self.df.iloc[idx, 'en']
        tgt_text = self.df.iloc[idx, 'vn']
    
        # Tokenize source text
        src_tokenized = self.tokenizer(src_text, truncation=True, max_length=self.max_len)
        
        # Tokenize target text
        tgt_tokenized = self.tokenizer(tgt_text, truncation=True, max_length=self.max_len)
        
        return {
            "input_ids": torch.tensor(src_tokenized['input_ids']),
            "labels": torch.tensor(tgt_tokenized['input_ids']),
        }
    
import torch
from torch.nn.utils.rnn import pad_sequence

def create_collate_fn(tokenizer):
    """
    Creates a collate function that pads batches to the longest sequence.
    This is a factory function so the collate_fn has access to the tokenizer.
    """
    def collate_fn(batch):
        """
        Custom collate function to handle padding for each batch dynamically.
        """
        # --- Separate items from the batch ---
        input_ids_list = [item['input_ids'] for item in batch]
        labels_list = [item['labels'] for item in batch]

        # --- Pad sequences ---
        # `pad_sequence` stacks tensors and pads them to the longest tensor in the list.
        # `batch_first=True` means the output shape will be (batch_size, sequence_length).
        
        # Pad input_ids. Use the tokenizer's pad_token_id for padding.
        input_ids_padded = pad_sequence(
            input_ids_list, 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        
        # Pad labels. Use -100 for padding in labels so they are ignored in loss calculation.
        labels_padded = pad_sequence(
            labels_list, 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        
        # The decoder_input_ids are created from the labels by the model automatically,
        # so we just need to provide the padded labels.

        return {
            'input_ids': input_ids_padded,
            'labels': labels_padded,
        }
    return collate_fn
