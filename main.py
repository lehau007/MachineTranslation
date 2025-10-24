import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from collections import Counter
import re
from tqdm import tqdm
import numpy as np

from data.dataset import TranslationDataset, create_collate_fn
from model.machine_translation import MachineTranslation
from train.train import train_loop

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

if __name__ == "__main __":
    train_df = pd.read_csv("")
    # Tokenizer of XLM Roberta
    tokenizer_name="xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create dataset and dataloader
    train_dataset = TranslationDataset(train_df, tokenizer, max_len=32)
    my_collate_fn = create_collate_fn(tokenizer)

    # The collate_fn will be applied to each batch of data.
    data_loader = DataLoader(train_dataset, batch_size=16, collate_fn=my_collate_fn)

    model = MachineTranslation(
        vocab_size=tokenizer.vocab_size,
        d_model=768, # Smaller model for quick example
        num_heads=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        device='cpu'
    ).to('cpu')

    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=1)  # Ignore padding tokens

    print("Start training...")
    train_loop(model, data_loader, optimizer, criterion, device='cpu', epochs=3)
    print("Train OK!")

    # Test and evaluate
    

    # Save
    model.save("translation_model.pth")