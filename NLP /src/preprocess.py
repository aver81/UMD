
import re
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

class IMDBCsvDataset(Dataset):
    def __init__(self, df, vocab, tokenizer, seq_len=50):
        self.texts = df["review"].tolist()
        self.labels = df["sentiment"].map(lambda x: 1 if str(x).lower().strip() == "positive" else 0).tolist()
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(self.texts[idx])
        tokens = self.tokenizer(text)
        ids = self.vocab(tokens)[: self.seq_len]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return torch.tensor(ids, dtype=torch.long), label

def yield_tokens(texts, tokenizer):
    for text in texts:
        yield tokenizer(clean_text(text))

def load_data(csv_path, max_vocab_size=10000, seq_len=50, batch_size=32, test_split=0.5):
    """
    Load IMDb dataset from a local CSV with columns: review, sentiment
    Returns train_loader, test_loader, vocab_size
    """

    tokenizer = get_tokenizer("basic_english")

   
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)

   
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_split))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    
    vocab = build_vocab_from_iterator(
        yield_tokens(train_df["review"], tokenizer),
        specials=["<unk>"],
        max_tokens=max_vocab_size,
    )
    vocab.set_default_index(vocab["<unk>"])

    
    train_data = IMDBCsvDataset(train_df, vocab, tokenizer, seq_len)
    test_data = IMDBCsvDataset(test_df, vocab, tokenizer, seq_len)

    
    def collate_batch(batch):
        text_list, label_list = zip(*batch)
        text_padded = pad_sequence(text_list, batch_first=True, padding_value=0)
        labels = torch.stack(label_list)
        return text_padded, labels

    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader, len(vocab)
