import pandas as pd
from tokenizers import Tokenizer

import torch
from torch.utils.data import Dataset, DataLoader

tokenizer = Tokenizer.from_file("data/tokenizer.json")
SEQ_LEN = 512
tokenizer.enable_truncation(max_length=SEQ_LEN)
tokenizer.enable_padding(pad_id=1, length=SEQ_LEN+1)

class PretrainDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = pd.read_parquet(self.data_path)
        # preprocess
        self.df = self.df[self.df['language'] == 'en']
        self.df['text'] = self.df['text'].str.lower()
        self.df['text'] = self.df['text'].str.replace(r"[^a-z0-9 [:space:][:punct:]]", '', regex=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        # outputs seq_in, seq_out
        row = self.df.iloc[idx]
        output = tokenizer.encode(row['text'])

        seq_in, seq_out = output.ids[:-1], output.ids[1:]
        
        seq_in, seq_out = torch.tensor(seq_in), torch.tensor(seq_out)

        return seq_in, seq_out


if __name__ == '__main__':
    dataset = PretrainDataset(data_path='data/dev/012_00000.parquet')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for (seq_in, seq_out) in dataloader:
        print(seq_in)
