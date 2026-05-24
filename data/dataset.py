import pandas as pd
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

import torch
from torch.utils.data import Dataset, DataLoader

tokenizer = Tokenizer.from_file("data/tokenizer.json")
BOS_TOKEN = 2
SEQ_LEN = 512
tokenizer.enable_truncation(max_length=SEQ_LEN)
tokenizer.enable_padding(pad_id=1, length=SEQ_LEN+1)
# tokenizer.post_processor = TemplateProcessing(
#     single="[BOS] $A [EOS]",
#     special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
# )

class PretrainDataset(Dataset):
    def __init__(self, data_path: str, do_preprocess: bool = False):
        self.data_path = data_path
        self.df = pd.read_parquet(self.data_path)

        self.do_preprocess = do_preprocess
        if do_preprocess:
            # preprocess
            punctuation_map = {
                0x201C: 0x22,  # LEFT DOUBLE QUOTATION MARK -> "
                0x201D: 0x22,  # RIGHT DOUBLE QUOTATION MARK -> "
                0x2018: 0x27,  # LEFT SINGLE QUOTATION MARK -> '
                0x2019: 0x27,  # RIGHT SINGLE QUOTATION MARK -> '
            }
            self.df = self.df[self.df['language'] == 'en']
            self.df['text'] = self.df['text'].str.translate(punctuation_map)
            self.df['text'] = self.df['text'].str.lower()
            self.df['text'] = self.df['text'].str.replace(r"[^a-z0-9 [:space:][:punct:]]", '', regex=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        # outputs seq_in, seq_out
        row = self.df.iloc[idx]

        # output = tokenizer.encode(row['text'])

        seq_out = row.tolist()

        seq_in = seq_out.copy()
        seq_in.pop()
        seq_in = [BOS_TOKEN] + seq_in
        
        seq_in, seq_out = torch.tensor(seq_in, dtype=torch.long), torch.tensor(seq_out, dtype=torch.long)

        return seq_in, seq_out




if __name__ == '__main__':
    dataset = PretrainDataset(data_path='data/train/000_00000.parquet')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for (seq_in, seq_out) in dataloader:
        print(f'seq_in.shape: {seq_in.shape}')
        print(f'seq_out.shape: {seq_out.shape}')
        input('Hold...')
