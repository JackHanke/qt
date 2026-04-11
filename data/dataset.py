import pandas as pd
from torch.utils.data import Dataset, DataLoader


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
        row = self.df.iloc[idx]

        text = row['text']
        print(text.strip().lower())
        print(f'length: {len(text.split())}')

        



        # lower

        # tokenize

        # 
        pass


if __name__ == '__main__':
    dataset = PretrainDataset(data_path='data/dev/012_00000.parquet')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for (seq_in, seq_out) in dataloader:
        print(seq_in)
