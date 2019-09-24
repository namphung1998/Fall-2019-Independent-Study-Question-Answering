from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import string

class NumberDataset(Dataset):
    def __init__(self, low, high):
        self.samples = [x for x in range(low, high)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        n = self.samples[idx]
        successors = torch.arange(4).float() + n + 1
        noisy = torch.randn(4) + successors
        return n, n + 1

class VietnameseNameDataset(Dataset):
    def __init__(self, path):
        self.name_df = pd.read_csv(path, header=None)
        self.all_letters = string.ascii_letters + '#'
        self.char2idx = {c: i for i, c in enumerate(self.all_letters)}
        self.idx2char = {i: c for i, c in enumerate(self.all_letters)}
        self.one_hots = pd.get_dummies(list(self.all_letters))

    def __len__(self):
        return len(self.name_df)

    def __to_index_list(self, name):
        return [self.char2idx[c] for c in name]
    
    def __getitem__(self, i):
        name = self.name_df.iloc[i, 0]
        target_name = name[1:] + '#'

        return name, target_name

if __name__ == "__main__":
    dataloader = DataLoader(VietnameseNameDataset('data/Vietnamese.txt'), batch_size=16)

    for name, target_name in dataloader:
        print(name)
        print(target_name)
        break

    # dataloader = DataLoader(NumberDataset(0, 100), batch_size=16)
    # for n, other in dataloader:
    #     print(n)
    #     print(other)
    #     break