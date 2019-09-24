from torch.utils.data import Dataset, DataLoader
import torch
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class VietnameseNameDataset(Dataset):
    def __init__(self, path, eos_token='#'):
        self.name_df = pd.read_csv(path, header=None)

    def __len__(self):
        return len(self.name_df)
    
    def __getitem__(self, i):
        name = self.name_df.iloc[i, 0]
        target_name = name[1:] + '#'

        return name

def split_data(dataset, test_split=0.1, shuffle=True):
    train, test = train_test_split(dataset, test_size=test_split, shuffle=shuffle)

    return train, test

def batch_iter(data, batch_size, eos_token='#', shuffle=True):
    num_batch = len(data) // batch_size
    indices = list(range(len(data)))

    if shuffle:
        np.random.shuffle(data)
    
    for i in range(num_batch):
        batch_indices = indices[i * batch_size : (i+1) * batch_size]
        names = [data[idx] for idx in batch_indices]
        names.sort(key=lambda x: len(x), reverse=True)

        target_names = [x[1:] + eos_token for x in names]

        yield names, target_names


if __name__ == "__main__":

    dataset = VietnameseNameDataset('data/Vietnamese.txt')
    train, test = split_data(dataset)

    iterator = batch_iter(train, 16)
    names, target_names = next(iterator)

