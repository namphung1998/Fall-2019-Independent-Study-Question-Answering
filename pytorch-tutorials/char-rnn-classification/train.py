import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import tqdm

from model import GRUClassifier
from preprocessing import *

def read_corpus(categories):
    data = []
    for i, c in enumerate(categories):
        filename = f'data/names/{c}.txt'
        lines = readLines(filename)
        data.extend([(l, i) for l in lines])
    return data

def load_data(categories, test_split=0.1):
    data = read_corpus(categories)
    np.random.shuffle(data)
    n_train = int((1 - test_split) * len(data))

    X_train = [x for x, _ in data[:n_train]]
    y_train = [x for _, x in data[:n_train]]

    X_test = [x for x, _ in data[n_train:]]
    y_test = [x for _, x in data[n_train:]]

    return X_train, y_train, X_test, y_test


def pad_names(names, pad_token):
    res = []
    lengths = []
    max_length = 0
    for n in names:
        max_length = max(max_length, len(n))
        lengths.append(len(n))

    for n in names:
        if len(n) < max_length:
            res.append(n + pad_token * (max_length - len(n)))
        else:
            res.append(n)
    return res, lengths


def batch_iter(X_train, y_train, batch_size, shuffle=False):
    num_batch = math.ceil(len(X_train) / batch_size)
    indices = list(range(len(X_train)))

    if shuffle:
        np.random.shuffle(indices)

    for i in range(num_batch):
        batch_indices = indices[i * batch_size : (i + 1) * batch_size]
        examples = [(X_train[idx], y_train[idx]) for idx in batch_indices]
        examples.sort(key=lambda x: len(x[0]), reverse=True)

        names = [x[0] for x in examples]
        categories = [x[1] for x in examples]
        names, lengths = pad_names(names, '#')

        yield names, torch.tensor(categories), lengths

if __name__ == "__main__":    

    model = GRUClassifier(n_letters, 32, len(all_categories))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    num_epochs = 20

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        running_loss = 0.0
        for names, categories, lengths in batch_iter(X_train, y_train, 16, shuffle=True):
            input_padded = batch_to_tensor(names, lengths[0])
            optimizer.zero_grad()
            outputs = model(input_padded, lengths)
            loss = criterion(outputs, categories)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
        print(running_loss)

    # torch.save(model.state_dict(), 'models/test2.pt')


