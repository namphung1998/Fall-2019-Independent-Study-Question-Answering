import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader.data_loaders import VietnameseNameDataset, batch_iter, split_data
from data_loader.vocab import Vocab

from model.models import GRUGenerator

def train(data_path, num_epochs=2):
    dataset = VietnameseNameDataset(data_path)
    train_data, _ = split_data(dataset, test_split=0.1)
    vocab = Vocab()

    model = GRUGenerator(len(vocab), 32, 32, len(vocab), 0, vocab)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    losses = []

    for epoch in range(num_epochs):
        for names, target_names in batch_iter(train_data, 16):
            target = vocab.chars2indices(target_names)
            loss = 0.0
            optimizer.zero_grad()
            outputs = model(names)
            for i in range(outputs.shape[0]):
                y = outputs[i]
                loss += criterion(y[:len(target[i])], torch.LongTensor(target[i]))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    
    return losses

if __name__ == "__main__":
    losses = train('data_loader/data/Vietnamese.txt', num_epochs=1000)
    plt.plot(losses)
    plt.savefig('loss.png')