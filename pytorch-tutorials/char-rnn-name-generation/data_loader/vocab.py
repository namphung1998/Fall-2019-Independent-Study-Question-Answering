import torch
import string

class Vocab(object):
    def __init__(self, end_token='>', start_token='<', pad_token='#'):
        self.pad_token = pad_token
        self.char2idx = {c: i+3 for i, c in enumerate(string.ascii_letters)}
        self.char2idx[pad_token] = 0
        self.char2idx[start_token] = 1
        self.char2idx[end_token] = 2
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def __getitem__(self, char):
        return self.char2idx[char]

    def __contains__(self, char):
        return char in self.char2idx

    def __setitem__(self, key, value):
        raise ValueError('Vocab is read-only')

    def __len__(self):
        return len(self.char2idx)

    def chars2indices(self, words):
        if type(words) == list:
            return [[self[c] for c in word] for word in words]
        return [self[c] for c in words]

    def indices2chars(self, indices):
        return [self.idx2char[i] for i in indices]

    def pad_words(self, words):
        max_length = 0
        for word in words:
            max_length = max(max_length, len(word))

        padded_words = []
        for word in words:
            if len(word) < max_length:
                padded_words.append(word + [self[self.pad_token]] * (max_length - len(word)))
            else:
                padded_words.append(word)
        
        return padded_words


    def to_input_tensor(self, words):
        char_ids = self.chars2indices(words)
        padded_words = self.pad_words(char_ids)
        padded_words = torch.tensor(padded_words, dtype=torch.long)

        return padded_words

if __name__ == "__main__":
    vocab = Vocab()
    print(vocab.to_input_tensor(['hello']))

