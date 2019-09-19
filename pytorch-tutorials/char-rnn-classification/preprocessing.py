import glob
import os
import unicodedata
import string
import torch

all_letters = string.ascii_letters + " .,;#'"
n_letters = len(all_letters)

def find_files(path):
    return glob.glob(path)

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in string.ascii_letters + ".,:'")

# Read a file and split into lines
def readLines(filename):
    with open(filename, 'r', encoding='utf-8') as input_file:
        lines = input_file.read().strip().split('\n')
    
    return [unicode_to_ascii(l) for l in lines]

def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

def batch_to_tensor(names, max_length):
    """
    @param names (List[str]) 
    """
    tensor = torch.zeros(max_length, len(names), n_letters)
    for n_i, name in enumerate(names):
        for i, letter in enumerate(name):
            tensor[i][n_i][letter_to_index(letter)] = 1
    return tensor


if __name__ == "__main__":
    # category_lines = {}
    # all_categories = []
    # for filename in find_files('data/names/*.txt'):
    #     category = os.path.splitext(os.path.basename(filename))[0]
    #     all_categories.append(category)
    #     lines = readLines(filename)
    #     category_lines[category] = lines

    print(letter_to_tensor('J'))
