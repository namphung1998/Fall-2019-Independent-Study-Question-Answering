import torch
from preprocessing import *
from train import load_data, batch_iter

from model import GRUClassifier

def predict(model, names, lengths, categories):
    outputs = model(batch_to_tensor(names, lengths[0]), lengths)
    _, prediction = torch.max(outputs, dim=1)

    return categories[prediction.item()]



all_categories = []
for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)

all_categories.sort()

X_train, y_train, X_test, y_test = load_data(all_categories)

model = GRUClassifier(n_letters, 32, len(all_categories))
model.load_state_dict(torch.load('models/test2.pt'))

if __name__ == "__main__":
    print(predict(model, ['Gomes'], [len('Gomes')], all_categories))

# test_iter = batch_iter(X_test, y_test, 16)

# with torch.no_grad():
#     correct = 0
#     for names, categories, lengths in test_iter:
#         outputs = model(batch_to_tensor(names, lengths[0]), lengths)
#         _, prediction = torch.max(outputs, dim=1)

#         correct += torch.sum(prediction == categories).item()

#     print(correct / len(X_test))

