from datasets import load_dataset

dataset = load_dataset("NbAiLab/nbnn_language_detection")

train_data = dataset["train"]
test_data = dataset["test"]

from collections import Counter

counter = Counter()

for row in train_data:
    counter.update(row["text"].lower().split())

vocab = {w:i+1 for i,(w,_) in enumerate(counter.most_common(10000))}


MAX_LEN = 50

def tokenize(text):

    tokens = text.lower().split()

    ids = [vocab.get(t,0) for t in tokens][:MAX_LEN]

    ids += [0]*(MAX_LEN - len(ids))

    return ids


import torch
from torch.utils.data import Dataset

class LanguageDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data[idx]

        x = torch.tensor(tokenize(row["text"]))

        y = 0 if row["language"] == "nob" else 1

        return x, torch.tensor(y)

from torch.utils.data import DataLoader

train_dataset = LanguageDataset(train_data)
test_dataset = LanguageDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

from torch import nn

class LanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 128)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):

        x = self.embedding(x)

        x = x.mean(dim=1)

        return self.classifier(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LanguageModel(len(vocab)+1).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(loader):

    model.train()

    for X,y in loader:

        X,y = X.to(device), y.to(device)

        pred = model(X)

        loss = loss_fn(pred,y)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()


def evaluate(loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X,y in loader:

            X,y = X.to(device), y.to(device)

            pred = model(X)

            correct += (pred.argmax(1)==y).sum().item()

            total += y.size(0)

    print("accuracy:", correct/total)

for epoch in range(5):

    train(train_loader)

    evaluate(test_loader)


def predict(text):

    x = torch.tensor(tokenize(text)).unsqueeze(0).to(device)

    with torch.no_grad():

        pred = model(x)

    return "bokmål" if pred.argmax()==0 else "nynorsk"


print(predict("Kva synast du om å lære spansk?"))
print(predict("hva synst du om å lære spansk"))
