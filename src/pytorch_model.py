import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np


class NeuralNet(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


def run_pytorch_model(X_train, X_test, y_train, y_test):

    # Convert everything to numpy safely
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Convert to tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float().view(-1, 1)
    X_test = torch.tensor(X_test).float()

    model = NeuralNet(X_train.shape[1])

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):

        model.train()

        outputs = model(X_train)

        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = model(X_test).detach().numpy()
    preds = (preds > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)

    print(f"PyTorch Accuracy: {acc:.4f}")

    return acc