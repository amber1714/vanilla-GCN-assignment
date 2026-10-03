import random

import numpy as np
import torch

from data import load_karate_club
from model import VanillaGCN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def train_model(epochs: int = 300, lr: float = 0.01, weight_decay: float = 5e-4):
    set_seed(42)
    _, adjacency, features, labels, train_idx, val_idx, test_idx = load_karate_club()

    model = VanillaGCN(
        in_features=features.size(1),
        hidden_features=16,
        num_classes=int(labels.max().item()) + 1,
        dropout=0.5,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adjacency)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, adjacency)
            train_acc = accuracy(logits[train_idx], labels[train_idx])
            val_acc = accuracy(logits[val_idx], labels[val_idx])

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(features, adjacency)
        test_acc = accuracy(logits[test_idx], labels[test_idx])
        predictions = logits.argmax(dim=1)

    print(f"Final test accuracy: {test_acc:.3f}")
    return model, predictions, test_acc


if __name__ == "__main__":
    train_model()
