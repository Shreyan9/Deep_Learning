import argparse

import torch
import torch.nn as nn

from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric
from homework.models import Classifier, save_model


def train(epochs: int = 40, lr: float = 2e-3, batch_size: int = 128, seed: int = 2024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    train_data = load_data("classification_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size)
    val_data = load_data("classification_data/val", transform_pipeline="default", shuffle=False, batch_size=batch_size)

    model = Classifier().to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_data)
    )

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_metric = AccuracyMetric()
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_metric.add(logits.argmax(dim=1), label)

        model.eval()
        val_metric = AccuracyMetric()
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                val_metric.add(model.predict(img), label)

        train_acc = train_metric.compute()["accuracy"]
        val_acc = val_metric.compute()["accuracy"]
        print(f"epoch {epoch + 1:2d}/{epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            save_model(model.cpu())
            model.to(device)

    print(f"best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))
