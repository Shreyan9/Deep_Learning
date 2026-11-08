import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric
from homework.models import Detector, save_model


def augment(image, track, depth):
    """
    Batch augmentation on the GPU.
    A horizontal flip must also swap the left (1) and right (2) boundary labels and flip depth.
    """
    flip = torch.rand(image.size(0), device=image.device) < 0.5
    if flip.any():
        image = torch.where(flip[:, None, None, None], image.flip(-1), image)
        depth = torch.where(flip[:, None, None], depth.flip(-1), depth)
        flipped_track = track.flip(-1)
        flipped_track = torch.where(flipped_track == 1, 2, torch.where(flipped_track == 2, 1, flipped_track))
        track = torch.where(flip[:, None, None], flipped_track, track)

    # brightness / contrast jitter (labels unaffected)
    b = 1 + (torch.rand(image.size(0), 1, 1, 1, device=image.device) - 0.5) * 0.4
    c = (torch.rand(image.size(0), 1, 1, 1, device=image.device) - 0.5) * 0.1
    image = (image * b + c).clamp(0, 1)

    return image, track, depth


def train(
    epochs: int = 40,
    lr: float = 2e-3,
    batch_size: int = 32,
    depth_weight: float = 5.0,
    seed: int = 2024,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    train_data = load_data("drive_data/train", transform_pipeline="default", shuffle=True, batch_size=batch_size)
    val_data = load_data("drive_data/val", transform_pipeline="default", shuffle=False, batch_size=batch_size)

    model = Detector().to(device)
    # background dominates the image, so up-weight the two lane boundary classes
    class_weights = torch.tensor([1.0, 4.0, 4.0], device=device)
    seg_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_data)
    )

    best_score = -1.0
    for epoch in range(epochs):
        model.train()
        for batch in train_data:
            image = batch["image"].to(device)
            track = batch["track"].to(device)
            depth = batch["depth"].to(device)

            image, track, depth = augment(image, track, depth)

            logits, pred_depth = model(image)
            loss = seg_loss_fn(logits, track) + depth_weight * F.l1_loss(pred_depth, depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        metric = DetectionMetric()
        with torch.inference_mode():
            for batch in val_data:
                image = batch["image"].to(device)
                track = batch["track"].to(device)
                depth = batch["depth"].to(device)
                pred, pred_depth = model.predict(image)
                metric.add(pred, track, pred_depth, depth)

        m = metric.compute()
        # keep the checkpoint with the best mix of IoU and depth error
        score = m["iou"] - m["abs_depth_error"]
        print(
            f"epoch {epoch + 1:2d}/{epochs} loss={loss.item():.4f} "
            f"iou={m['iou']:.4f} acc={m['accuracy']:.4f} "
            f"depth_err={m['abs_depth_error']:.4f} tp_depth_err={m['tp_depth_error']:.4f}"
        )

        if score >= best_score:
            best_score = score
            save_model(model.cpu())
            model.to(device)

    print(f"best (iou - depth_err) = {best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--depth_weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))
