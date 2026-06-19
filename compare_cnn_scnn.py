import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hadamard_transform import HadamardCompression
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from cnn import CIFARCNN
from scnn import CIFARSCNN
import wandb
def main():
    wandb.init(
    project="SCNN",
    name="CNN vs SCNN",
     config={
        "keep_ratio": 0.10
     }
    )
    train_transform = transforms.Compose([
        HadamardCompression(keep_ratio=0.10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    test_transform = transforms.Compose([
        HadamardCompression(keep_ratio=0.10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    trainset = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    testset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    testloader = DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    cnn = CIFARCNN().cuda(0)
    scnn = CIFARSCNN().cuda(1)
    criterion = nn.CrossEntropyLoss()
    cnn_optimizer = optim.Adam(
        cnn.parameters(),
        lr=1e-3
    )
    scnn_optimizer = optim.Adam(
        scnn.parameters(),
        lr=1e-3
    )
    cnn_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer,
        T_max=250
    )
    scnn_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        scnn_optimizer,
        T_max=250
    )
    best_cnn_acc = 0.0
    best_scnn_acc = 0.0
    num_epochs = 250

    for epoch in range(num_epochs):

        cnn.train()
        scnn.train()

        cnn_train_loss = 0.0
        scnn_train_loss = 0.0

        cnn_correct = 0
        scnn_correct = 0

        train_total = 0

        for images, labels in trainloader:

            cnn_images = images.cuda(0, non_blocking=True)
            cnn_labels = labels.cuda(0, non_blocking=True)

            scnn_images = images.cuda(1, non_blocking=True)
            scnn_labels = labels.cuda(1, non_blocking=True)

            cnn_optimizer.zero_grad()
            scnn_optimizer.zero_grad()

            cnn_outputs = cnn(cnn_images)
            scnn_outputs = scnn(scnn_images)

            cnn_loss = criterion(
                cnn_outputs,
                cnn_labels
            )

            scnn_loss = criterion(
                scnn_outputs,
                scnn_labels
            )

            cnn_loss.backward()
            scnn_loss.backward()

            cnn_optimizer.step()
            scnn_optimizer.step()

            cnn_train_loss += cnn_loss.item()
            scnn_train_loss += scnn_loss.item()

            cnn_preds = cnn_outputs.argmax(dim=1)
            scnn_preds = scnn_outputs.argmax(dim=1)

            cnn_correct += (cnn_preds == cnn_labels).sum().item()
            scnn_correct += (scnn_preds == scnn_labels).sum().item()

            train_total += labels.size(0)

        cnn_train_loss /= len(trainloader)
        scnn_train_loss /= len(trainloader)

        cnn_train_acc = 100.0 * cnn_correct / train_total
        scnn_train_acc = 100.0 * scnn_correct / train_total

        cnn.eval()
        scnn.eval()

        cnn_val_loss = 0.0
        scnn_val_loss = 0.0

        cnn_val_correct = 0
        scnn_val_correct = 0

        val_total = 0

        with torch.no_grad():

            for images, labels in testloader:

                cnn_images = images.cuda(0, non_blocking=True)
                cnn_labels = labels.cuda(0, non_blocking=True)

                scnn_images = images.cuda(1, non_blocking=True)
                scnn_labels = labels.cuda(1, non_blocking=True)

                cnn_outputs = cnn(cnn_images)
                scnn_outputs = scnn(scnn_images)

                cnn_loss = criterion(
                    cnn_outputs,
                    cnn_labels
                )

                scnn_loss = criterion(
                    scnn_outputs,
                    scnn_labels
                )

                cnn_val_loss += cnn_loss.item()
                scnn_val_loss += scnn_loss.item()

                cnn_preds = cnn_outputs.argmax(dim=1)
                scnn_preds = scnn_outputs.argmax(dim=1)

                cnn_val_correct += (cnn_preds == cnn_labels).sum().item()
                scnn_val_correct += (scnn_preds == scnn_labels).sum().item()

                val_total += labels.size(0)

        cnn_val_loss /= len(testloader)
        scnn_val_loss /= len(testloader)

        cnn_val_acc = 100.0 * cnn_val_correct / val_total
        scnn_val_acc = 100.0 * scnn_val_correct / val_total

        cnn_scheduler.step()
        scnn_scheduler.step()

        if cnn_val_acc > best_cnn_acc:
            best_cnn_acc = cnn_val_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": cnn.state_dict(),
                    "optimizer_state_dict": cnn_optimizer.state_dict(),
                    "accuracy": cnn_val_acc
                },
                "best_cnn.pth"
            )

        if scnn_val_acc > best_scnn_acc:
            best_scnn_acc = scnn_val_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": scnn.state_dict(),
                    "optimizer_state_dict": scnn_optimizer.state_dict(),
                    "accuracy": scnn_val_acc
                },
                "best_scnn.pth"
            )

        print(
            f"Epoch [{epoch+1:03d}/{num_epochs}] | "
            f"CNN Train Loss: {cnn_train_loss:.4f} | "
            f"CNN Train Acc: {cnn_train_acc:.2f}% | "
            f"CNN Val Loss: {cnn_val_loss:.4f} | "
            f"CNN Val Acc: {cnn_val_acc:.2f}% | "
            f"SCNN Train Loss: {scnn_train_loss:.4f} | "
            f"SCNN Train Acc: {scnn_train_acc:.2f}% | "
            f"SCNN Val Loss: {scnn_val_loss:.4f} | "
            f"SCNN Val Acc: {scnn_val_acc:.2f}%"
        )
        wandb.log({
            "cnn/train_loss": cnn_train_loss,
            "cnn/train_acc": cnn_train_acc,
            "cnn/val_loss": cnn_val_loss,
            "cnn/val_acc": cnn_val_acc,

            "scnn/train_loss": scnn_train_loss,
            "scnn/train_acc": scnn_train_acc,
            "scnn/val_loss": scnn_val_loss,
            "scnn/val_acc": scnn_val_acc,
        })
    print(f"Best CNN Accuracy: {best_cnn_acc:.2f}%")
    print(f"Best SCNN Accuracy: {best_scnn_acc:.2f}%")