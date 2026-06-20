import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import wandb

from torchvision import transforms
from torch.utils.data import DataLoader

from hadamard_transform import HadamardCompression
from cnn import CIFARCNN
from scnn import CIFARSCNN
from torchvision.datasets import CIFAR10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--name", type=str, default="SCNN vs CNN")
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()


def main():
    args = get_args()

    torch.backends.cudnn.benchmark = True

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    wandb.init(
        project="SCNN",
        name=args.name,
        config=vars(args)
    )

    transform = transforms.Compose([
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
        transform=transform
    )

    testset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    cnn = CIFARCNN().to(device0)
    scnn = CIFARSCNN(10, args.num_steps).to(device1)

    criterion = nn.CrossEntropyLoss()

    cnn_optimizer = optim.Adam(cnn.parameters(), lr=1e-3)
    scnn_optimizer = optim.Adam(scnn.parameters(), lr=1e-3)

    cnn_scheduler = optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer, T_max=args.epochs)
    scnn_scheduler = optim.lr_scheduler.CosineAnnealingLR(scnn_optimizer, T_max=args.epochs)

    best_cnn_acc = 0.0
    best_scnn_acc = 0.0

    for epoch in range(args.epochs):

        cnn.train()
        scnn.train()

        cnn_loss_sum = 0.0
        scnn_loss_sum = 0.0

        cnn_correct = 0
        scnn_correct = 0
        total = 0

        for images, labels in trainloader:

            images = images

            cnn_images = images.to(device0, non_blocking=True)
            scnn_images = images.to(device1, non_blocking=True)

            labels0 = labels.to(device0, non_blocking=True)
            labels1 = labels.to(device1, non_blocking=True)

            cnn_optimizer.zero_grad(set_to_none=True)
            scnn_optimizer.zero_grad(set_to_none=True)

            cnn_outputs = cnn(cnn_images)
            scnn_outputs = scnn(scnn_images)

            cnn_loss = criterion(cnn_outputs, labels0)
            scnn_loss = criterion(scnn_outputs, labels1)

            cnn_loss.backward()
            scnn_loss.backward()

            cnn_optimizer.step()
            scnn_optimizer.step()

            cnn_loss_sum += cnn_loss.item()
            scnn_loss_sum += scnn_loss.item()

            cnn_correct += (cnn_outputs.argmax(1) == labels0).sum().item()
            scnn_correct += (scnn_outputs.argmax(1) == labels1).sum().item()

            total += labels.size(0)

        cnn.train_acc = 100 * cnn_correct / total
        scnn.train_acc = 100 * scnn_correct / total

        cnn.eval()
        scnn.eval()

        cnn_val_loss = 0.0
        scnn_val_loss = 0.0

        cnn_val_correct = 0
        scnn_val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in testloader:

                cnn_images = images.to(device0, non_blocking=True)
                scnn_images = images.to(device1, non_blocking=True)

                labels0 = labels.to(device0, non_blocking=True)
                labels1 = labels.to(device1, non_blocking=True)

                cnn_outputs = cnn(cnn_images)
                scnn_outputs = scnn(scnn_images)

                cnn_loss = criterion(cnn_outputs, labels0)
                scnn_loss = criterion(scnn_outputs, labels1)

                cnn_val_loss += cnn_loss.item()
                scnn_val_loss += scnn_loss.item()

                cnn_val_correct += (cnn_outputs.argmax(1) == labels0).sum().item()
                scnn_val_correct += (scnn_outputs.argmax(1) == labels1).sum().item()

                val_total += labels.size(0)

        cnn_val_acc = 100 * cnn_val_correct / val_total
        scnn_val_acc = 100 * scnn_val_correct / val_total

        cnn_scheduler.step()
        scnn_scheduler.step()

        if cnn_val_acc > best_cnn_acc:
            best_cnn_acc = cnn_val_acc
            torch.save(cnn.state_dict(), "best_cnn.pth")

        if scnn_val_acc > best_scnn_acc:
            best_scnn_acc = scnn_val_acc
            torch.save(scnn.state_dict(), "best_scnn.pth")

        wandb.log({
            "cnn/train_loss": cnn_loss_sum / len(trainloader),
            "cnn/train_acc": cnn.train_acc,
            "cnn/val_loss": cnn_val_loss / len(testloader),
            "cnn/val_acc": cnn_val_acc,

            "scnn/train_loss": scnn_loss_sum / len(trainloader),
            "scnn/train_acc": scnn.train_acc,
            "scnn/val_loss": scnn_val_loss / len(testloader),
            "scnn/val_acc": scnn_val_acc
        })

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"CNN acc {cnn_val_acc:.2f}% | "
            f"SCNN acc {scnn_val_acc:.2f}%"
        )

    print("Best CNN:", best_cnn_acc)
    print("Best SCNN:", best_scnn_acc)


if __name__ == "__main__":
    main()