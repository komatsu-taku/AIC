from matplotlib import transforms
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchinfo import summary
import numpy as np
import argparse
from tqdm import tqdm

from vgg.model import VGG, VGG16
from cnn.model import CNN
from utils.utils import load_checkpoint, save_checkpoint


def main(args: argparse.Namespace):
    
    # データセットの作成
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root = "cifar10",
        train = True,
        transform = train_transforms,
        download=True
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle = True, drop_last=True, num_workers=args.num_workers
    )

    test_dataset = datasets.CIFAR10(
        root = "cifar10",
        train = False,
        transform = test_transforms,
        download = True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size = args.batch_size,
        shuffle = False, num_workers=args.num_workers
    )

    # モデルの作成
    if args.model == "cnn":
        model = CNN(num_classes=10)
    elif args.model == "vgg":
        model = VGG(num_classes=10)
    elif args.model == "vgg16":
        model = VGG16(num_classes=10)

    # summary の表示
    input_size = (64, 3, 64, 64)
    summary(model, input_size)

    # loss 
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # gpu
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    
    # load savepoints if need / want
    start_n_iter = 0
    start_epoch = 0
    if args.resume:
        checkpoints = load_checkpoint(args.path_to_checkpoints)
        model.load_state_dict(checkpoints["net"])
        start_epoch = checkpoints["epoch"]
        start_n_iter = checkpoints["n_iter"]
        optimizer.load_state_dict(checkpoints["optim"])
        print("last checkpoints restored")
    
    n_iter = start_n_iter
    for epoch in range(start_epoch, args.epochs):
        print(f"\n[Epochs {epoch +1}]")
        model.train()

        for data in tqdm(train_dataloader):
            images, labels = data
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 1 == 0:
            model.eval()

            correct = 0
            total = 0

            with torch.no_grad():
                for data in tqdm(test_dataloader):
                    images, labels = data
                    if use_cuda:
                        images = images.cuda()
                        labels = labels.cuda()
                    
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            print(f"Accuracy on test set : {100 * correct / total : .2f}")

            checkpoints = {
                "model" : model.state_dict(),
                "epoch" : epoch,
                "n_iter" : n_iter,
                "optim" : optimizer.state_dict()
            }
            save_checkpoint(checkpoints, f"{args.model}/checkpoints.ckpt")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--path_checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--model", type=str, choices=["cnn", "vgg", "vgg16"], default="VGG"
    )

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())