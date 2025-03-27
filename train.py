from matplotlib import pyplot as plt
from efficientNetModel import *
from datatset import *
from config import *
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tqdm import tqdm

args = cArgs()
print('device: ', args.device)


def train_fn(model, train_loader, optimizer, loss_fn):
    loop = tqdm(train_loader)
    total_loss = []
    for batch, (image, label) in enumerate(loop):
        image = image.to(args.device)
        label = label.float().to(args.device)

        predictions = model(image).squeeze(1)

        loss = loss_fn(predictions, label)
        total_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Average training loss per batch: {sum(total_loss)/len(total_loss)}')
    return sum(total_loss)/len(total_loss)


def main():
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # load training and validation data
    train_data = DogsVsWolvesDataset(args.train_dir)
    val_data = DogsVsWolvesDataset(args.val_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False)

    # create and load model
    model = EfficientNet('b0', num_classes=1).to(args.device)  
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    #load_checkpoint(args.save_path, model, optimizer, args.lr)

    for epoch in range(args.epochs):
        print(f'Training epoch {epoch+1}..\n')
        epoch_loss = train_fn(model, train_loader, optimizer, loss_fn)
        train_losses.append(epoch_loss)

        if args.save_model and (epoch+1) % args.save_frequency == 0:
            save_checkpoint(model, optimizer, args.save_path)

        if (epoch+1) % args.validation_frequency == 0:
            print('Validating the Accuracy..')
            val_accuracy = getAccuracy(model, val_loader, args.device)
            print(f'Validation Accuracy after epoch {epoch+1}: {val_accuracy}')
            val_accuracies.append(val_accuracy)

            # Giả định bạn có train_accuracy tính toán được
            train_accuracy = getAccuracy(model, train_loader, args.device)
            train_accuracies.append(train_accuracy)

    # Vẽ biểu đồ Loss và Accuracy
    plt.figure(figsize=(10, 4))

    # Vẽ Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss', linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    # Vẽ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy', linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")

    plt.show()

if __name__ == '__main__':
    main()