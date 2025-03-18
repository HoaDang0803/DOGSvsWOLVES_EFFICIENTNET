import torch
import numpy as np
import os
import random
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader

from datatset import DogsVsWolvesDataset
from efficientNetModel import EfficientNet


def createTrainAndEval(root_dir='data', val_size=100):
    os.makedirs(os.path.join(root_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'train', 'dogs'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'train', 'wolves'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'val', 'dogs'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'val', 'wolves'), exist_ok=True)

    dog_list = os.listdir(os.path.join(root_dir, 'dogs'))
    wolves_list = os.listdir(os.path.join(root_dir, 'wolves'))

    assert len(dog_list) ==  1000, 'Something not write with base data..'
    assert len(wolves_list) ==  1000, 'Something not write with base data..'

    random.shuffle(dog_list)
    random.shuffle(wolves_list)

    dog_val = dog_list[:val_size]
    dog_train = dog_list[val_size:]

    wolf_val = wolves_list[:val_size]
    wolf_train = wolves_list[val_size:]


    print('Creating validation set..')
    for idx in tqdm(range(val_size)):
        shutil.copyfile(os.path.join(root_dir,'dogs',dog_val[idx]), os.path.join(root_dir, 'val', 'dogs', os.path.basename(dog_val[idx])))
        shutil.copyfile(os.path.join(root_dir,'wolves',wolf_val[idx]), os.path.join(root_dir, 'val', 'wolves', os.path.basename(wolf_val[idx])))
    print('Creating training set..')
    for idx in tqdm(range(1000 - val_size)):
        shutil.copyfile(os.path.join(root_dir,'dogs',dog_train[idx]), os.path.join(root_dir, 'train', 'dogs', os.path.basename(dog_train[idx])))
        shutil.copyfile(os.path.join(root_dir,'wolves',wolf_train[idx]), os.path.join(root_dir, 'train', 'wolves', os.path.basename(wolf_train[idx])))
    print('Done!')


def save_checkpoint(model, optimizer, filename):
    print("Saving checkpoint!")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("loading checkpoint")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    try:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    except RuntimeError as e:
        print("Error while trying to update optimizer lr")
        raise RuntimeError(e)


def getMeanAndStd(dataloader):
    """returns the mean and std of a dataset"""
    # std[x] = sqrt(E[x**2] - E[x]**2)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(dataloader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])  # mean of batch across HXW, for each channel
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def getAccuracy(model, val_loader, device, debug=False):
    total_correct = 0.0
    total = 0.0

    model.eval()
    for (image_batch, labels) in tqdm(val_loader):
        image_batch = image_batch.to(device)
        labels = labels.int()
        predictions = model(image_batch).int()
        labels = np.array([label.item() for label in labels])
        predictions = np.array([0 if prediction.item() < 0.5 else 1 for prediction in predictions])

        total_correct += sum(labels == predictions)
        total += image_batch.shape[0]

        if debug:
            print('labels:')
            print(labels)
            print('predictions:')
            print(predictions)
            print('correct: ', sum(labels==predictions))
            print('total per batch:', image_batch.shape[0])

    model.train()
    return total_correct / total



if __name__ == '__main__':
    ds = DogsVsWolvesDataset('data/val')
    dl = DataLoader(ds, batch_size=10, shuffle=False)
    model = EfficientNet('b0', num_classes=1)

    print('Accuracy: ', getAccuracy(model, dl, 'cpu', debug=True))