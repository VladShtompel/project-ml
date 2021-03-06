import math
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import ImageDataSet


''' This file includes different data-related utility functions '''


def data_map(path: str) -> dict:
    """ Create a mapping of the classes and the data
    This mapping will then be used for nearly everything data-related.
    """
    dmap = {}
    for root, dirs, files in os.walk(path):
        if files:
            k = os.path.split(root)[-1]
            k = int(k.split('_')[0])
            dmap[k] = np.array([os.path.join(root, f) for f in files])

    return dmap


def get_mean_std(dmap: dict) -> (torch.Tensor, torch.Tensor):
    """ Calculated mean, std of the data in the given map """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])
    n_samples = sum([len(v) for v in dmap.values()])
    images = torch.zeros((n_samples, 3, 32, 32), dtype=torch.float32)

    idx = 0
    for samples in dmap.values():
        for path in samples:
            img = cv2.imread(path)[:, :, ::-1].copy()
            img = transform(img)
            images[idx] = img
            idx += 1

    images = images.reshape(3, -1)
    return images.mean(dim=-1), images.std(dim=-1)


def get_new_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ Returns a new resnet18 model with the last layer changed appropriately """
    model = models.resnet18(pretrained=pretrained)
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, num_classes)
    return model


def k_fold_splitter(dmap: dict, *, k: int, shuffle: bool = False) -> dict:
    """ This is a generator, initialize by passing a datamap and number of folds.
        Each call will then yield the next splitting of (k-1), (1) folds """
    if shuffle:
        dmap_new = {}
        for cls, samples in dmap.items():
            temp = samples.copy()
            np.random.shuffle(temp)
            dmap_new[cls] = temp

        dmap = dmap_new

    num_samples_in_fold_per_cls = {cls: math.floor(len(samples) / k) for cls, samples in dmap.items()}

    for fold in range(k):
        train_folds = {}
        test_fold = {}

        # generate split
        for cls, samples in dmap.items():
            num_in_fold = num_samples_in_fold_per_cls[cls]

            all_idx = np.arange(len(samples))
            idx_in = (fold * num_in_fold <= all_idx) & (all_idx < (fold + 1) * num_in_fold)
            idx_out = ~idx_in

            train_folds[cls] = samples[idx_out].copy()
            test_fold[cls] = samples[idx_in].copy()

        yield {'train': train_folds, 'test': test_fold}


def get_data_loaders(data_maps: dict, _transforms: dict, batch_sizes: dict) -> (DataLoader, DataLoader):
    """ A helper function to get the appropriate data loaders """
    mean, std = get_mean_std(data_maps['train'])

    train_dset = ImageDataSet(data_maps['train'], mean, std, data_transforms=_transforms['train'])
    test_dset = ImageDataSet(data_maps['test'], mean, std, data_transforms=_transforms['test'])

    train_loader = DataLoader(train_dset, batch_size=batch_sizes['train'], shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=batch_sizes['test'])

    return train_loader, test_loader


def get_df(path):
    """ Get a new or an existing results dataframe """
    if os.path.isfile(path):
        return pd.read_csv(path)

    else:
        return pd.DataFrame(columns=['Dataset', 'Algorithm', 'CV-Fold', 'Hyper-parameters', 'Accuracy', 'TPR', 'FPR',
                                     'Precision', 'AUC', 'PR-Curve', 'Train_time', 'Infer_time'])


def get_new_row(data_set: str, algo: str, fold: int, params: dict, evaluation: dict, timing: dict) -> dict:
    """ Transform the inpout data into a dataframe row """
    row = {
        'Dataset': data_set,
        'Algorithm': algo,
        'CV-Fold': fold,
        'Hyper-parameters': params,
        'Accuracy': evaluation['accuracy'].item(),
        'TPR': evaluation['tpr'],
        'FPR': evaluation['fpr'],
        'Precision': evaluation['precision'].item(),
        'AUC': evaluation['auc_roc'],
        'PR-Curve': evaluation['auc_pr'],
        'Train_time': timing['train'],
        'Infer_time': timing['infer']
    }
    return row
