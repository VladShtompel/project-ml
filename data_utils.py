import math
import os
import random

import cv2
import numpy as np
import torch
from torchvision import transforms


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def to_tensor(array: np.ndarray) -> torch.Tensor:
    array = torch.tensor(array)
    array = torch.permute(array, (2, 0, 1))
    return array


def data_map(path: str) -> dict:
    dmap = {}
    for root, dirs, files in os.walk(path):
        if files:
            k = os.path.split(root)[-1]
            k = int(k.split('_')[0])
            dmap[k] = np.array([os.path.join(root, f) for f in files])

    return dmap


def get_mean_std(dmap: dict) -> (torch.Tensor, torch.Tensor):
    resize = transforms.Resize((64, 64))  # TODO: size?
    n_samples = sum([len(v) for v in dmap.values()])
    images = torch.zeros((n_samples, 3, 64, 64), dtype=torch.float32)

    idx = 0
    for samples in dmap.values():
        for path in samples:
            img = cv2.imread(path)[:, :, ::-1].copy()
            img = to_tensor(img)
            img = resize(img)
            images[idx] = img
            idx += 1

    images = images.reshape(3, -1)
    return images.mean(dim=-1), images.std(dim=-1)


def kfold_splitter(dmap: dict, *, k: int, shuffle: bool = False) -> (dict, dict):
    if shuffle:
        dmap_new = {}
        for cls, samples in dmap.items():
            temp = samples.copy()
            np.random.shuffle(temp)
            dmap_new[cls] = temp

        dmap = dmap_new

    num_samples_in_fold_per_cls = {cls: math.ceil(len(samples) / k) for cls, samples in dmap.items()}

    for fold in range(k):
        train_folds = {}
        test_fold = {}

        for cls, samples in dmap.items():
            num_in_fold = num_samples_in_fold_per_cls[cls]

            all_idx = np.arange(len(samples))
            idx_in = (fold * num_in_fold <= all_idx) & (all_idx < (fold + 1) * num_in_fold)
            idx_out = ~idx_in

            train_folds[cls] = samples[idx_out].copy()
            test_fold[cls] = samples[idx_in].copy()

        yield train_folds, test_fold
