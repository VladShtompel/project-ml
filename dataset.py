import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.transforms import Compose


class ImageDataSet(Dataset):
    def __init__(self, data_map: dict, mean: torch.Tensor, std: torch.Tensor, data_transforms: Compose,
                 preload: bool = False):

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._data = data_map
        self._transform = data_transforms
        self._len = sum([len(v) for v in self._data.values()])
        self._classes = len(self._data.keys())
        self._cache = None
        self._mean = mean.unsqueeze(1).unsqueeze(1)
        self._std = std.unsqueeze(1).unsqueeze(1)

        if preload:
            self._load_all()

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        if idx < self.len:
            if self._cache is None:
                image, label = self._load_item_from_disk(idx)
            else:
                image, label = self._load_item_from_ram(idx)

        else:
            raise IndexError(f"Index {idx:,} is out of range for dataset with length {self.len:,}")

        # augment, normalize
        image = self._transform(image)
        image = self._normalize(image)
        label = torch.tensor(label)
        return image, label

    def _load_all(self):
        self._cache = {}
        for idx in tqdm(range(self.len), desc='Loading data', ncols=100):
            item = self._load_item_from_disk(idx)
            self._cache[idx] = item

    def _load_item_from_disk(self, idx: int) -> (np.ndarray, int):
        for cls, samples in self._data.items():
            if idx < len(samples):
                item = {'X': samples[idx], 'y': cls}
                break

            else:
                idx -= len(samples)

        image = cv2.imread(item['X'])[:, :, ::-1].copy()
        label = item['y']
        return image, label

    def _load_item_from_ram(self, idx: int) -> (np.ndarray, int):
        temp, label = self._cache[idx]
        image = temp.copy()
        return image, label

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / (self.std + 1e-7)

    @property
    def len(self):
        return self._len

    @property
    def classes(self):
        return self._classes

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def transforms(self):
        return self._transform

    @property
    def device(self):
        return self._device
