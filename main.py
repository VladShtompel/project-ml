import torch
try:
    torch.multiprocessing.set_start_method("spawn", force=True)
except:
    pass

from torchvision import transforms
from data_utils import data_map, seed_all
from functools import partial
from optim_utils import optimize
import optuna
from optuna.pruners import MedianPruner

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

seed_all(42)


if __name__ == '__main__':
    path = r'data/vgg-cats'
    dmap = data_map(path)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]),

        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
    }

    hyperopt_func = partial(optimize, dmap=dmap, transforms=data_transforms)
    pruner = MedianPruner(n_startup_trials=10)
    study = optuna.create_study(study_name='Hyperopt', direction='maximize', pruner=pruner)
    study.optimize(hyperopt_func, n_trials=50)
    print(study.best_value, study.best_params)
