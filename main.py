import os
import time

from torch import nn
from torchvision import transforms
from data_utils import data_map, seed_all, kfold_splitter, get_new_model, get_data_loaders, get_df, get_new_row
from functools import partial
from optim_utils import hyperopt, fit_model, eval_model, get_device, get_scorers, get_hyper_params, get_auc_fpr_tpr, \
    time_model
import optuna
from optuna.pruners import MedianPruner, ThresholdPruner, PercentilePruner
from RandAugment import RandAugment

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

seed_all(42)


if __name__ == '__main__':
    results_path = r'results.csv'
    N = 2
    M = 9
    trns = {
        'RandAugment': transforms.Compose([
            transforms.ToPILImage(),
            RandAugment(N, M),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),

        'Baseline': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ])
    }

    device = get_device()
    results_df = get_df(results_path)
    train_epochs = 1
    hopt_epochs = 1
    hopt_trials = 1

    for algo in ['Baseline', 'RandAugment']:

        data_transforms = {
            'train': trns[algo],

            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        }
        for data_set in sorted(os.listdir('data')):
            if 'cifar-100_5' not in data_set and 'cats' not in data_set:
                continue

            path = os.path.join('data', data_set)
            dmap = data_map(path)
            for fold, data_maps in enumerate(kfold_splitter(dmap, k=10), 1):
                # 3 x 50 hyperopt trials
                hyperopt_func = partial(hyperopt, dmap=data_maps['train'], transforms=data_transforms, epochs=hopt_epochs)
                pruner = PercentilePruner(0.9, n_startup_trials=10)
                study = optuna.create_study(study_name=f'Hyperopt {path}', direction='maximize', pruner=pruner)
                study.optimize(hyperopt_func, n_trials=hopt_trials)
                print(study.best_value, study.best_params)

                batch_sizes = {'train': study.best_params['batch_size'], 'test': 256}
                train_dl, test_dl = get_data_loaders(data_maps, data_transforms, batch_sizes)

                # now train model for 10-15 epoch
                func, hyper = get_hyper_params(study.best_params)
                model = get_new_model(num_classes=train_dl.dataset.classes)
                optimizer = func(model.parameters(), **hyper)

                package = {'model': model, 'criterion': nn.CrossEntropyLoss(), 'optimizer': optimizer}

                train_start = time.time()
                model = fit_model(package, train_dl, epochs=train_epochs, device=device)
                train_time = time.time() - train_start

                # eval, get all requested metrics
                scorers = get_scorers(train_dl.dataset.classes)
                evaluation = eval_model(model, test_dl, scorers, device=device)
                evaluation.update(get_auc_fpr_tpr(evaluation['roc'], evaluation['pr']))

                infer_time = time_model(model, train_dl.dataset, 1000, 1, device)
                timing = {'train': train_time, 'infer': infer_time}

                row = get_new_row(data_set, algo, fold, study.best_params, evaluation, timing)
                results_df = results_df.append(row, ignore_index=True)
                results_df.to_csv(results_path, index=False)
