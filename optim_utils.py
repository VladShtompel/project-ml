import numpy as np
import torch
import torchmetrics
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
import torch.optim as optim
from tqdm import tqdm

from data_utils import kfold_splitter, get_mean_std
from dataset import ImageDataSet
import optuna
from optuna import TrialPruned


def optimize(trial: optuna.Trial, dmap: dict, transforms: dict):
    # optuna suggestions..
    batch = trial.suggest_int('batch_size', low=32, high=192, step=32)
    chosen_optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    lr = trial.suggest_loguniform('learning_rate', low=1e-5, high=1e-3)
    # wd = trial.suggest_loguniform('weight_decay', low=1e-6, high=1e-4)
    kwargs = {'lr': lr, 'weight_decay': 1e-5}

    if chosen_optimizer == 'sgd':
        nest = trial.suggest_categorical('nesterov', [True, False])
        kwargs.update({'nesterov': nest, 'momentum': 0.9})
        optim_func = optim.SGD

    elif chosen_optimizer == 'adam':
        amsgrad = trial.suggest_categorical('amsgrad', [True, False])
        kwargs.update({'amsgrad': amsgrad})
        optim_func = optim.Adam

    else:
        raise KeyError(f"Chosen unknown optimizer: {chosen_optimizer}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    scores = []

    for idx, (train, test) in enumerate(kfold_splitter(dmap, k=3)):
        print(40 * "#" + f" INTERNAL FOLD {idx + 1} " + 40 * "#")
        mean, std = get_mean_std(train)

        train_dset = ImageDataSet(train, mean, std, data_transforms=transforms['train'], preload=True)
        test_dset = ImageDataSet(test, mean, std, data_transforms=transforms['test'], preload=True)

        train_loader = DataLoader(train_dset, batch_size=batch, shuffle=True)
        test_loader = DataLoader(test_dset, batch_size=batch)

        # get pretrained model, change last layer
        model = models.resnet34(pretrained=True)
        num_feats = model.fc.in_features
        model.fc = nn.Linear(num_feats, train_dset.classes)

        scorer = torchmetrics.classification.F1(train_dset.classes)
        optimizer = optim_func(model.parameters(), **kwargs)
        package = {'model': model, 'optimizer': optimizer, 'criterion': criterion, 'device': device}

        fold_score = fit_and_evaluate(package, train_loader, test_loader, scorer, epochs=5)

        trial.report(fold_score, idx)
        if trial.should_prune():
            TrialPruned()

        scores.append(fold_score)

    return np.mean(scores)


def fit_and_evaluate(optimization_package: dict, train: DataLoader, test: DataLoader, scorer, epochs: int) -> float:
    model = optimization_package['model']
    optimizer = optimization_package['optimizer']
    criterion = optimization_package['criterion']
    device = optimization_package['device']
    model.to(device)

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        # Each epoch has a training and validation phase

        # Train loop
        model.train()  # Set model to training mode
        for batch in tqdm(train, desc=f'Epoch {epoch + 1}', ncols=100):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

    score = eval_model(model, test, scorer, device)
    print()
    return score


def eval_model(model: nn.Module, data_loader: DataLoader, scorer, device='cpu') -> float:
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()  # Set model to evaluate mode
    for batch in tqdm(data_loader, desc=f'Evaluating', ncols=100):
        images, labels = batch
        images = images.to(device)

        # forward pass
        with torch.no_grad():
            outputs = model(images).detach().cpu()
            preds = torch.softmax(outputs, dim=1)
            scorer(preds, labels)

    torch.cuda.empty_cache()
    score = scorer.compute()
    return score.item()
