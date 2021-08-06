import time
import random
from typing import Union
import numpy as np
import torch
import torchmetrics
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import auc
from data_utils import k_fold_splitter, get_new_model, get_data_loaders
import optuna
from optuna import TrialPruned
from dataset import ImageDataSet


''' This file includes different optimization-related utility functions '''


def hyperopt(trial: optuna.Trial, dmap: dict, transforms: dict, epochs: int):
    # optuna suggestions..
    batch = trial.suggest_categorical('batch_size', [32, 128, 256])
    chosen_optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    lr = trial.suggest_loguniform('lr', low=1e-5, high=1e-3)
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

    device = get_device()
    scores = []

    for idx, data_maps in enumerate(k_fold_splitter(dmap, k=3)):
        print("\n" + 40 * "#" + f" INTERNAL FOLD {idx + 1} " + 40 * "#")
        train_loader, test_loader = get_data_loaders(data_maps, transforms, batch_sizes={'train': batch, 'test': 512})

        # get pretrained model, change last layer
        model = get_new_model(train_loader.dataset.classes)
        optimizer = optim_func(model.parameters(), **kwargs)
        package = {'model': model, 'optimizer': optimizer, 'criterion': nn.CrossEntropyLoss()}

        scorer = torchmetrics.classification.F1(train_loader.dataset.classes)

        model = fit_model(package, train_loader, epochs, device)
        fold_score = eval_model(model, test_loader, {"F1": scorer}, device)['F1']

        scores.append(fold_score)

        if idx < 2:
            trial.report(float(np.mean(scores)), idx)
            if trial.should_prune():
                raise TrialPruned()

    return np.mean(scores)


def fit_model(optimization_package: dict, train: DataLoader, epochs: int, device: Union[str, torch.device] = 'cpu') -> \
        nn.Module:
    """ Generic function to fit a model to some given dataset """
    model = optimization_package['model']
    optimizer = optimization_package['optimizer']
    criterion = optimization_package['criterion']

    model.to(device)
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        torch.cuda.empty_cache()
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

    torch.cuda.empty_cache()
    return model


def eval_model(model: nn.Module, test: DataLoader, scorers: dict, device: Union[str, torch.device]) -> dict:
    """ Generic function to run a model in eval mode on some data and record scores """
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()  # Set model to evaluate mode
    for batch in tqdm(test, desc=f'Evaluating', ncols=100):
        images, labels = batch
        images = images.to(device)

        # forward pass
        with torch.no_grad():
            outputs = model(images).detach().cpu()
            preds = torch.softmax(outputs, dim=1)

            for scorer in scorers.values():
                scorer(preds, labels)

    torch.cuda.empty_cache()
    return {name: scorer.compute() for name, scorer in scorers.items()}


def get_scorers(num_classes: int) -> dict:
    acc = torchmetrics.classification.Accuracy(num_classes=num_classes, compute_on_step=False)
    roc = torchmetrics.classification.ROC(num_classes=num_classes, compute_on_step=False)
    prec = torchmetrics.classification.Precision(num_classes=num_classes, compute_on_step=False)
    pr = torchmetrics.classification.PrecisionRecallCurve(num_classes=num_classes, compute_on_step=False)
    return {'accuracy': acc, 'roc': roc, 'precision': prec, 'pr': pr}


def get_auc_fpr_tpr(roc, pr) -> dict:
    """ this function uses precalculated fpr, tpr, precision, recall to calculated AUC under both curves """
    fpr, tpr, _ = roc

    fpr = torch.mean(torch.stack(fpr), dim=0).numpy()
    tpr = torch.mean(torch.stack(tpr), dim=0).numpy()
    auc_roc = auc(fpr, tpr)

    pr_auc = np.mean([auc(r.numpy(), p.numpy()) for p, r, _ in zip(*pr)])

    fpr = np.mean(fpr)
    tpr = np.mean(tpr)
    return {'auc_roc': auc_roc, 'auc_pr': pr_auc, 'fpr': fpr, 'tpr': tpr}


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_hyper_params(study_results: dict) -> (optim.Optimizer, dict):
    """ Helper function to translate optuna study results into an optimizer function and kwargs"""
    study_results = study_results.copy()
    opt = study_results.pop('optimizer')
    _ = study_results.pop('batch_size')
    kwargs = {'weight_decay': 1e-5}

    if opt == 'sgd':
        func = optim.SGD
        kwargs.update({'momentum': 0.9})

    elif opt == 'adam':
        func = optim.Adam

    else:
        raise KeyError(f"Chosen unknown optimizer: {optim}")

    kwargs.update(study_results)
    return func, kwargs


def time_model(model: nn.Module, dataset: ImageDataSet, num_infer: int, batch: int, device: Union[str, torch.device]) -> float:
    """ Measure model inference time on a given amount of samples, with some batch size """
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()
    num_infer //= batch

    t = time.time()
    for sample in DataLoader(dataset, batch):
        image, _ = sample
        image = image.to(device)
        with torch.no_grad():
            _ = model(image)

        num_infer -= 1
        if num_infer == 0:
            break

    record = time.time() - t
    torch.cuda.empty_cache()
    return record


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
