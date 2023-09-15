import torch
import torch.utils.data as data
import torch.nn as nn
import sklearn


def compute_metric(dataset_loader : data.DataLoader, model: nn.Module, metric):
    preds, targets = concatenate_predictions(dataset_loader, model)
    return metric(y_pred=preds.numpy(), y_true=targets.numpy())


def concatenate_predictions(dataset: data.DataLoader, model: nn.Module):

    with torch.no_grad():
        preds, targets = [], []
        for X, y in dataset:
            preds.append(model(X).argmax(dim=-1))
            targets.append(y)
        return torch.concatenate(preds), torch.concatenate(targets)


def concatenate_true_class_probs(dataset: data.DataLoader, model: nn.Module):

    with torch.no_grad():
        probs = []
        for X, y in dataset:
            out = torch.softmax(model(X), dim=-1)
            p = torch.gather(out, dim=-1, index=torch.unsqueeze(y, dim=1))
            probs.append(torch.squeeze(p))

        return torch.concatenate(probs)


def get_best_predictions(dataset: data.DataLoader, model: nn.Module, n=5):
    probs = concatenate_true_class_probs(dataset, model)
    probs, idxs = torch.sort(probs, descending=True)

    return probs[:n], idxs[:n]


def get_worst_predictions(dataset: data.DataLoader, model: nn.Module, n=5):
    probs = concatenate_true_class_probs(dataset, model)
    probs, idxs = torch.sort(probs, descending=False)

    return probs[:n], idxs[:n]