import math
from typing import Callable, Tuple, List
from tqdm import tqdm
import time
import numpy as np
import torch
from torch import LongTensor, FloatTensor
from tensorskipgram.evaluation.data import SentenceData
from torch.utils.data import Dataset, DataLoader


def formatTime():
    return time.strftime('%X %x %Z')


def pearson(predictions, labels):
    return np.corrcoef(predictions, labels)[0, 1]


def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes, dtype=torch.float)
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0, floor-1] = 1
    else:
        target[0, floor-1] = ceil - label
        target[0, ceil-1] = label - floor
    return target


def train_batch(network: torch.nn.Module,
                X_sentence1: SentenceData,
                X_sentence2: SentenceData,
                Y_batch: LongTensor,
                loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
                optimizer: torch.optim.Optimizer) -> float:
    network.train()
    prediction_batch = network(X_sentence1, X_sentence2)  # forward pass
    Y_val = map_label_to_target(Y_batch.item(), 5)
    batch_loss = loss_fn(prediction_batch, Y_val)  # loss calculation
    batch_loss.backward()  # gradient computation
    optimizer.step()  # back-propagation
    optimizer.zero_grad()  # gradient reset
    return batch_loss.item()


def train_batch_e(network: torch.nn.Module,
                X_sentence1: SentenceData,
                X_sentence2: SentenceData,
                Y_batch: LongTensor,
                loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
                optimizer: torch.optim.Optimizer) -> float:
    network.train()
    prediction_batch = network(X_sentence1, X_sentence2)  # forward pass
    batch_loss = loss_fn(prediction_batch.unsqueeze(0), Y_batch[0].long())  # loss calculation
    batch_loss.backward()  # gradient computation
    optimizer.step()  # back-propagation
    optimizer.zero_grad()  # gradient reset
    return batch_loss.item()

def train_epoch(name: str,
                network: torch.nn.Module,
                dataloader: DataLoader,
                loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
                optimizer: torch.optim.Optimizer,
                device: str,
                epoch_idx: int) -> float:
    datalen = len(dataloader)
    loss = 0.
    for i, (x_sentence1, x_sentence2, y_batch) in enumerate(dataloader):
        x_sentence1 = list(map(lambda d: d.to(device), x_sentence1))
        x_sentence2 = list(map(lambda d: d.to(device), x_sentence2))
        y_batch = y_batch.to(device, dtype=torch.float32)
        loss += train_batch(network=network, X_sentence1=x_sentence1,
                            X_sentence2=x_sentence2, Y_batch=y_batch,
                            loss_fn=loss_fn, optimizer=optimizer)
        if i % 100 == 0:
            perc = round(100*i/float(datalen), 2)
            print(f'Batch {i}/{datalen} ({perc}%), Epoch: {epoch_idx}   Model {name}')
            print(formatTime())
            print('Loss {}'.format(loss / (i+1)))
    loss /= (i+1)  # divide loss by number of batches for consistency
    return loss

def train_epoch_e(name: str,
                network: torch.nn.Module,
                dataloader: DataLoader,
                loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
                optimizer: torch.optim.Optimizer,
                device: str,
                epoch_idx: int) -> float:
    datalen = len(dataloader)
    loss = 0.
    for i, (x_sentence1, x_sentence2, y_batch) in enumerate(dataloader):
        x_sentence1 = list(map(lambda d: d.to(device), x_sentence1))
        x_sentence2 = list(map(lambda d: d.to(device), x_sentence2))
        y_batch = y_batch.to(device, dtype=torch.float32)
        loss += train_batch_e(network=network, X_sentence1=x_sentence1,
                            X_sentence2=x_sentence2, Y_batch=y_batch,
                            loss_fn=loss_fn, optimizer=optimizer)
        if i % 100 == 0:
            perc = round(100*i/float(datalen), 2)
            print(f'Batch {i}/{datalen} ({perc}%), Epoch: {epoch_idx}   Model {name}')
            print(formatTime())
            print('Loss {}'.format(loss / (i+1)))
    loss /= (i+1)  # divide loss by number of batches for consistency
    return loss


def train_epochs(name: str,
                 network: torch.nn.Module,
                 dataloader: DataLoader,
                 dataset_train: Dataset,
                 dataset_dev: Dataset,
                 dataset_test: Dataset,
                 loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
                 optimizer: torch.optim.Optimizer,
                 device: str,
                 num_epochs: int) -> Tuple[List[float], List[float], List[float]]:
    epoch_losses = []
    trains = [evaluate(network, dataset_train)]
    evals = [evaluate(network, dataset_dev)]
    tests = [evaluate(network, dataset_test)]
    for i in range(num_epochs):
        epoch_losses.append(train_epoch(name, network, dataloader, loss_fn, optimizer, device, i+1))
        train = evaluate(network, dataset_train)
        trains.append(train)
        eval = evaluate(network, dataset_dev)
        evals.append(eval)
        test = evaluate(network, dataset_test)
        tests.append(test)
        print(f"Train: {train}, Eval: {eval},    Test: {test}")
    return epoch_losses, trains, evals, tests

def train_epochs_e(name: str,
                  network: torch.nn.Module,
                  dataloader: DataLoader,
                  dataset_train: Dataset,
                  dataset_dev: Dataset,
                  dataset_test: Dataset,
                  loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
                  optimizer: torch.optim.Optimizer,
                  device: str,
                  num_epochs: int) -> Tuple[List[float], List[float], List[float]]:
    epoch_losses = []
    trains = [evaluate_e(network, dataset_train)]
    evals = [evaluate_e(network, dataset_dev)]
    tests = [evaluate_e(network, dataset_test)]
    for i in range(num_epochs):
        epoch_losses.append(train_epoch_e(name, network, dataloader, loss_fn, optimizer, device, i+1))
        train = evaluate_e(network, dataset_train)
        trains.append(train)
        eval = evaluate_e(network, dataset_dev)
        evals.append(eval)
        test = evaluate_e(network, dataset_test)
        tests.append(test)
        print(f"Train: {train}, Eval: {eval},    Test: {test}")
    return epoch_losses, trains, evals, tests

def evaluate(network: torch.nn.Module,
             dataset: Dataset):
    network.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for (s1, s2, l) in tqdm(dataset):
            model_pred = network(s1, s2)
            pred = torch.dot(torch.arange(1, 6).float(), torch.exp(model_pred))
            preds.append(pred.item())
            trues.append(l.item())
    return pearson(preds, trues)


def evaluate_e(network: torch.nn.Module,
               dataset: Dataset):
    network.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for (s1, s2, l) in tqdm(dataset):
            pred = torch.argmax(network(s1, s2))
            preds.append(pred.item())
            trues.append(l.item())
        correct = [p for i, p in enumerate(preds) if p == trues[i]]
    return len(correct)/float(len(trues))


def train_batch_dot(network: torch.nn.Module,
                    X_sentence1: SentenceData,
                    X_sentence2: SentenceData,
                    Y_batch: LongTensor,
                    loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
                    optimizer: torch.optim.Optimizer) -> float:
    network.train()
    prediction_batch = network(X_sentence1, X_sentence2)  # forward pass
    # Y_val = Y_batch.item()
    # pred_val = prediction_batch.item()
    # Y_val = map_label_to_target(Y_batch.item(), 5)
    # pred_val = map_label_to_target((((prediction_batch.item()*0.5)+0.5)*4)+1, 5)
    # batch_loss = loss_fn(pred_val, Y_val)  # loss calculation

    batch_loss = loss_fn(prediction_batch, (Y_batch[0]-3)/2.)  # loss calculation
    batch_loss.backward()
    # try:
        # batch_loss.backward()  # gradient computation
    # except RuntimeError:
    #     print(X_sentence1)
    #     print(X_sentence2)
    optimizer.step()  # back-propagation
    optimizer.zero_grad()  # gradient reset
    return batch_loss.item()


def train_epoch_dot(name: str,
                    network: torch.nn.Module,
                    dataloader: DataLoader,
                    loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
                    optimizer: torch.optim.Optimizer,
                    device: str,
                    epoch_idx: int) -> float:
    datalen = len(dataloader)
    loss = 0.
    for i, (x_sentence1, x_sentence2, y_batch) in enumerate(dataloader):
        x_sentence1 = list(map(lambda d: d.to(device), x_sentence1))
        x_sentence2 = list(map(lambda d: d.to(device), x_sentence2))
        y_batch = y_batch.to(device, dtype=torch.float32)
        loss += train_batch_dot(network=network, X_sentence1=x_sentence1,
                                X_sentence2=x_sentence2, Y_batch=y_batch,
                                loss_fn=loss_fn, optimizer=optimizer)
        if i % 100 == 0:
            perc = round(100*i/float(datalen), 2)
            print(f'Batch {i}/{datalen} ({perc}%), Epoch: {epoch_idx}   Model: {name}')
            print(formatTime())
            print('Loss {}'.format(loss / (i+1)))
    loss /= (i+1)  # divide loss by number of batches for consistency
    return loss


def evaluate_dot(network: torch.nn.Module,
                 dataset: Dataset):
    network.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for (s1, s2, l) in tqdm(dataset):
            model_pred = network(s1, s2)   # will be between 0 and 1
            # pred = min(torch.tensor([5.]), model_pred)
            # pred = torch.dot(torch.arange(1, 6).float(), torch.exp(model_pred))
            preds.append(model_pred.item())
            trues.append(l.item())
    return pearson(preds, trues)


def train_epochs_dot(name: str,
                     network: torch.nn.Module,
                     dataloader: DataLoader,
                     dataset_train: Dataset,
                     dataset_dev: Dataset,
                     dataset_test: Dataset,
                     loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
                     optimizer: torch.optim.Optimizer,
                     device: str,
                     num_epochs: int) -> Tuple[List[float], List[float], List[float]]:
    epoch_losses = []
    trains = [evaluate_dot(network, dataset_train)]
    evals = [evaluate_dot(network, dataset_dev)]
    tests = [evaluate_dot(network, dataset_test)]
    for i in range(num_epochs):
        epoch_losses.append(train_epoch_dot(name, network, dataloader, loss_fn, optimizer, device, i+1))
        train = evaluate_dot(network, dataset_train)
        trains.append(train)
        eval = evaluate_dot(network, dataset_dev)
        evals.append(eval)
        test = evaluate_dot(network, dataset_test)
        tests.append(test)
        print(f"Train: {train}, Eval: {eval},    Test: {test}")
    return epoch_losses, trains, evals, tests
