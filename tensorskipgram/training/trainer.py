import time
import torch
from torch import FloatTensor, LongTensor
from typing import Callable
from torch.utils.data import DataLoader


def format_time():
    return time.strftime('%X %x %Z')


def train_batch(network: torch.nn.Module,
                X_args: LongTensor,
                X_funcs: LongTensor,
                X_contexts: LongTensor,
                Y_batch: LongTensor,
                loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
                optimizer: torch.optim.Optimizer) -> float:
    network.train()
    prediction_batch = network(X_args, X_funcs, X_contexts)  # forward pass
    batch_loss = loss_fn(prediction_batch, Y_batch)  # loss calculation
    batch_loss.backward()  # gradient computation
    optimizer.step()  # back-propagation
    optimizer.zero_grad()  # gradient reset
    return batch_loss.item()


def train_epoch(network: torch.nn.Module,
                dataloader: DataLoader,
                loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
                optimizer: torch.optim.Optimizer,
                device: str,
                epoch_idx: int) -> float:
    datalen = len(dataloader)
    loss = 0.
    for i, (x_args, x_funcs, x_contexts, y_batch) in enumerate(dataloader):
        x_args = x_args.to(device)  # convert back to your chosen device
        x_funcs = x_funcs.to(device)
        x_contexts = x_contexts.to(device)
        y_batch = y_batch.to(device, dtype=torch.float32)
        loss += train_batch(network=network, X_args=x_args, X_funcs=x_funcs,
                            X_contexts=x_contexts, Y_batch=y_batch,
                            loss_fn=loss_fn, optimizer=optimizer)
        if i % 100 == 0:
            perc = round(100*i/float(datalen), 2)
            print(f'Batch {i}/{datalen} ({perc}%), Epoch: {epoch_idx}')
            print(format_time())
            print('Loss {}'.format(loss / (i+1)))
    loss /= (i+1)  # divide loss by number of batches for consistency
    return loss
