import time
import torch
from tensorskipgram.models.model import *
from tensorskipgram.data.dataset import *
from torch import FloatTensor, LongTensor
from typing import Tuple, List, Callable, Optional
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def formatTime():
    return time.strftime('%X %x %Z')
    # timeNow = time()
    # return ('%d:%02d:%02d' % (timeNow // 3600, (timeNow % 3600) // 60, timeNow % 60))


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
                device: str) -> float:
    loss = 0.
    for i, (x_args, x_funcs, x_contexts, y_batch) in enumerate(dataloader):
        x_args = x_args.to(device).view(-1)  # convert back to your chosen device
        x_funcs = x_funcs.to(device).view(-1)
        x_contexts = x_contexts.to(device).view(-1)
        y_batch = y_batch.to(device, dtype=torch.float32).view(-1)
        loss += train_batch(network=network, X_args=x_args, X_funcs=x_funcs,
                            X_contexts=x_contexts, Y_batch=y_batch,
                            loss_fn=loss_fn, optimizer=optimizer)
        if i % 100 == 0:
            print('Batch {}'.format(i))
            print(formatTime())
            print('Loss {}'.format(loss / (i+1)))
    loss /= (i+1) # divide loss by number of batches for consistency
    return loss


# subj_matskipgram_modelCPU = MatrixSkipgram(noun_vocab_size=noun_vocab_size,
#                                            functor_vocab_size=317,
#                                            context_vocab_size=context_vocab_size,
#                                            embed_size=100, nounMatrix=torch.tensor(nounMatrix))
#
# subj_matskipgram_modelCPU.to('cpu')
# optCPU = torch.optim.Adam(subj_matskipgram_modelCPU.parameters())
# loss_fnCPU = torch.nn.BCEWithLogitsLoss()
#
# subj_matskipgram_modelGPU = MatrixSkipgram(noun_vocab_size=noun_vocab_size,
#                                            functor_vocab_size=317,
#                                            context_vocab_size=context_vocab_size,
#                                            embed_size=100, nounMatrix=torch.tensor(nounMatrix))
# subj_matskipgram_modelGPU.to('cuda')
# optGPU = torch.optim.Adam(subj_matskipgram_modelGPU.parameters(), lr=0.005)
# loss_fnGPU = torch.nn.BCEWithLogitsLoss()
#
# NUM_EPOCHS = 5

# def timeCPUrun(model, loader, loss_fn, opt):
#     model.to('cpu')
#     print(formatTime())
#     epoch_loss = train_epoch(model, loader, loss_fn, opt, device='cpu')
#     return epoch_loss
#
#
# def timeGPUrun(model, loader, loss_fn, opt):
#     model.to('cuda')
#     print(formatTime())
#     epoch_loss = train_epoch(model, loader, loss_fn, opt, device='cuda')
#     return epoch_loss

# timeCPUrun(subj_matskipgram_modelCPU, subj_dataloader6, loss_fnCPU, optCPU)
# timeGPUrun(subj_matskipgram_modelGPU, subj_dataloader6, loss_fnGPU, optGPU)
# little test
# testData = torch.tensor([1,2]), torch.tensor([1,2]), torch.tensor([1,2]), torch.tensor([1, 0], dtype=torch.float32)
# testPred = subj_matskipgram_modelCPU(testData[0],testData[1],testData[2])
