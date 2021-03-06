import os
import torch
from tqdm import tqdm
import numpy as np
from typing import Tuple, List, Dict
from torch import LongTensor
from torch.utils.data import Dataset
from tensorskipgram.preprocessing.util import load_obj_fn


def create_noun_matrix(space_fn: str, index2word: List[str], lower2upper: Dict):
    spaceFile = open(space_fn, 'r')
    space = {}
    print("Loading vectors...")
    for ln in tqdm(spaceFile.readlines()):
        ln = ln.strip().split()
        key = ln[0]
        vec = np.array([float(b) for b in ln[1:]])
        space[key] = vec
    nounMatrix = np.zeros((len(index2word), 100))
    print("Filling noun matrix...")
    for i in range(len(index2word)):
        nounMatrix[i] = space[lower2upper[index2word[i]]]
    assert np.count_nonzero(nounMatrix) == np.prod(nounMatrix.shape)
    print("Done filling noun matrix!")
    return nounMatrix


class SkipgramDataset(Dataset):
    def __init__(self, data_filename: str) -> None:
        assert data_filename.endswith('.npy')
        print("Loading data...")
        data = np.load(data_filename, encoding='latin1').T
        self.X_targets, self.X_contexts, self.Y = data

    def __len__(self) -> int:
        return len(self.X_targets)

    def __getitem__(self, idx: int) -> Tuple[LongTensor, LongTensor, LongTensor]:
        return self.X_targets[idx], self.X_contexts[idx], self.Y[idx]


class MatrixSkipgramDataset(Dataset):
    def __init__(self, data_filename: str, arg: str, negk: int) -> None:
        if not os.path.exists(data_filename):
            print("Data not found, please run a data creator before training!")
        if data_filename.endswith('.npy'):
            print("Loading data...")
            data = np.load(data_filename, encoding='latin1', allow_pickle=True).T
        elif data_filename.endswith('.p'):
            print("Loading data...")
            data = load_obj_fn(data_filename).T
        print("Preparing data in PyTorch format...")
        assert arg in ['subj', 'obj']
        if arg == 'subj':  # objects are fixed nouns
            self.X_contexts, self.X_funcs, self.X_args, self.Y = \
             list(map(lambda x: torch.tensor(list(x), dtype=torch.long), tqdm(data)))
        if arg == 'obj':  # subjects are fixed nouns (X_args)
            self.X_args, self.X_funcs, self.X_contexts, self.Y = \
             list(map(lambda x: torch.tensor(list(x), dtype=torch.long), tqdm(data)))
        self.batchidx = negk+1
        print("Done preparing data!")

    def __len__(self) -> int:
        return int(len(self.X_args) / self.batchidx)

    def __getitem__(self, idx: int) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor]:
        assert idx < len(self)
        lidx = idx*self.batchidx
        ridx = (idx+1)*self.batchidx
        return self.X_args[lidx:ridx], self.X_funcs[lidx:ridx], self.X_contexts[lidx:ridx], self.Y[lidx:ridx]
