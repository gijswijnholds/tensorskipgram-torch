import numpy as np
from typing import Tuple, List
from torch import LongTensor

class SkipgramDataset(Dataset):
    def __init__(self, data_filename: str) -> None:
        assert data_filename.endswith('.npy')
        print("Loading data...")
        data = np.load(data_filename, encoding='latin1').T
        self.X_targets, self.X_contexts, self.Y = data

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[LongTensor, LongTensor, LongTensor]:
        return self.X_targets[idx], self.X_contexts[idx], self.Y[idx]


class MatrixSkipgramDataset(Dataset):
    def __init__(self, data_filename: str, arg: str) -> None:
        assert data_filename.endswith('.npy')
        assert arg in ['subject', 'object']
        print("Loading data...")
        data = np.load(data_filename, encoding='latin1').T
        if arg == 'subject':  # objects are fixed nouns
            self.X_contexts, self.X_funcs, self.X_args, self.Y = data
        if arg == 'object':  # subjects are fixed nouns (X_args)
            self.X_args, self.X_funcs, self.X_contexts, self.Y = data

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor]:
        return self.X_args[idx], self.X_funcs[idx], self.X_contexts[idx], self.Y[idx]


sickDataFN = ""
