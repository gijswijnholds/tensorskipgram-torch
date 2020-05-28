import torch
import pickle
import numpy as np
from typing import Tuple, List
from torch import LongTensor
from torch.utils.data import Dataset, DataLoader
from tensorskipgram.data.util import load_obj_fn


def createNounMatrix(index2word):
    spaceInFN = '/import/gijs-shared/gijs/spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt'
    spaceFile = open(spaceInFN, 'r')
    space = {}
    for ln in spaceFile.readlines():
        ln = ln.strip().split()
        key = ln[0]
        vec = np.array([float(b) for b in ln[1:]])
        space[key] = vec
    lower2upper = load_obj_fn('/import/gijs-shared/gijs/skipprob_data/lower2upper.pkl')

    nounMatrix = np.zeros((len(index2word), 100))
    print("Filling noun matrix...")
    for i in range(len(index2word)):
        nounMatrix[i] = space[lower2upper[index2word[i]]]
    assert np.count_nonzero(nounMatrix) == np.prod(nounMatrix.shape)
    print("Done filling noun matrix!")
    return nounMatrix


def loadArgAnalysers():
    subj_preprocFileName = '/import/gijs-shared/gijs/skipprob_data/preproc_sick_subj.pkl'
    obj_preprocFileName = '/import/gijs-shared/gijs/skipprob_data/preproc_sick_obj.pkl'
    print("Loading preprocs...")
    subj_preproc = load_obj_fn(subj_preprocFileName)
    obj_preproc = load_obj_fn(obj_preprocFileName)

    subj_i2w, subj_w2i, subj_i2c, subj_i2ns = (subj_preproc['index2word'],
                                               subj_preproc['word2index'],
                                               subj_preproc['index2count'],
                                               subj_preproc['index2negsample'])

    obj_i2w, obj_w2i, obj_i2c, obj_i2ns = (obj_preproc['index2word'],
                                           obj_preproc['word2index'],
                                           obj_preproc['index2count'],
                                           obj_preproc['index2negsample'])
    return (subj_i2w, subj_w2i, subj_i2c, subj_i2ns,
            obj_i2w, obj_w2i, obj_i2c, obj_i2ns)


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
        assert data_filename.endswith('.npy')
        assert arg in ['subject', 'object']
        print("Loading data...")
        data = np.load(data_filename, encoding='latin1').T
        if arg == 'subject':  # objects are fixed nouns
            self.X_contexts, self.X_funcs, self.X_args, self.Y = list(map(lambda x: torch.tensor(list(x), dtype=torch.long), data))
        if arg == 'object':  # subjects are fixed nouns (X_args)
            self.X_args, self.X_funcs, self.X_contexts, self.Y = list(map(lambda x: torch.tensor(list(x), dtype=torch.long), data))
        self.batchidx = negk+1

    def __len__(self) -> int:
        return int(len(self.X_args) / self.batchidx)

    def __getitem__(self, idx: int) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor]:
        lidx = idx*self.batchidx
        ridx = (idx+1)*self.batchidx
        return self.X_args[lidx:ridx], self.X_funcs[lidx:ridx], self.X_contexts[lidx:ridx], self.Y[lidx:ridx]



sick_data_fn_subj = "/import/gijs-shared/gijs/skipprob_data/training_data_sick_subject/train_data_proper_asym_ns=5.npy"
sick_data_fn_obj = "/import/gijs-shared/gijs/skipprob_data/training_data_sick_object/train_data_proper_asym_ns=5.npy"

subjDataset = MatrixSkipgramDataset(sick_data_fn_subj, arg='subject', negk=5)
# objDataset = MatrixSkipgramDataset(sick_data_fn_obj, arg='object')

subj_dataloader6 = DataLoader(subjDataset,
                              shuffle=True,
                              batch_size=1)

subj_dataloader48 = DataLoader(subjDataset,
                               shuffle=True,
                               batch_size=8)

subj_dataloader96 = DataLoader(subjDataset,
                               shuffle=True,
                               batch_size=16)

subj_dataloader96 = DataLoader(subjDataset,
                               shuffle=True,
                               batch_size=64)
print("Loading analysers...")
(subj_i2w, subj_w2i, subj_i2c, subj_i2ns,
 obj_i2w, obj_w2i, obj_i2c, obj_i2ns) = \
    loadArgAnalysers()

noun_vocab_size = len(obj_i2w)
context_vocab_size = len(subj_i2w)
nounMatrix = createNounMatrix(obj_i2w)
nounMatrix = torch.tensor(nounMatrix, dtype=torch.float32)
