import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorskipgram.trainer import train_epoch
from tensorskipgram.data.preprocessing import Preprocessor
from tensorskipgram.data.prepare_data import subj_data_fn
from tensorskipgram.data.dataset import MatrixSkipgramDataset
from tensorskipgram.models.model import MatrixSkipgram


def createNounMatrix(index2word, lower2upper):
    spaceInFN = '/import/gijs-shared/gijs/spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt'
    spaceFile = open(spaceInFN, 'r')
    space = {}
    print("Loading vectors...")
    for ln in spaceFile.readlines():
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


folder = '/import/gijs-shared/gijs'
data_folder = os.path.join(folder, 'verb_data')
space_fn = os.path.join(folder, 'spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt')
verb_dict_fn = os.path.join(folder, 'verb_data/verb_counts_all_corpus_verbs_dict.p')
verbs_fn = os.path.join(folder, 'verb_data/sick_verbs_full.txt')
preproc_fn = os.path.join(folder, 'verb_data/preproc_sick_verbcounts.p')
model_path = os.path.join(folder, 'verb_data/matrixskipgram_subj')


def main():
    my_preproc = Preprocessor(preproc_fn, space_fn, verb_dict_fn, verbs_fn)
    subj_i2w = my_preproc.preproc['subj']['i2w']
    obj_i2w = my_preproc.preproc['obj']['i2w']
    verb_i2v = my_preproc.preproc['verb']['i2v']
    lower2upper = my_preproc.preproc['l2u']

    noun_vocab_size = len(obj_i2w)
    context_vocab_size = len(subj_i2w)
    functor_vocab_size = len(verb_i2v)
    nounMatrix = createNounMatrix(obj_i2w, lower2upper)
    nounMatrix = torch.tensor(nounMatrix, dtype=torch.float32)

    print("Preparing data loader...")
    subj_dataset = MatrixSkipgramDataset(subj_data_fn, arg='subject', negk=5)
    subj_dataloader6 = DataLoader(subj_dataset, shuffle=True, batch_size=1)

    print("Training model...")

    subj_matskipgram_modelGPU = MatrixSkipgram(noun_vocab_size=noun_vocab_size,
                                               functor_vocab_size=functor_vocab_size,
                                               context_vocab_size=context_vocab_size,
                                               embed_size=100, nounMatrix=nounMatrix)

    subj_matskipgram_modelGPU.to('cuda')
    optGPU = torch.optim.Adam(subj_matskipgram_modelGPU.parameters(), lr=0.005)
    loss_fnGPU = torch.nn.BCEWithLogitsLoss()

    NUM_EPOCHS = 5
    epoch_losses = []
    for i in range(NUM_EPOCHS):
        epoch_loss = train_epoch(subj_matskipgram_modelGPU, subj_dataloader6,
                                 loss_fnGPU, optGPU, device='cuda')
        epoch_losses.append(epoch_loss)
        print("Saving model...")
        torch.save(subj_matskipgram_modelGPU, model_path+f'_epoch{i}.p')
        print("Done saving model, ready for another epoch!")

    return epoch_losses
