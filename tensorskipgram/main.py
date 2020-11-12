import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorskipgram.trainer import train_epoch
from tensorskipgram.data.preprocessing import Preprocessor
from tensorskipgram.config import subj_data_fn, obj_data_fn, noun_space_fn
from tensorskipgram.data.dataset import create_noun_matrix, MatrixSkipgramDataset
from tensorskipgram.models.model import MatrixSkipgram
from tensorskipgram.data.main_preprocessing import main as preprocessing_main
from tensorskipgram.data.main_evaluation import main as evaluation_main
from tensorskipgram.data.main_training import main as training_main

folder = '/import/gijs-shared/gijs'
data_folder = os.path.join(folder, 'verb_data')
verb_dict_fn = os.path.join(folder, 'verb_data/verb_counts_all_corpus_verbs_dict.p')
verbs_fn = os.path.join(folder, 'verb_data/sick_verbs_full.txt')
preproc_fn = os.path.join(folder, 'verb_data/preproc_sick_verbcounts.p')
model_path = os.path.join(folder, 'verb_data/matrixskipgram_subj')
# model_path = os.path.join(folder, 'verb_data/matrixskipgram_obj')


def main(bs=1, lr=0.005, ne=5):
    my_preproc = Preprocessor(preproc_fn, noun_space_fn, verb_dict_fn, verbs_fn)
    subj_i2w = my_preproc.preproc['subj']['i2w']
    obj_i2w = my_preproc.preproc['obj']['i2w']
    verb_i2v = my_preproc.preproc['verb']['i2v']
    lower2upper = my_preproc.preproc['l2u']

    # noun_vocab_size = len(subj_i2w)
    # context_vocab_size = len(obj_i2w)
    noun_vocab_size = len(obj_i2w)
    context_vocab_size = len(subj_i2w)
    functor_vocab_size = len(verb_i2v)
    nounMatrix = createNounMatrix(noun_space_fn, obj_i2w, lower2upper)
    # nounMatrix = createNounMatrix(subj_i2w, lower2upper)
    nounMatrix = torch.tensor(nounMatrix, dtype=torch.float32)


    # print("Preparing data loader...")
    # subj_dataset = MatrixSkipgramDataset(subj_data_fn, arg='subject', negk=5)
    # subj_dataloader6 = DataLoader(subj_dataset, shuffle=True, batch_size=1)
    print("Preparing data loader...")
    subj_dataset = MatrixSkipgramDataset(subj_data_fn, arg='subject', negk=5)
    subj_dataloader = DataLoader(subj_dataset, shuffle=True, batch_size=bs)
    # print("Preparing data loader...")
    # obj_dataset = MatrixSkipgramDataset(obj_data_fn, arg='object', negk=5)
    # obj_dataloader = DataLoader(obj_dataset, shuffle=True, batch_size=bs)

    print("Training model...")

    subj_matskipgram_modelGPU = MatrixSkipgram(noun_vocab_size=noun_vocab_size,
                                              functor_vocab_size=functor_vocab_size,
                                              context_vocab_size=context_vocab_size,
                                              embed_size=100, nounMatrix=nounMatrix)

    # obj_matskipgram_modelGPU.to('cuda')
    subj_matskipgram_modelGPU.to(1)
    optGPU = torch.optim.Adam(subj_matskipgram_modelGPU.parameters(), lr=lr)
    loss_fnGPU = torch.nn.BCEWithLogitsLoss()

    NUM_EPOCHS = ne
    epoch_losses = []
    for i in range(NUM_EPOCHS):
        epoch_loss = train_epoch(subj_matskipgram_modelGPU, subj_dataloader,
                                 loss_fnGPU, optGPU, device=1, epoch_idx=i+1)
                                 # loss_fnGPU, optGPU, device='cuda', epoch_idx=i+1)
        # epoch_loss = train_epoch(subj_matskipgram_modelGPU, obj_dataloader6,
                                 # loss_fnGPU, optGPU, device='cuda', epoch_idx=i+1)
        epoch_losses.append(epoch_loss)
        print("Saving model...")
        e = i+1
        torch.save(subj_matskipgram_modelGPU, model_path+f'_bs={bs}_lr={lr}_epoch{e}.p')
        print("Done saving model, ready for another epoch!")

    return epoch_losses

# subj_i2w = my_preproc.preproc['subj']['i2w']
# obj_i2w = my_preproc.preproc['obj']['i2w']
# verb_i2v = my_preproc.preproc['verb']['i2v']
# lower2upper = my_preproc.preproc['l2u']
#
# noun_vocab_size = len(subj_i2w)
# context_vocab_size = len(obj_i2w)
# # noun_vocab_size = len(obj_i2w)
# # context_vocab_size = len(subj_i2w)
# functor_vocab_size = len(verb_i2v)
# # nounMatrix = createNounMatrix(obj_i2w, lower2upper)
# nounMatrix = createNounMatrix(subj_i2w, lower2upper)
# nounMatrix = torch.tensor(nounMatrix, dtype=torch.float32)
#
#
# # print("Preparing data loader...")
# # subj_dataset = MatrixSkipgramDataset(subj_data_fn, arg='subject', negk=5)
# # subj_dataloader6 = DataLoader(subj_dataset, shuffle=True, batch_size=1)
# print("Preparing data loader...")
# # subj_dataset = MatrixSkipgramDataset(subj_data_fn, arg='subject', negk=5)
# # subj_dataloader1 = DataLoader(subj_dataset, shuffle=True, batch_size=1)
# # print("Preparing data loader...")
# obj_dataset = MatrixSkipgramDataset(obj_data_fn, arg='object', negk=5)
# obj_dataloader1 = DataLoader(obj_dataset, shuffle=True, batch_size=1)
# obj_dataloader10 = DataLoader(obj_dataset, shuffle=True, batch_size=10)
#
# print("Training model...")
#
# obj_matskipgram_modelGPU = MatrixSkipgram(noun_vocab_size=noun_vocab_size,
#                                           functor_vocab_size=functor_vocab_size,
#                                           context_vocab_size=context_vocab_size,
#                                           embed_size=100, nounMatrix=nounMatrix)
#
# obj_matskipgram_modelGPU.to('cuda')
# optGPU = torch.optim.Adam(obj_matskipgram_modelGPU.parameters(), lr=0.005)
# loss_fnGPU = torch.nn.BCEWithLogitsLoss()
spaceInFN = '/import/gijs-shared/gijs/spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt'


def final_main():
    preprocessing_main()
    training_main()
    evaluation_main()
