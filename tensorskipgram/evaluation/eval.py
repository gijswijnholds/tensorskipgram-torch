import os
import torch
from tensorskipgram.data.preprocessing import Preprocessor
from tensorskipgram.evaluation.data import SICKPreprocessor, SICKDataset
from tensorskipgram.evaluation.model import SentenceEmbedderSimilarity
from tensorskipgram.evaluation.trainer import *
from torch.utils.data import DataLoader
task_fn = '/homes/gjw30/ExpCode/compdisteval/experiment_data/SICK/SICK.txt'
folder = '/import/gijs-shared/gijs'
data_folder = os.path.join(folder, 'verb_data')
space_fn = os.path.join(folder, 'spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt')
verb_dict_fn = os.path.join(folder, 'verb_data/verb_counts_all_corpus_verbs_dict.p')
verbs_fn = os.path.join(folder, 'verb_data/sick_verbs_full.txt')
preproc_fn = os.path.join(folder, 'verb_data/preproc_sick_verbcounts.p')
data_fn = os.path.join(folder, 'verb_data/sick_dataset.p')
sick_data_fn = os.path.join(folder, 'sick_data.p')

my_preproc = Preprocessor(preproc_fn, space_fn, verb_dict_fn, verbs_fn)
lower2upper = my_preproc.preproc['l2u']
allnouns = set(lower2upper.keys())
allverbs = set(my_preproc.preproc['verb']['i2v'])


sick_preproc = SICKPreprocessor(task_fn, sick_data_fn, allnouns, allverbs)
sick_dataset_train = SICKDataset(data_fn=data_fn, setting='train')
sick_dataset_dev = SICKDataset(data_fn=data_fn, setting='dev')
sick_dataset_test = SICKDataset(data_fn=data_fn, setting='test')
noun_matrix = sick_preproc.create_noun_matrix(space_fn, lower2upper)
verb_subj_cube = sick_preproc.create_verb_cube('subj', my_preproc.preproc['verb']['v2i'], folder)
verb_subj_cube = verb_subj_cube.reshape(verb_subj_cube.shape[0], 10000)
verb_obj_cube = sick_preproc.create_verb_cube('obj', my_preproc.preproc['verb']['v2i'], folder)
verb_obj_cube = verb_obj_cube.reshape(verb_obj_cube.shape[0], 10000)

embedder_model = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 100)
embedder_model.subj_verb_embedding.weight.requires_grad = True
embedder_model.obj_verb_embedding.weight.requires_grad = True
loss_fn = torch.nn.KLDivLoss()
opt = torch.optim.Adam(embedder_model.parameters(), lr=0.005)
sick_dataloader = DataLoader(sick_dataset_train, shuffle=True, batch_size=1)

num_epochs = 10
epoch_losses = []

evals = [evaluate(embedder_model, sick_dataset_dev)]
tests = [evaluate(embedder_model, sick_dataset_test)]

for i in range(num_epochs):
    epoch_losses.append(train_epoch(embedder_model, sick_dataloader, loss_fn, opt, 'cpu', i+1))
    eval = evaluate(embedder_model, sick_dataset_dev)
    evals.append(eval)
    test = evaluate(embedder_model, sick_dataset_test)
    tests.append(test)
    print(f"Eval: {eval},    Test: {test}")
