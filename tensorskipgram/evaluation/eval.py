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


embedder_model_flex_verbs = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50)
embedder_model_flex_verbs.subj_verb_embedding.weight.requires_grad = True
embedder_model_flex_verbs.obj_verb_embedding.weight.requires_grad = True

embedder_model_flex_nouns = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50)
embedder_model_flex_nouns.noun_embedding.weight.requires_grad = True

embedder_model_fix_all = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50)

embedder_model_flex_all = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50)
embedder_model_flex_all.noun_embedding.weight.requires_grad = True
embedder_model_flex_all.subj_verb_embedding.weight.requires_grad = True
embedder_model_flex_all.obj_verb_embedding.weight.requires_grad = True

loss_fn = torch.nn.KLDivLoss()
opt_flex_verbs = torch.optim.Adam(embedder_model_flex_verbs.parameters(), lr=0.005)
opt_flex_nouns = torch.optim.Adam(embedder_model_flex_nouns.parameters(), lr=0.005)
opt_flex_all = torch.optim.Adam(embedder_model_flex_all.parameters(), lr=0.005)
opt_fix_all = torch.optim.Adam(embedder_model_fix_all.parameters(), lr=0.005)

sick_dataloader = DataLoader(sick_dataset_train, shuffle=True, batch_size=1)

model_configs = [('fix_all', embedder_model_fix_all, opt_fix_all),
                 ('flex_nouns', embedder_model_flex_nouns, opt_flex_nouns),
                 ('flex_verbs', embedder_model_flex_verbs, opt_flex_verbs),
                 ('flex_all', embedder_model_flex_all, opt_flex_all)]
num_epochs = 5

model_results = {'fix_all': (), 'flex_nouns': (), 'flex_verbs': (), 'flex_all': ()}

for (name, model, opt) in model_configs:
    epoch_losses = []
    evals = [evaluate(model, sick_dataset_dev)]
    tests = [evaluate(model, sick_dataset_test)]
    for i in range(num_epochs):
        epoch_losses.append(train_epoch(model, sick_dataloader, loss_fn, opt, 'cpu', i+1))
        eval = evaluate(model, sick_dataset_dev)
        evals.append(eval)
        test = evaluate(model, sick_dataset_test)
        tests.append(test)
        print(f"Eval: {eval},    Test: {test}")
    model_results[name] = (epoch_losses, evals, tests)

evals = [evaluate(embedder_model, sick_dataset_dev)]
tests = [evaluate(embedder_model, sick_dataset_test)]

for i in range(num_epochs):
    epoch_losses.append(train_epoch(embedder_model, sick_dataloader, loss_fn, opt, 'cpu', i+1))
    eval = evaluate(embedder_model, sick_dataset_dev)
    evals.append(eval)
    test = evaluate(embedder_model, sick_dataset_test)
    tests.append(test)
    print(f"Eval: {eval},    Test: {test}")
