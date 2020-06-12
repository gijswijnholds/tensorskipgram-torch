import os
import torch
from tensorskipgram.data.preprocessing import Preprocessor
from tensorskipgram.evaluation.data import SICKPreprocessor, SICKDataset, SICKDatasetNouns
from tensorskipgram.evaluation.model import SentenceEmbedderSimilarity, SentenceEmbedderSimilarityDot, VecSimDot
from tensorskipgram.evaluation.trainer import *
from tensorskipgram.data.util import load_obj_fn, dump_obj_fn
from torch.utils.data import DataLoader
task_fn = '/homes/gjw30/ExpCode/compdisteval/experiment_data/SICK/SICK.txt'
folder = '/import/gijs-shared/gijs'
data_folder = os.path.join(folder, 'verb_data')
space_fn = os.path.join(folder, 'spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt')
verb_dict_fn = os.path.join(folder, 'verb_data/verb_counts_all_corpus_verbs_dict.p')
verbs_fn = os.path.join(folder, 'verb_data/sick_verbs_full.txt')
preproc_fn = os.path.join(folder, 'verb_data/preproc_sick_verbcounts.p')
data_fn = os.path.join(folder, 'verb_data/sick_dataset.p')
data_fn_nouns = os.path.join(folder, 'verb_data/sick_dataset_nouns.p')
sick_data_fn = os.path.join(folder, 'sick_data.p')
model_data_fn = os.path.join(folder, 'verb_data/sick_model_data.p')
sick_dataset_train = SICKDataset(data_fn=data_fn, setting='train')
sick_dataset_dev = SICKDataset(data_fn=data_fn, setting='dev')
sick_dataset_test = SICKDataset(data_fn=data_fn, setting='test')

sick_noun_dataset_train = SICKDatasetNouns(data_fn=data_fn_nouns, setting='train')
sick_noun_dataset_dev = SICKDatasetNouns(data_fn=data_fn_nouns, setting='dev')
sick_noun_dataset_test = SICKDatasetNouns(data_fn=data_fn_nouns, setting='test')

# vec_mats = {'noun_matrix': noun_matrix, 'verb_subj_cube': verb_subj_cube, 'verb_obj_cube': verb_obj_cube}
# dump_obj_fn(vec_mats, model_data_fn)
vec_mats = load_obj_fn(model_data_fn)
noun_matrix = vec_mats['noun_matrix']
verb_subj_cube = vec_mats['verb_subj_cube']
verb_obj_cube = vec_mats['verb_obj_cube']

embedder_model_flex_verbs = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=False, flex_verbs=True)
embedder_model_flex_nouns = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=True, flex_verbs=False)
embedder_model_fix_all = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=False, flex_verbs=False)
embedder_model_flex_all = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=True, flex_verbs=True)
# dot_embedder_model_flex_verbs = SentenceEmbedderSimilarityDot(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=False, flex_verbs=True)
# dot_embedder_model_fix_all = SentenceEmbedderSimilarityDot(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=False, flex_verbs=False)
dot_embedder_model_flex_nouns = SentenceEmbedderSimilarityDot(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=True, flex_verbs=False)
dot_embedder_model_flex_all = SentenceEmbedderSimilarityDot(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=True, flex_verbs=True)

# static_model = VecSimDot(noun_matrix, 100, flex_nouns=False)
# evaluate_dot(static_model, sick_noun_dataset_train)
# evaluate_dot(static_model, sick_noun_dataset_dev)
# evaluate_dot(static_model, sick_noun_dataset_test)

# loss_fn = torch.nn.KLDivLoss()
loss_fn = torch.nn.MSELoss()
opt_flex_verbs = torch.optim.Adam(embedder_model_flex_verbs.parameters())), lr=0.0075)
opt_flex_nouns = torch.optim.Adam(embedder_model_flex_nouns.parameters(), lr=0.0075)
opt_flex_all = torch.optim.Adam(embedder_model_flex_all.parameters(), lr=0.0075)
opt_fix_all = torch.optim.Adam(embedder_model_fix_all.parameters(), lr=0.0075)
opt_flex_verbs_dot = torch.optim.Adam(dot_embedder_model_flex_verbs.parameters(), lr=0.0075)
opt_flex_all_dot = torch.optim.Adam(dot_embedder_model_flex_all.parameters(), lr=0.0075)

epoch_losses, evals, tests = train_epochs(embedder_model_flex_verbs, sick_dataloader, sick_dataset_dev, sick_dataset_test, loss_fn, opt_fix_all, 'cpu', 5)

sick_dataloader = DataLoader(sick_dataset_train, shuffle=True, batch_size=1)
# epoch_loss1 = train_epoch(embedder_model_fix_all, sick_dataloader, loss_fn, opt_fix_all, 'cpu', 1)

# epoch_loss = train_epoch_dot(dot_embedder_model_flex_verbs, sick_dataloader, loss_fn, opt_flex_verbs, 'cpu', 1)

# epoch_losses, evals, tests = train_epochs(embedder_model_flex_verbs, sick_dataloader, sick_dataset_dev, sick_dataset_test, loss_fn, opt_fix_all, 'cpu', 5)

model_configs = [('fix_all', embedder_model_fix_all, opt_fix_all),
                 ('flex_nouns', embedder_model_flex_nouns, opt_flex_nouns),
                 ('flex_verbs', embedder_model_flex_verbs, opt_flex_verbs),
                 ('flex_all', embedder_model_flex_all, opt_flex_all)]
num_epochs = 1

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
        print(f"Eval: {eval},    Test: {test}   Model: {name}")
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

# my_preproc = Preprocessor(preproc_fn, space_fn, verb_dict_fn, verbs_fn)
# lower2upper = my_preproc.preproc['l2u']
# allnouns = set(lower2upper.keys())
# allverbs = set(my_preproc.preproc['verb']['i2v'])
# sick_preproc = SICKPreprocessor(task_fn, sick_data_fn, allnouns, allverbs)
# noun_matrix = sick_preproc.create_noun_matrix(space_fn, lower2upper)
# verb_subj_cube = sick_preproc.create_verb_cube('subj', my_preproc.preproc['verb']['v2i'], folder)
# verb_subj_cube = verb_subj_cube.reshape(verb_subj_cube.shape[0], 10000)
# verb_obj_cube = sick_preproc.create_verb_cube('obj', my_preproc.preproc['verb']['v2i'], folder)
# verb_obj_cube = verb_obj_cube.reshape(verb_obj_cube.shape[0], 10000)
