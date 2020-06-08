import os
from tensorskipgram.data.preprocessing import Preprocessor
from tensorskipgram.evaluation.data import SICKPreprocessor, SICKDataset
from tensorskipgram.evaluation.model import SentenceEmbedder

task_fn = '/homes/gjw30/ExpCode/compdisteval/experiment_data/SICK/SICK.txt'
folder = '/import/gijs-shared/gijs'
data_folder = os.path.join(folder, 'verb_data')
space_fn = os.path.join(folder, 'spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt')
verb_dict_fn = os.path.join(folder, 'verb_data/verb_counts_all_corpus_verbs_dict.p')
verbs_fn = os.path.join(folder, 'verb_data/sick_verbs_full.txt')
preproc_fn = os.path.join(folder, 'verb_data/preproc_sick_verbcounts.p')
data_fn = os.path.join(folder, 'verb_data/sick_dataset.p')

my_preproc = Preprocessor(preproc_fn, space_fn, verb_dict_fn, verbs_fn)
lower2upper = my_preproc.preproc['l2u']
allnouns = set(lower2upper.keys())
allverbs = set(my_preproc.preproc['verb']['i2v'])


sick_preproc = SICKPreprocessor(task_fn, allnouns, allverbs)
sick_dataset = SICKDataset(data_fn=data_fn)
noun_matrix = sick_preproc.create_noun_matrix(space_fn, lower2upper)
verb_subj_cube = sick_preproc.create_verb_cube('subj', my_preproc.preproc['verb']['v2i'], folder)
verb_subj_cube = verb_subj_cube.reshape(verb_subj_cube.shape[0], 10000)
verb_obj_cube = sick_preproc.create_verb_cube('obj', my_preproc.preproc['verb']['v2i'], folder)
verb_obj_cube = verb_obj_cube.reshape(verb_obj_cube.shape[0], 10000)

embedder_model = SentenceEmbedder(noun_matrix, verb_subj_cube, verb_obj_cube)
