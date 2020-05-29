import os
from tensorskipgram.data.preprocessing import Preprocessor
from tensorskipgram.data.training_data_creator import DataCreator
from tensorskipgram.data.dataset import MatrixSkipgramDataset
from torch.utils.data import Dataset, DataLoader

folder = '/import/gijs-shared/gijs'
data_folder = os.path.join(folder, 'verb_data')
space_fn = os.path.join(folder, 'spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt')
verb_dict_fn = os.path.join(folder, 'verb_data/verb_counts_all_corpus_verbs_dict.p')
verbs_fn = os.path.join(folder, 'verb_data/sick_verbs_full.txt')
preproc_fn = os.path.join(folder, 'verb_data/preproc_sick_verbcounts.p')


# my_preproc = Preprocessor(preproc_fn, space_fn, verb_dict_fn, verbs_fn)
# my_data_creator = DataCreator(my_preproc, data_folder)

subj_data_fn = os.path.join(data_folder, 'subj_train_data.p')
obj_data_fn = os.path.join(data_folder, 'obj_train_data.p')
# subj_dataset = MatrixSkipgramDataset(subj_data_fn, arg='subject', negk=5)
# obj_dataset = MatrixSkipgramDataset(obj_data_fn, arg='object', negk=5)
