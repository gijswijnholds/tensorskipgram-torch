import os
from tensorskipgram.data.training_data_creator import Preprocessor, DataCreator
from tensorskipgram.data.dataset import MatrixSkipgramDataset
from torch.utils.data import Dataset, DataLoader

folder = '/import/gijs-shared/gijs'
data_folder = os.path.join(folder, 'verb_data')
space_fn = os.path.join(folder, 'spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt')
verb_dict_fn = os.path.join(folder, 'verb_data/verb_counts_all_corpus_verbs_dict.p')
verbs_fn = os.path.join(folder, 'verb_data/sick_verbs_full.txt')
preproc_fn = os.path.join(folder, 'verb_data/preproc_sick_verbcounts.p')

# subj_dataset = MatrixSkipgramDataset(subj_data_fn, arg='subject', negk=5)
# obj_dataset = MatrixSkipgramDataset(obj_data_fn, arg='object', negk=5)
