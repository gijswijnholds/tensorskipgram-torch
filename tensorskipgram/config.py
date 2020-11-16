import os

base_folder = '/import/gijs-shared/gijs'
ukwackypedia_split_folder = os.path.join(base_folder,
                                         'corpora/ukwackypedia_split')
svo_triples_fn = os.path.join(base_folder,
                              'verb_data/verb_counts_all_corpus_verbs_dict.p')
noun_space_fn = os.path.join(base_folder,
                             'spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt')
verblist_fn = os.path.join(base_folder, 'verb_data/all_1160_verbs.txt')
verblist_sick_fn = os.path.join(base_folder, 'verb_data/sick_verbs_full.txt')

subj_data_fn = os.path.join(base_folder, 'verb_data/subj_train_data.p')
obj_data_fn = os.path.join(base_folder, 'verb_data/obj_train_data.p')

sick_subj_data_fn = os.path.join(base_folder, 'skipprob_data/training_data_sick_subject/train_data_proper_asym_ns=5.npy')
sick_obj_data_fn = os.path.join(base_folder, 'skipprob_data/training_data_sick_object/train_data_proper_asym_ns=5.npy')

preproc_fn = os.path.join(base_folder, 'verb_data/preprocessor.p')

model_path_subj = os.path.join(base_folder, 'verb_data/matrixskipgram_subj')
model_path_obj = os.path.join(base_folder, 'verb_data/matrixskipgram_obj')


model_path_subj_conc = os.path.join(base_folder, 'verb_data/matrixskipgram_subj_bs=11_lr=0.001_epoch1.p')
model_path_obj_conc = os.path.join(base_folder, 'verb_data/matrixskipgram_obj_bs=11_lr=0.001_epoch1.p')
