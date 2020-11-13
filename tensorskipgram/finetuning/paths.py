import os

task_fn = '/homes/gjw30/ExpCode/compdisteval/experiment_data/SICK/SICK.txt'
folder = '/import/gijs-shared/gijs'
data_folder = os.path.join(folder, 'verb_data')
space_fn = os.path.join(folder, 'spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt')
verb_dict_fn = os.path.join(folder, 'verb_data/verb_counts_all_corpus_verbs_dict.p')
verbs_fn = os.path.join(folder, 'verb_data/sick_verbs_full.txt')
preproc_fn = os.path.join(folder, 'verb_data/preproc_sick_verbcounts.p')
data_fn = os.path.join(folder, 'verb_data/sick_dataset.p')
data_fn_e = os.path.join(folder, 'verb_data/sick_dataset_e.p')
data_fn_nouns = os.path.join(folder, 'verb_data/sick_dataset_nouns.p')
data_fn_e_nouns = os.path.join(folder, 'verb_data/sick_dataset_e_nouns.p')
sick_data_fn = os.path.join(folder, 'sick_data.p')
model_data_fn = os.path.join(folder, 'verb_data/sick_model_data.p')
model_rel_data_fn = os.path.join(folder, 'verb_data/sick_model_rel_data.p')
model_kron_data_fn = os.path.join(folder, 'verb_data/sick_model_kron_data.p')
