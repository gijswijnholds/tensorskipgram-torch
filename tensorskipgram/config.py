import os

base_folder = '/import/gijs-shared/gijs'
ukwackypedia_split_folder = os.path.join(base_folder,
                                         'corpora/ukwackypedia_split')
svo_triples_fn = os.path.join(base_folder,
                              'verb_data/verb_counts_all_corpus_verbs_dict.p')
noun_space_fn = os.path.join(base_folder,
                             'spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt')
verblist_fn = os.path.join(base_folder, 'verb_data/all_1160_verbs.txt')
# verblist_with_gaps_fn = os.path.join(base_folder, 'verb_data/all_1180_verbs.txt')
verblist_sick_fn = os.path.join(base_folder, 'verb_data/sick_verbs_full.txt')
verblist_paragaps_fn = os.path.join(base_folder, 'verb_data/allgappingverbs.txt')

subj_data_fn = os.path.join(base_folder, 'verb_data/subj_train_data_1160.p')
obj_data_fn = os.path.join(base_folder, 'verb_data/obj_train_data_1160.p')

subj_data_fn2 = os.path.join(base_folder, 'skipprob_data/training_data_combined_subject/train_data_proper_asym_ns=5.npy')
sick_subj_data_fn = os.path.join(base_folder, 'skipprob_data/training_data_sick_subject/train_data_proper_asym_ns=5.npy')
sick_obj_data_fn = os.path.join(base_folder, 'skipprob_data/training_data_sick_object/train_data_proper_asym_ns=5.npy')

preproc_fn = os.path.join(base_folder, 'verb_data/preprocessor_1160.p')

model_path_subj = os.path.join(base_folder, 'verb_data/matrixskipgram_subj')
model_path_obj = os.path.join(base_folder, 'verb_data/matrixskipgram_obj')


# model_path_subj_conc = os.path.join(base_folder, 'verb_data/matrixskipgram_subj_bs=11_lr=0.001_epoch1.p')
# model_path_obj_conc = os.path.join(base_folder, 'verb_data/matrixskipgram_obj_bs=11_lr=0.001_epoch1.p')


model_path_subj_conc = os.path.join(base_folder, 'spaces/conll_spaces/matrices_1160_arg_subj_context_obj.txt')
model_path_obj_conc = os.path.join(base_folder, 'spaces/conll_spaces/matrices_1160_arg_obj_context_subj.txt')


exp_base_folder = '/homes/gjw30/ExpCode/compdisteval/experiment_data/'
ml2008_path = os.path.join(exp_base_folder, 'ML2008/ML2008.txt')
ml2010_path = os.path.join(exp_base_folder, 'ML2010/ML2010.txt')
gs2011_path = os.path.join(exp_base_folder, 'GS2011/GS2011data.txt')
ks2013_path = os.path.join(exp_base_folder, 'KS2013/KS2013-CoNLL.txt')
ks2014_path = os.path.join(exp_base_folder, 'KS2014/KS2014.txt')
elldis_path = os.path.join(exp_base_folder, 'WS2018/ELLDIS_CORRECTED.txt')
ellsim_path = os.path.join(exp_base_folder, 'WS2018/ELLSIM_CORRECTED.txt')

menverb_path = os.path.join(exp_base_folder, 'MEN/MEN_dataset_lemma_form_full')
simlex_path = os.path.join(exp_base_folder, 'SimLex-999/SimLex-999.txt')
verbsim_path = os.path.join(exp_base_folder, 'VerbSim/200601-GWC-130verbpairs.txt')
simverbdev_path = os.path.join(exp_base_folder, 'SIMVERB3500/SimVerb-500-dev.txt')
simverbtest_path = os.path.join(exp_base_folder, 'SIMVERB3500/SimVerb-3000-test.txt')
relpron_path = os.path.join(exp_base_folder, 'RELPRON/relpron.test')
paragaps_path = os.path.join(exp_base_folder, 'PARGAP/pargaps_2020.txt')
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
# spaceInFN = '/import/gijs-shared/gijs/spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt'

verblist_with_gaps_fn = os.path.join(base_folder, 'verb_data/all_1180_verbs.txt')
preproc_gaps_fn = os.path.join(base_folder, 'verb_data/preprocessor_1180.p')
subj_data_gaps_fn = os.path.join(base_folder, 'verb_data/subj_train_data_1180.p')
obj_data_gaps_fn = os.path.join(base_folder, 'verb_data/obj_train_data_1180.p')
model_path_subj_gaps = os.path.join(base_folder, 'verb_data/matrixskipgram_pargaps_subj')
model_path_obj_gaps = os.path.join(base_folder, 'verb_data/matrixskipgram_pargaps_obj')
model_out_path_subj_gaps = os.path.join(base_folder, 'verb_data/matrixskipgram_pargaps_arg_subj_context_obj.txt')
model_out_path_obj_gaps = os.path.join(base_folder, 'verb_data/matrixskipgram_pargaps_arg_obj_context_subj.txt')

model_out_path_subj_gaps2 = os.path.join(base_folder, 'verb_data/matrixskipgram_pargaps_arg_subj_context_obj_bs=110.txt')
model_out_path_obj_gaps2 = os.path.join(base_folder, 'verb_data/matrixskipgram_pargaps_arg_obj_context_subj_bs=110.txt')

model_out_path_subj_gapss = [os.path.join(base_folder, f'verb_data/matrixskipgram_pargaps_arg_subj_context_obj_bs=110_epoch={e}.txt')
                             for e in [1,2,3,4,5]]
model_out_path_obj_gapss = [os.path.join(base_folder, f'verb_data/matrixskipgram_pargaps_arg_obj_context_subj_bs=110_epoch={e}.txt')
                            for e in [1,2,3,4,5]]

relational_mats_out_fn = os.path.join(base_folder, 'verb_data/relational_tensors_1180_verbs.txt')
kronecker_mats_out_fn = os.path.join(base_folder, 'verb_data/kronecker_tensors_1180_verbs.txt')
bert_mats_out_fn = os.path.join(base_folder, 'verb_data/relational_bert_tensors_1180_verbs.txt')
bert_in_context_mats_out_fn = os.path.join(base_folder, 'verb_data/relational_context_bert_tensors_1180_verbs.txt')

bert_space_fn = os.path.join(base_folder, 'spaces/bert_vectors/bert_768_nouns.txt')
kronecker_bert_mats_out_fn = os.path.join(base_folder, 'spaces/bert_vectors/kronecker_bert_tensors_1180_verbs.txt')
