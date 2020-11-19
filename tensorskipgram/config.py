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

# 
# model_path_subj_conc = os.path.join(base_folder, 'spaces/conll_spaces/matrices_1160_arg_subj_context_obj.txt')
# model_path_obj_conc = os.path.join(base_folder, 'spaces/conll_spaces/matrices_1160_arg_obj_context_subj.txt')


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
