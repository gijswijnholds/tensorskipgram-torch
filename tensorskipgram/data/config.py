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
