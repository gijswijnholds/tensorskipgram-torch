from tensorskipgram.evaluation.data \
    import SICKDataset, SICKDatasetNouns, SICKPreprocessor
from tensorskipgram.evaluation.paths \
    import data_fn, data_fn_e, data_fn_nouns, data_fn_e_nouns, model_data_fn,
from tensorskipgram.evaluation.paths \
    import model_rel_data_fn, model_kron_data_fn
from tensorskipgram.evaluation.paths \
    import preproc_fn, task_fn, space_fn, sick_data_fn, verb_dict_fn, verbs_fn, folder
from tensorskipgram.data.util import load_obj_fn
from tensorskipgram.data.preprocessing import Preprocessor


sick_dataset_train = SICKDataset(data_fn=data_fn, setting='train')
sick_dataset_train_filter = SICKDataset(data_fn=data_fn, setting='train', filter=True)
sick_dataset_dev = SICKDataset(data_fn=data_fn, setting='dev')
sick_dataset_test = SICKDataset(data_fn=data_fn, setting='test')

sick_dataset_e_train = SICKDataset(data_fn=data_fn_e, setting='train', data_sort='entailment')
sick_dataset_e_dev = SICKDataset(data_fn=data_fn_e, setting='dev', data_sort='entailment')
sick_dataset_e_test = SICKDataset(data_fn=data_fn_e, setting='test', data_sort='entailment')

sick_noun_dataset_train = SICKDatasetNouns(data_fn=data_fn_nouns, setting='train')
sick_noun_dataset_dev = SICKDatasetNouns(data_fn=data_fn_nouns, setting='dev')
sick_noun_dataset_test = SICKDatasetNouns(data_fn=data_fn_nouns, setting='test')

sick_noun_dataset_e_train = SICKDatasetNouns(data_fn=data_fn_e_nouns, setting='train', data_sort='entailment')
sick_noun_dataset_e_dev = SICKDatasetNouns(data_fn=data_fn_e_nouns, setting='dev', data_sort='entailment')
sick_noun_dataset_e_test = SICKDatasetNouns(data_fn=data_fn_e_nouns, setting='test', data_sort='entailment')

def reload_representations():
    vec_mats = load_obj_fn(model_data_fn)
    noun_matrix = vec_mats['noun_matrix']
    verb_subj_cube = vec_mats['verb_subj_cube']
    verb_obj_cube = vec_mats['verb_obj_cube']
    return noun_matrix, verb_subj_cube, verb_obj_cube

noun_matrix, verb_subj_cube, verb_obj_cube = reload_representations()


def recreate_representations():
    my_preproc = Preprocessor(preproc_fn, space_fn, verb_dict_fn, verbs_fn)
    lower2upper = my_preproc.preproc['l2u']
    allnouns = set(lower2upper.keys())
    allverbs = set(my_preproc.preproc['verb']['i2v'])
    sick_preproc = SICKPreprocessor(task_fn, sick_data_fn, allnouns, allverbs)
    noun_matrix = sick_preproc.create_noun_matrix(space_fn, lower2upper)
    verb_subj_cube = sick_preproc.create_verb_cube('subj', my_preproc.preproc['verb']['v2i'], folder, setting='skipgram')
    verb_subj_cube = verb_subj_cube.reshape(verb_subj_cube.shape[0], 10000)
    verb_obj_cube = sick_preproc.create_verb_cube('obj', my_preproc.preproc['verb']['v2i'], folder, setting='skipgram')
    verb_obj_cube = verb_obj_cube.reshape(verb_obj_cube.shape[0], 10000)
    return noun_matrix, verb_subj_cube, verb_obj_cube


def recreate_representations_relational():
    my_preproc = Preprocessor(preproc_fn, space_fn, verb_dict_fn, verbs_fn)
    lower2upper = my_preproc.preproc['l2u']
    allnouns = set(lower2upper.keys())
    allverbs = set(my_preproc.preproc['verb']['i2v'])
    sick_preproc = SICKPreprocessor(task_fn, sick_data_fn, allnouns, allverbs)
    noun_matrix = sick_preproc.create_noun_matrix(space_fn, lower2upper)
    verb_subj_cube = \
        sick_preproc.create_verb_cube('subj',
                                      my_preproc.preproc['verb']['v2i'],
                                      folder, setting='relational',
                                      verb_counts=my_preproc.preproc['verb']['v2c'],
                                      lower2upper=lower2upper)
    verb_subj_cube = verb_subj_cube.reshape(verb_subj_cube.shape[0], 10000)
    verb_obj_cube = \
        sick_preproc.create_verb_cube('obj',
                                      my_preproc.preproc['verb']['v2i'],
                                      folder, setting='relational',
                                      verb_counts=my_preproc.preproc['verb']['v2c'],
                                      lower2upper=lower2upper)
    verb_obj_cube = verb_obj_cube.reshape(verb_subj_cube.shape[0], 10000)
    return noun_matrix, verb_subj_cube, verb_obj_cube
    # something something

def reload_representations_relational():
    vec_mats = load_obj_fn(model_rel_data_fn)
    noun_matrix = vec_mats['noun_matrix']
    verb_subj_cube = vec_mats['verb_subj_cube']
    verb_obj_cube = vec_mats['verb_obj_cube']
    return noun_matrix, verb_subj_cube, verb_obj_cube

def save_representations(noun_matrix, verb_subj_cube, verb_obj_cube, dump_fn):
    vec_mats = {'noun_matrix': noun_matrix, 'verb_subj_cube': verb_subj_cube,
                'verb_obj_cube': verb_obj_cube}
    dump_obj_fn(vec_mats, dump_fn)


def reload_old_skipgram_matrices():
    # skipprob_sick_joint_100_proper_matrixsubject_subject_ns=10_lr=0.05_bs=11_epoch=1_asym=True_v=20.shelve
