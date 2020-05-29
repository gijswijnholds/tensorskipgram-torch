import os
import numpy as np
from tqdm import tqdm
from tensorskipgram.data.preprocessing import Preprocessor
from tensorskipgram.data.util import dump_obj_fn


def create_train_data_verb(verb, counts, v2i, subj_w2i, obj_w2i, i2ns, arg, ns_k=5):
    vi = v2i[verb]
    v_counts = list(counts.items())

    assert arg in ['subj', 'obj']   # which one is negatively sampled?
    if arg == 'subj':
        neg_samples = iter(np.random.choice(len(subj_w2i),
                           ns_k*sum(counts.values()), p=i2ns))

        def subj_f(s): return [s]+[next(neg_samples) for _ in range(ns_k)]

        def obj_f(o): return (ns_k+1)*[o]
    elif arg == 'obj':
        neg_samples = iter(np.random.choice(len(obj_w2i),
                           ns_k*sum(counts.values()), p=i2ns))

        def subj_f(s): return (ns_k+1)*[s]

        def obj_f(o): return [o]+[next(neg_samples) for _ in range(ns_k)]
    svols = [np.array([np.array(subj_f(subj_w2i[s])),
                       np.array((ns_k+1)*[vi]),
                       np.array(obj_f(obj_w2i[o])),
                       np.array([1.] + ns_k*[0.])], dtype=object).T
             for (s, o), f in v_counts for i in range(f)]
    array = np.array(svols)
    return array


def create_train_data(verbs, verbCounts, v2i, subj_w2i, obj_w2i, i2ns, arg, ns_k=5):
    """Create training data for the Verb-Object model, in which the object is
    fixed and negative sampling happens for subjects."""
    print("Generating data...")
    verbArrays = [create_train_data_verb(verb, verbCounts[verb], v2i, subj_w2i,
                                         obj_w2i, i2ns, arg, ns_k=ns_k)
                  for verb in tqdm(verbCounts)]
    print("Concatenating arrays...")
    singleVerbArrays = np.concatenate(verbArrays)
    print("Shuffling batches...")
    np.random.shuffle(singleVerbArrays)
    print("Concatenating batches...")
    finalVerbArrays = np.concatenate(singleVerbArrays)
    print("Done creating training data!...")
    return finalVerbArrays


class DataCreator(object):
    def __init__(self, preproc: Preprocessor, data_folder: str):
        if preproc.preproc is None:
            print("Preprocessing has not been done, please run preprocessor setup.")
        self.preproc = preproc
        self.data_folder = data_folder

    def setup(self):
        verb_preproc = self.preproc.preproc['verb']
        verbs, v2i, v2c = verb_preproc['i2v'], verb_preproc['v2i'], verb_preproc['v2c']
        subj_preproc = self.preproc.preproc['subj']
        subj_w2i, subj_i2ns = subj_preproc['w2i'], subj_preproc['i2ns']
        obj_preproc = self.preproc.preproc['obj']
        obj_w2i, obj_i2ns = obj_preproc['w2i'], obj_preproc['i2ns']
        subj_train_data = create_train_data(verbs, v2c, v2i, subj_w2i, obj_w2i,
                                            subj_i2ns, 'subj', ns_k=5)
        obj_train_data = create_train_data(verbs, v2c, v2i, subj_w2i, obj_w2i,
                                           obj_i2ns, 'obj', ns_k=5)
        print("Dumping subj_neg data...")
        dump_obj_fn(subj_train_data,
                    os.path.join(self.data_folder, 'subj_train_data.p'))
        print("Dumping obj_neg data...")
        dump_obj_fn(obj_train_data,
                    os.path.join(self.data_folder, 'obj_train_data.p'))
        print("All done!")
