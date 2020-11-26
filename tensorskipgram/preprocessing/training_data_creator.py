import os
import numpy as np
from tqdm import tqdm
from tensorskipgram.preprocessing.util import load_obj_fn, dump_obj_fn, stopwords
from typing import List, Set, Dict


def load_nouns(space_fn: str) -> List[str]:
    with open(space_fn, 'r') as file:
        nouns = [ln.split()[0] for ln in file.readlines()]
    return nouns


def load_verbs(verbs_fn: str) -> List[str]:
    with open(verbs_fn, 'r') as file:
        verbs = [ln.strip() for ln in file.readlines()]
    return verbs


def load_verb_counts(verb_dict_fn: str, verbs: List[str], nouns: Set[str],
                     stopwords: Set[str]):
    print("Opening verb counts...")
    verb_dict = load_obj_fn(verb_dict_fn)
    print("Filtering verb counts...")
    verb_dict_out = {v: {(s, o): verb_dict[v][(s, o)]
                         for (s, o) in verb_dict[v]
                         if s in nouns and o in nouns
                         and s not in stopwords and o not in stopwords}
                     for v in tqdm(verbs) if v in verb_dict}
    return verb_dict_out


def get_argument_preproc(verb_counts, i):
    print("Getting argument preproc...")
    argFreqs = [(args[i], verb_counts[v][args]) for v in tqdm(verb_counts)
                for args in verb_counts[v]]
    arg2c = {}
    for (s, f) in argFreqs:
        if s in arg2c:
            arg2c[s] += f
        else:
            arg2c[s] = f
    arg_i2w = sorted(arg2c.keys())
    arg_w2i = {w: i for i, w in enumerate(arg_i2w)}
    arg_i2c = [arg2c[w] for w in arg_i2w]
    arg_nsSum = float(sum([c**0.75 for c in arg_i2c]))
    arg_i2ns = [(c**0.75) / arg_nsSum for c in arg_i2c]
    return arg_i2w, arg_w2i, arg_i2c, arg_i2ns


def create_lower_to_upper(nouns: Set[str]) -> Dict[str, str]:
    noun_dict = {n: n for n in nouns}
    noun_dict_lower = {n.lower(): n for n in nouns}
    noun_dict_lower.update(noun_dict)
    return noun_dict_lower

# preprocessor = Preprocessor(preproc_fn, noun_space_fn, svo_triples_fn, verblist_fn)

class Preprocessor(object):
    def __init__(self, preproc_fn: str, space_fn: str, verb_dict_fn: str, verbs_fn: str):
        self.preproc_fn = preproc_fn
        self.space_fn = space_fn
        self.verb_dict_fn = verb_dict_fn
        self.verbs_fn = verbs_fn
        if os.path.exists(preproc_fn):
            print("Loading preprocessing data...")
            self.preproc = load_obj_fn(preproc_fn)
        else:
            print("Preprocessing has not been done, will run setup to create" +
                  "and save a preprocessor.")
            self.setup()

    def setup(self):
        print("Loading nouns...")
        nouns = load_nouns(self.space_fn)
        print("Filtering nouns...")
        check_nouns = set(nouns + [n.lower() for n in nouns])
        lower_to_upper = create_lower_to_upper(nouns)
        print("Loading and sorting verbs...")
        i2v = sorted(list(set(load_verbs(self.verbs_fn))))
        v2i = {v: i for i, v in enumerate(i2v)}
        print("Loading verb counts...")
        v2c = load_verb_counts(self.verb_dict_fn, i2v, check_nouns, stopwords)
        verb_preproc = {'i2v': i2v, 'v2i': v2i, 'v2c': v2c}
        print("Creating argument preprocessors...")
        subj_i2w, subj_w2i, subj_i2c, subj_i2ns = get_argument_preproc(v2c, 0)
        obj_i2w, obj_w2i, obj_i2c, obj_i2ns = get_argument_preproc(v2c, 1)
        subj_preproc = {'i2w': subj_i2w, 'w2i': subj_w2i, 'i2c': subj_i2c, 'i2ns': subj_i2ns}
        obj_preproc = {'i2w': obj_i2w, 'w2i': obj_w2i, 'i2c': obj_i2c, 'i2ns': obj_i2ns}
        preproc = {'verb': verb_preproc, 'subj': subj_preproc, 'obj': obj_preproc, 'l2u': lower_to_upper}
        self.preproc = preproc
        print("Dumping preprocessor...")
        dump_obj_fn(preproc, self.preproc_fn)


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


def create_train_data(verbs, verb_counts, v2i, subj_w2i, obj_w2i, i2ns, arg, ns_k=5):
    """Create training data for the Verb-Object model, in which the object is
    fixed and negative sampling happens for subjects."""
    print("Generating data...")
    verb_arrays = [create_train_data_verb(verb, verb_counts[verb], v2i, subj_w2i,
                                          obj_w2i, i2ns, arg, ns_k=ns_k)
                   for verb in tqdm(verb_counts)]
    print("Concatenating arrays...")
    single_verb_arrays = np.concatenate(verb_arrays)
    print("Shuffling batches...")
    np.random.shuffle(single_verb_arrays)
    print("Concatenating batches...")
    final_verb_arrays = np.concatenate(single_verb_arrays)
    print("Done creating training data!...")
    return final_verb_arrays


class DataCreator(object):
    def __init__(self, preproc: Preprocessor, subj_data_fn: str, obj_data_fn: str):
        if preproc.preproc is None:
            print("Preprocessing has not been done, please run preprocessor setup.")
        self.preproc = preproc
        self.subj_data_fn = subj_data_fn
        self.obj_data_fn = obj_data_fn

    def setup(self, neg_samples: int = 5) -> None:
        verb_preproc = self.preproc.preproc['verb']
        verbs, v2i, v2c = verb_preproc['i2v'], verb_preproc['v2i'], verb_preproc['v2c']
        subj_preproc = self.preproc.preproc['subj']
        subj_w2i, subj_i2ns = subj_preproc['w2i'], subj_preproc['i2ns']
        obj_preproc = self.preproc.preproc['obj']
        obj_w2i, obj_i2ns = obj_preproc['w2i'], obj_preproc['i2ns']
        subj_train_data = create_train_data(verbs, v2c, v2i, subj_w2i, obj_w2i,
                                            subj_i2ns, 'subj', ns_k=neg_samples)
        obj_train_data = create_train_data(verbs, v2c, v2i, subj_w2i, obj_w2i,
                                           obj_i2ns, 'obj', ns_k=neg_samples)
        print("Dumping subj_neg data...")
        dump_obj_fn(subj_train_data, self.subj_data_fn)
        print("Dumping obj_neg data...")
        dump_obj_fn(obj_train_data, self.obj_data_fn)
        print("All done!")
