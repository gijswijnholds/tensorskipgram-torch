import os
from tqdm import tqdm
from tensorskipgram.data.util import load_obj_fn, dump_obj_fn


space_fn = '/import/gijs-shared/gijs/spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt'
verb_dict_fn = '/import/gijs-shared/gijs/verb_data/verb_counts_all_corpus_verbs_dict.p'
verbs_fn = '/import/gijs-shared/gijs/verb_data/sick_verbs_full.txt'
preproc_fn = '/import/gijs-shared/gijs/verb_data/preproc_sick_verbcounts.p'

def load_nouns(space_fn):
    with open(space_fn, 'r') as file:
        nouns = [ln.split()[0] for ln in file.readlines()]
    return nouns


def load_verbs(verbs_fn):
    with open(verbs_fn, 'r') as file:
        verbs = [ln.strip() for ln in file.readlines()]
    return verbs


def load_verb_counts(verb_dict_fn, verbs, nouns):
    print("Opening verb counts...")
    verb_dict = load_obj_fn(verb_dict_fn)
    print("Filtering verb counts...")
    verb_dict_out = {v: {(s, o): verb_dict[v][(s, o)]
                         for (s, o) in verb_dict[v]
                         if s in nouns and o in nouns}
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


class Preprocessor(object):
    def __init__(self, preproc_fn, space_fn, verb_dict_fn, verbs_fn):
        self.preproc_fn = preproc_fn
        self.space_fn = space_fn
        self.verb_dict_fn = verb_dict_fn
        self.verbs_fn = verbs_fn
        if os.path.exists(preproc_fn):
            print("Loading preprocessing data...")
            self.preproc = load_obj_fn(preproc_fn)
        else:
            print("Preprocessing has not been done, please run setup.")

    def setup(self):
        nouns = load_nouns(self.space_fn)
        check_nouns = set(nouns + [n.lower() for n in nouns])
        i2v = sorted(load_verbs(verbs_fn))
        v2i = {v: i for i, v in enumerate(i2v)}
        v2c = load_verb_counts(self.verb_dict_fn, i2v, check_nouns)
        verb_preproc = {'i2v': i2v, 'v2i': v2i, 'v2c': v2c}
        subj_i2w, subj_w2i, subj_i2c, subj_i2ns = get_argument_preproc(v2c, 0)
        obj_i2w, obj_w2i, obj_i2c, obj_i2ns = get_argument_preproc(v2c, 1)
        subj_preproc = {'i2w': subj_i2w, 'w2i': subj_w2i, 'i2c': subj_i2c, 'i2ns': subj_i2ns}
        obj_preproc = {'i2w': obj_i2w, 'w2i': obj_w2i, 'i2c': obj_i2c, 'i2ns': obj_i2ns}
        preproc = {'verb': verb_preproc, 'subj': subj_preproc, 'obj': obj_preproc}
        self.preproc = preproc
        dump_obj_fn(preproc, self.preproc_fn)
        return self.preproc
# nouns = load_nouns(space_fn)
# lower_nouns = [n.lower() for n in nouns]
# check_nouns = set(nouns + lower_nouns)
# verbs = load_verbs(verbs_fn)
# verb_counts_new = load_verb_counts(verb_dict_fn, verbs, set(nouns))
my_preproc = Preprocessor(preproc_fn, space_fn, verb_dict_fn, verbs_fn)