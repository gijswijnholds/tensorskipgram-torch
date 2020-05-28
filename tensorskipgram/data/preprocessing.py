from tensorskipgram.data.util import load_obj_fn

space_fn = '/import/gijs-shared/gijs/spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt'
verb_counts_fn = '/import/gijs-shared/gijs/verb_data/verb_counts_all_corpus_verbs_dict.p'
verbs_fn = '/import/gijs-shared/gijs/verb_data/sick_verbs_full.txt'


def load_nouns(space_fn):
    with open(space_fn, 'r') as file:
        nouns = [ln.split()[0] for ln in file.readlines()]
    return nouns


def load_verbs(verbs_fn):
    with open(verbs_fn, 'r') as file:
        verbs = [ln.strip() for ln in file.readlines()]
    return


def load_verb_counts(verb_dict_fn, verbs, nouns):
    verb_dict = load_obj_fn(verb_dict_fn)
    verb_dict_out = {v: {(s, o): verb_dict[v][(s, o)]
                         for (s, o) in verb_dict[v]
                         if s in nouns and o in nouns}
                     for v in verbs}
    return verb_dict_out


nouns = load_nouns(space_fn)
verbs = load_verbs(verbs_fn)
verb_counts = load_verb_counts(verb_counts_fn, verbs, set(nouns))
