"""The SICK dataset."""
import spacy
from tqdm import tqdm


def load_spacy():
    return spacy.load('en_core_web_lg', disable=["ner", "textcat",
                                                 "entity_ruler", "sentencizer",
                                                 "merge_entities"])


def parse_sents(sents, model):
    return [model(s) for s in tqdm(sents)]


def get_verbs_args(parse):
    return [(t, [c for c in t.children if c.dep_ == 'nsubj'],
             [c for c in t.children if c.dep_ == 'dobj'])
            for t in parse if t.pos_ == 'VERB']


def verb_filter(vargs):
    return [v for v in vargs if not (v[1] == [] and v[2] == [])]


def get_subjs_objs_list(vargs):
    return [s for (v, ss, os) in vargs for s in ss + os]


def get_verb_list(vargs):
    return [v for (v, ss, os) in vargs]


def get_verb_args_list(vargs):
    return [s for (v, ss, os) in vargs for s in [v] + ss + os]


def get_split_parse(parse):
    vargs = verb_filter(get_verbs_args(parse))
    varg_list = get_verb_args_list(vargs)
    sent_split = [t.text for t in parse if t not in varg_list]
    vargs = [(v.text, [s.text for s in ss], [o.text for o in os])
             for (v, ss, os) in vargs]
    return sent_split, vargs


class SICK(object):
    def __init__(self, fn):
        self.fn = fn
        self.data = self.load_data()
        self.sentences = self.split_data()
        self.parse_data = self.parse_data()
        self.noun_vocab, self.verb_vocab = self.create_vocabs()

    def load_data(self):
        with open(self.fn, 'r') as in_file:
            lines = [ln.strip().split('\t') for ln in in_file.readlines()][1:]
        sentence_data = [tuple(ln[1:5]) for ln in lines]
        sentence_data = [(s1, s2, el, float(rl))
                         for (s1, s2, el, rl) in sentence_data]
        return sentence_data

    def split_data(self):
        return list(set([s for (s1, s2, _, _) in self.data for s in (s1, s2)]))

    def parse_data(self):
        print("Loading Spacy model...")
        nlp = load_spacy()
        print("Parsing sentences...")
        parses = parse_sents(self.sentences, nlp)
        s2a = {s.text: get_split_parse(s) for s in parses}
        return s2a

    def create_vocabs(self):
        vocab_noun = set([w for (split, vargs) in self.parse_data.values()
                          for w in split + get_subjs_objs_list(vargs)])
        vocab_verbs = set([w for (split, vargs) in self.parse_data.values()
                           for w in get_verb_list(vargs)])
        return vocab_noun, vocab_verbs
