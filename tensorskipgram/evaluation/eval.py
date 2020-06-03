"""Evaluation code."""
import spacy
from tqdm import tqdm


def load_spacy():
    return spacy.load('en_core_web_lg', disable=["ner", "textcat",
                                                 "entity_ruler", "sentencizer",
                                                 "merge_entities"])


def parse_sents(sents, model):
    return [model(s) for s in tqdm(sents)]


def getVerbsAndArgs(parse):
    return [(t.text, [c for c in t.children if c.dep_ == 'nsubj'],
            [c for c in t.children if c.dep_ == 'dobj'])
            for t in parse if t.pos_ == 'VERB']


def s2aFilter(t):
    s, vargs = t
    return s, [v for v in vargs if not (v[1] == [] and v[2] == [])]


class SICK(object):
    def __init__(self):
        self.fn = '/homes/gjw30/ExpCode/compdisteval/experiment_data/SICK/SICK.txt'
        self.data = self.load_data()
        self.sentences = self.split_data()
        self.parse_data = self.parse_data()
        # self.index_data = self.index_data()

    def load_data(self):
        with open(self.fn, 'r') as in_file:
            lines = [ln.strip().split('\t') for ln in in_file.readlines()][1:]
        sentence_data = [tuple(ln[1:5]) for ln in lines]
        sentence_data = [(s1, s2, el, float(rl))
                         for (s1, s2, el, rl) in sentence_data]
        return sentence_data

    def split_data(self):
        sentences = []
        for (s1, s2, _, _) in self.data:
            sentences.append(s1)
            sentences.append(s2)
        return list(set(sentences))

    def parse_data(self):
        print("Loading Spacy model...")
        nlp = load_spacy()
        print("Parsing sentences...")
        parses = parse_sents(self.sentences, nlp)
        s2a = {s.text: getVerbsAndArgs(s) for s in parses}
        return corr_s2a

    def index_data(self, word2index, verb2index):
        pass
