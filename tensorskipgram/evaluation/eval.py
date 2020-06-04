"""Evaluation code."""
import spacy
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer


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


def get_verb_args_list(vargs):
    varg_list = []
    for (v, ss, os) in vargs:
        varg_list.append(v)
        varg_list += ss
        varg_list += os
    return varg_list


def get_split_parse(parse):
    vargs = verb_filter(get_verbs_args(parse))
    varg_list = get_verb_args_list(vargs)
    sent_split = [t.text for t in parse if t not in varg_list]
    vargs = [(v.text, [s.text for s in ss], [o.text for o in os])
             for (v, ss, os) in vargs]
    return sent_split, vargs


class SICK(object):
    def __init__(self, verb2index, lower2upper):
        self.fn = '/homes/gjw30/ExpCode/compdisteval/experiment_data/SICK/SICK.txt'
        self.data = self.load_data()
        self.sentences = self.split_data()
        self.parse_data = self.parse_data()
        self.lemmatiser = WordNetLemmatizer()
        self.word2index = word2index
        self.verb2index = verb2index
        # self.index_data = self.index_data()

    def setup(self, word2index, verb2index):
        # add stuff
        pass

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
        s2a = {s.text: get_split_parse(s) for s in parses}
        return s2a

    def lemmatise_noun(self, noun):
        if noun in self.word2index:
            return noun
        elif noun.lower() in self.word2index:
        return pass
        [n for n in sick_vocab_noun if n.lower() not in  my_preproc.preproc['l2u'] and wnl.lemmatize(n.lower(),pos='v') not in my_preproc.preproc['l2u'] and wnl.lemmatize(n.lower(),pos='n') not in my_preproc.preproc['l2u']]

        [' ', "'s", ',', '-', '.', '/', 'ATVs', 'Seadoo', 'amazedly', 'amusedly', 'backbends', 'bellbottoms', 'bmxs', 'corndogs', 'daschunds', 'gloved', 'graphitized', 'midspeech', 'mittened', 'motionlessly', 'n', "n't", 'riskily', 'rollerblader', 't', 'uninterestedly', 'unprotective', 'unwarily']
    def index_noun(self, noun):
        """Convert a noun to a token index (0 for oov words)."""
        corr_noun = lemmatise_noun(noun)
        if corr_noun in self.word2index:
            return self.word2index[corr_noun]
        else:
            return 0

    def augment_word2index(self, word2index):
        w2i = {w: word2index[w+1] for w in word2index}
        assert '<UNK>' not in w2i
        w2i['<UNK>'] = 0
        for n in ['/', 'uninterestedly', 'bmxs', 'unwarily', 'rollerblader',
                  'midspeech', ',', 'ATVs', 'bellbottoms', 'Seadoo', 'riskily',
                  'amazedly', 'corndogs', 'gloved', 'graphitized', 'backbends',
                  '-', 'motionlessly', 'n', 'mittened', "n't", '.',
                  'daschunds', 'unprotective', "'s", ' ', 'amusedly', 't']:
            w2i[n] = 0
        return w2i

    def index_verb(self):
        pass

    def index_data(self, word2index, verb2index):
        def varg_idx(vargs):
            return [(verb2index[v], [word2index[s] for s in ss],
                    [word2index[o] for o in os]) for (v, ss, os) in vargs]
        return {s: ([word2index[t] for t in split], varg_idx(vargs))
                for s, (split, vargs) in tqdm(self.parse_data.items())}


my_preproc = Preprocessor(preproc_fn, space_fn, verb_dict_fn, verbs_fn)
verb_v2i = my_preproc.preproc['verb']['v2i']
lower2upper = my_preproc.preproc['l2u']
word2index = {w: i for i, w in enumerate(lower2upper.values())}

sick = SICK(verb_v2i, lower2upper)
allStems = [wnl.lemmatize(v, pos='v') for v in allVerbs]
