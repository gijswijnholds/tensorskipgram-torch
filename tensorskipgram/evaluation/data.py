"""Evaluation code; dataset."""
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from tensorskipgram.evaluation.sick import SICK
from nltk.stem import WordNetLemmatizer
from tensorskipgram.data.preprocessing import Preprocessor

UNK_TOKEN = 'UNK'


def create_indexer(vocab_mapper):
    """Create an indexer from a vocab2vocab mapper.

    Remove the unknown token as we do not want to incorporate it in our network.
    """
    idx_mapper = {w: i for i, w in enumerate(set(vocab_mapper.values()))}
    word2index = {w: idx_mapper[vocab_mapper[w]] for w in vocab_mapper}
    return word2index


class SICKPreprocessor(object):
    def __init__(self, fn, nouns, verbs):
        self.sick = SICK(fn)
        self.lemmatiser = WordNetLemmatizer()
        self.nouns = nouns
        self.verbs = verbs
        self.verb2verb, self.verb2index = self.create_verb_indexer()
        self.word2word, self.word2index = self.create_word_indexer()

    def create_verb_indexer(self):
        """Create an indexer for verbs.

        Create a mapping from the sick verb vocabulary to the vocabulary
        used for the verb matrices, and then create an indexer from this.
        """
        verbs = {v: self.lemmatiser.lemmatize(v, pos='v')
                 for v in self.sick.verb_vocab}
        lemma_corrector = {'strip': 'stripe', 'rid': 'ride', 'shin': 'shine',
                           'frolic': 'frolick', 'scar': 'scare',
                           'star': 'stare', 'wakeboarding': 'wakeboard',
                           'unstitching': 'unstitch', 'wad': 'wade',
                           'stag': 'stage', 'waterskiing': 'waterski'}
        verbs2 = {v: lemma_corrector[verbs[v]] if verbs[v] in lemma_corrector
                  else verbs[v] for v in verbs}
        verb_vocab_mapper = {v: verbs2[v] if verbs2[v] in self.verbs
                             else UNK_TOKEN for v in verbs2}
        verb2verb = {v: verb_vocab_mapper[v] for v in verb_vocab_mapper
                     if verb_vocab_mapper[v] != UNK_TOKEN}
        verb2index = create_indexer(verb2verb)
        return verb2verb, verb2index

    def create_word_indexer(self):
        words = {w: self.lemmatiser.lemmatize(w) if w not in self.nouns
                 else w for w in self.sick.noun_vocab}
        words2 = {w: self.lemmatiser.lemmatize(w.lower()) if words[w] not in self.nouns
                  else words[w] for w in words}
        words3 = {w: self.lemmatiser.lemmatize(w, pos='v') if words2[w] not in self.nouns
                  else words2[w] for w in words2}
        word_corrector = {'ATVs': 'ATV', 'Rollerbladers': 'rollerbladers',
                          'Seadoo': 'jetski', 'amazedly': 'amazed',
                          'amusedly': 'amused', 'backbends': 'backflip',
                          'bellbottoms': 'flares', 'bmxs': 'bmx',
                          'corndogs': 'corndawg', 'daschunds': 'dachshund',
                          'gloved': 'glove', 'graphitized': 'graphite',
                          'midspeech': 'speech', 'mittened': 'mitten',
                          'motionlessly': 'motionless', 'riskily': 'risky',
                          'rollerblader': 'rollerbladers',
                          'uninterestedly': 'uninterested',
                          'unprotective': 'unprotected', 'unwarily': 'unwary'}
        words4 = {w: word_corrector[w] if words3[w] in word_corrector
                  else words3[w] for w in words3}
        word_vocab_mapper = {w: words4[w] if words4[w] in self.nouns
                             else UNK_TOKEN for w in words4}
        word2word = {v: word_vocab_mapper[v] for v in word_vocab_mapper
                     if word_vocab_mapper[v] != UNK_TOKEN}
        word2index = create_indexer(word2word)
        return word2word, word2index

    def index_parse(self, split_parse):
        """Create an indexed version of a split parse for processing."""
        words, vargs = split_parse
        word_idxs = [self.word2index[w] for w in words if w in self.word2index]
        verb_idxs = []
        for (v, ss, objs) in vargs:
            if (v not in self.verb2index or
                any([s not in self.word2index for s in ss]) or
                any([o not in self.word2index for o in objs])):
                if v in self.word2index:
                    word_idxs.append(self.word2index[v])
                word_idxs += [self.word2index[s] for s in ss if s in self.word2index]
                word_idxs += [self.word2index[o] for o in objs if o in self.word2index]
            else:
                verb_idxs.append((self.verb2index[v],
                                  [self.word2index[s] for s in ss if s in self.word2index],
                                  [self.word2index[o] for o in objs if o in self.word2index]))
        return word_idxs, verb_idxs

    def create_noun_matrix(self, space_fn, lower2upper):
        with open(space_fn, 'r') as file:
            lines = [ln.strip().split() for ln in file.readlines()]
        print("Loading vectors...")
        space = {ln[0]: np.array([float(b) for b in ln[1:]])
                 for ln in tqdm(lines)}
        indices = sorted(list(set(self.word2index.values())))
        noun_matrix = np.zeros((len(indices), 100))
        print("Filling noun matrix...")
        for w in self.word2index:
            noun_matrix[self.word2index[w]] = space[lower2upper[self.word2word[w]]]
        assert np.count_nonzero(noun_matrix) == np.prod(noun_matrix.shape)
        print("Done filling noun matrix!")
        return noun_matrix

    def create_verb_cube(self, arg):
        assert arg in ['subj', 'obj']
        pass


class SICKDataset(Dataset):
    def __init__(self, task_fn: str):
        self.preproc = SICKPreprocessor(task_fn)

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int):
        pass


task_fn = '/homes/gjw30/ExpCode/compdisteval/experiment_data/SICK/SICK.txt'
folder = '/import/gijs-shared/gijs'
data_folder = os.path.join(folder, 'verb_data')
space_fn = os.path.join(folder, 'spaces/tensor_skipgram_vector_spaces/skipgram_100_nouns.txt')
verb_dict_fn = os.path.join(folder, 'verb_data/verb_counts_all_corpus_verbs_dict.p')
verbs_fn = os.path.join(folder, 'verb_data/sick_verbs_full.txt')
preproc_fn = os.path.join(folder, 'verb_data/preproc_sick_verbcounts.p')

my_preproc = Preprocessor(preproc_fn, space_fn, verb_dict_fn, verbs_fn)
lower2upper = my_preproc.preproc['l2u']
allnouns = set(lower2upper.keys())
allverbs = set(my_preproc.preproc['verb']['i2v'])

sick_preproc = SICKPreprocessor(task_fn, allnouns, allverbs)
noun_matrix = sick_preproc.create_noun_matrix(space_fn, lower2upper)
verb_subj_cube = sick_preproc.create_verb_cube(arg='subj')
verb_obj_cube = sick_preproc.create_verb_cube(arg='obj')
