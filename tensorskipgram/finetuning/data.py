"""Evaluation code; dataset."""
import os
import torch
from torch import LongTensor
import numpy as np
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import Dataset
from tensorskipgram.evaluation.sick import SICK
from nltk.stem import WordNetLemmatizer
from tensorskipgram.data.util import load_obj_fn, dump_obj_fn

UNK_TOKEN = 'UNK'

SentenceData = Tuple[LongTensor, LongTensor, LongTensor, LongTensor]
EntailmentLabel = int
SimilarityLabel = float
entailment_map = {'NEUTRAL': 0., 'CONTRADICTION': 1., 'ENTAILMENT': 2.}


def create_indexer(vocab_mapper):
    """Create an indexer from a vocab2vocab mapper.

    Remove the unknown token as we do not want to incorporate it in our network.
    """
    idx_mapper = {w: i for i, w in enumerate(sorted(list(set(vocab_mapper.values()))))}
    word2index = {w: idx_mapper[vocab_mapper[w]] for w in vocab_mapper}
    return word2index


def open_model(folder, arg):
    model_path = os.path.join(folder, f'verb_data/matrixskipgram_{arg}')
    bs, lr, e = 100, 0.001, 5
    return torch.load(model_path+f'_bs={bs}_lr={lr}_epoch{e}.p',
                      map_location='cpu')


def get_verb_matrices(model, arg):
    return model.functor_embedding.weight


class SICKPreprocessor(object):
    def __init__(self, fn, data_fn, nouns, verbs):
        self.sick = SICK(fn, data_fn)
        self.lemmatiser = WordNetLemmatizer()
        self.nouns = nouns
        self.verbs = verbs
        self.verb2verb, self.verb2index = self.create_verb_indexer()
        self.word2word, self.word2index = self.create_word_indexer()
        self.space = None
        # self.index2verb = {i: v for v, i in self.verb2index.items()}
        # self.index2word = {i: v for v, i in self.word2index.items()}

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
        self.space = space
        indices = sorted(list(set(self.word2index.values())))
        noun_matrix = np.zeros((len(indices), 100))
        print("Filling noun matrix...")
        for w in self.word2index:
            noun_matrix[self.word2index[w]] = space[lower2upper[self.word2word[w]]]
        assert np.count_nonzero(noun_matrix) == np.prod(noun_matrix.shape)
        print("Done filling noun matrix!")

        return torch.tensor(noun_matrix, dtype=torch.float32)

    def create_verb_cube(self, arg, v2i, folder,
                         setting='skipgram', verb_counts=None, lower2upper=None):
        assert arg in ['subj', 'obj']
        assert self.space is not None
        indices = sorted(list(set(self.verb2index.values())))
        verb_cube = np.zeros((len(indices), 100, 100))
        if setting == 'skipgram':
            model = open_model(folder, arg)
            verb_space = get_verb_matrices(model, arg)
            for v in self.verb2index:
                verb_cube[self.verb2index[v]] = verb_space[v2i[self.verb2verb[v]]].reshape(100, 100).detach().numpy()
        elif setting == 'relational':
            for verb in tqdm(self.verb2index):
                if self.verb2verb[verb] not in verb_counts:
                    print(verb)
                    continue
                verb_matrix = np.zeros((100, 100))
                v_counts = verb_counts[self.verb2verb[verb]]
                for (s, o) in v_counts:
                    if lower2upper[s] in self.space and lower2upper[o] in self.space:
                        verb_matrix += np.outer(self.space[lower2upper[s]],
                                                self.space[lower2upper[o]])
                if arg == 'obj':
                    verb_matrix = verb_matrix.T
                verb_cube[self.verb2index[verb]] = verb_matrix
        elif setting == 'kronecker':
            for verb in tqdm(self.verb2index):
                if self.verb2verb[verb] not in self.space:
                    continue
                verb_vector = self.space[self.verb2verb[verb]]
                verb_matrix = np.outer(verb_vector, verb_vector)
                verb_cube[self.verb2index[verb]] = verb_matrix
        return torch.tensor(verb_cube, dtype=torch.float32)


def create_data_single(ws, vargs) -> SentenceData:
    """Sort indexed training data.

    Given a single list of word indices and verb-argument indices, sort the
    indices into the four types of combinations (word, verb-subj, verb-obj,
    verb-subj-obj).
    """
    words, verb_subj, verb_obj, verb_trans = ws, [], [], []
    for (v, ss, objs) in vargs:
        if ss != [] and objs == []:
            if len(ss) == 2:
                words.append(ss[1])
            verb_subj.append((v, ss[0]))
        elif ss == [] and objs != []:
            verb_obj.append((v, objs[0]))
        else:
            if len(ss) == 2:
                words.append(ss[1])
            verb_trans.append((v, ss[0], objs[0]))
    return torch.tensor(words), torch.tensor(verb_subj), torch.tensor(verb_obj), torch.tensor(verb_trans)


def create_data_pair_noun(preproc: SICKPreprocessor, s1, s2, el, rl, setting):
    data1 = [preproc.word2index[w] for w in s1.split()
             if w in preproc.word2index]
    data2 = [preproc.word2index[w] for w in s2.split()
             if w in preproc.word2index]
    if setting == 'relatedness':
        y = torch.tensor([rl])
    elif setting == 'entailment':
        y = torch.tensor([entailment_map[el]])
    return torch.tensor(data1), torch.tensor(data2), y


def create_data_pair(preproc: SICKPreprocessor, s1, s2, el, rl, setting):
    parse1 = preproc.index_parse(preproc.sick.parse_data[s1])
    data1 = create_data_single(*parse1)
    parse2 = preproc.index_parse(preproc.sick.parse_data[s2])
    data2 = create_data_single(*parse2)
    if setting == 'relatedness':
        y = torch.tensor([rl])
    elif setting == 'entailment':
        y = torch.tensor([entailment_map[el]])
    return data1, data2, y


def create_data_pairs_noun(preproc: SICKPreprocessor, setting='relatedness'):
    train_data = [create_data_pair_noun(preproc, s1, s2, el, rl, setting=setting)
                  for (s1, s2, el, rl, set) in preproc.sick.data
                  if set == 'TRAIN']
    dev_data = [create_data_pair_noun(preproc, s1, s2, el, rl, setting=setting)
                for (s1, s2, el, rl, set) in preproc.sick.data
                if set == 'TRIAL']
    test_data = [create_data_pair_noun(preproc, s1, s2, el, rl, setting=setting)
                 for (s1, s2, el, rl, set) in preproc.sick.data
                 if set == 'TEST']
    return {'train': train_data, 'dev': dev_data, 'test': test_data}


def create_data_pairs(preproc: SICKPreprocessor, setting='relatedness'):
    train_data = [create_data_pair(preproc, s1, s2, el, rl, setting=setting)
                  for (s1, s2, el, rl, set) in preproc.sick.data
                  if set == 'TRAIN']
    dev_data = [create_data_pair(preproc, s1, s2, el, rl, setting=setting)
                for (s1, s2, el, rl, set) in preproc.sick.data
                if set == 'TRIAL']
    test_data = [create_data_pair(preproc, s1, s2, el, rl, setting=setting)
                 for (s1, s2, el, rl, set) in preproc.sick.data
                 if set == 'TEST']
    return {'train': train_data, 'dev': dev_data, 'test': test_data}


class SICKDatasetNouns(Dataset):
    def __init__(self, data_fn: str, setting: str, data_sort='relatedness'):
        self.data_fn = data_fn
        self.setting = setting
        self.data_sort = data_sort
        if os.path.exists(data_fn):
            print("Data pairs found on disk, loading...")
            self.data_pairs = load_obj_fn(data_fn)[self.setting]
        else:
            print("Data pairs not found, please run create_data with a preproc.")
            self.data_pairs = None

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> LongTensor:
        return self.data_pairs[idx]

    def create_data(self, preproc: SICKPreprocessor):
        all_data = create_data_pairs_noun(preproc, setting=self.data_sort)
        dump_obj_fn(all_data, self.data_fn)
        self.data_pairs = all_data[self.setting]


class SICKDataset(Dataset):
    def __init__(self, data_fn: str, setting: str, filter: bool = False, data_sort ='relatedness'):
        self.data_fn = data_fn
        self.setting = setting
        self.filter = filter
        self.data_sort = data_sort
        if os.path.exists(data_fn):
            print("Data pairs found on disk, loading...")
            self.data_pairs = load_obj_fn(data_fn)[self.setting]
            if self.filter:
                self.data_pairs = [(s1, s2, l) for (s1, s2, l) in self.data_pairs
                                   if self.has_verbs(s1) and self.has_verbs(s2)]
        else:
            print("Data pairs not found, please run create_data with a preproc.")
            self.data_pairs = None

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[SentenceData, SentenceData, SimilarityLabel]:
        return self.data_pairs[idx]

    def has_verbs(self, s):
        return (s[1].dim() == 2 or s[2].dim() == 2 or s[3].dim() == 2)

    def create_data(self, preproc: SICKPreprocessor):
        all_data = create_data_pairs(preproc, setting=self.data_sort)
        dump_obj_fn(all_data, self.data_fn)
        self.data_pairs = all_data[self.setting]
