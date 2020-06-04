"""Evaluation code; dataset."""
from torch.utils.data import Dataset
from tensorskipgram.evaluation.sick import SICK
from nltk.stem import WordNetLemmatizer


class SICKPreprocessor(object):
    def __init__(self, fn):
        self.sick = SICK(fn)
        self.lemmatiser = WordNetLemmatizer()
        self.index2verb, self.verb2index = self.create_verb_indexer()
        self.index2word, self.word2index = self.create_word_indexer()

    def create_verb_indexer(self):
        verbs = [self.lemmatiser.lemmatize(v, pos='v')
                 for v in self.sick.verb_vocab]
        lemma_corrector = {'strip': 'stripe', 'rid': 'ride', 'shin': 'shine',
                           'frolic': 'frolick', 'scar': 'scare',
                           'star': 'stare', 'wakeboarding': 'wakeboard',
                           'unstitching': 'unstitch', 'wad': 'wade',
                           'stag': 'stage', 'waterskiing': 'waterski'}
        verbs = [lemma_corrector[v] if v in lemma_corrector
                 else v for v in verbs]
        excepted_verbs = ['dash', 'consist', 'extend', 'bore', 'box']
        index2verb = [v for v in verbs if v not in excepted_verbs]
        verb2index = {v: i for i, v in enumerate(index2verb)}
        return index2verb, verb2index

    def create_word_indexer(self):
        words = self.sick.noun_vocab
        word_corrector = {}
        words = [word_corrector[w] if w in word_corrector
                 else w for w in words]
        excepted_words = []
        index2word = [w for w in words if w not in excepted_words]
        word2index = {w: i for i, w in enumerate(index2word)}
        return index2word, word2index

    def index_parse(split_parse):
        pass


class SICKDataset(Dataset):
    def __init__(self, task_fn: str):
        self.preproc = SICKPreprocessor(task_fn)

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int):
        pass

task_fn = '/homes/gjw30/ExpCode/compdisteval/experiment_data/SICK/SICK.txt'
