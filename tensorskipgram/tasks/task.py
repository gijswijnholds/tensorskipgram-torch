from typing import List, TypeVar, Generic, Tuple, Callable


Sample = TypeVar('Sample')
Samples = List[Sample]


class Task(Generic[Sample]):
    def __init__(self, name: str, data: Samples,
                 get_nouns: Callable[Sample, List[str]],
                 get_verbs: Callable[Sample, List[str]]) -> None:
        self._name = name
        self._data = data
        self._nouns = list(set([get_nouns(s1, s2) for (s1, s2, sc) in data]))
        self._verbs = list(set([get_verbs(s1, s2) for (s1, s2, sc) in data]))

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> Samples:
        return self._data

    @property
    def nouns(self) -> Samples:
        return self._nouns

    @property
    def verbs(self) -> Samples:
        return self._verbs


SimilaritySample = Tuple[List[str], List[str], float]
ClassLabel = Tuple[List[str], List[str], int]

SimilarityTask = Task[SimilaritySample]
DisambiguationTask = Task[ClassSample]


class WordTag(object):
    def __init__(self, word, tag):
        self.word = word
        self.tag = tag

    def __eq__(self, other):
        return self.word == other.word and self.tag == other.tag

    def __hash__(self):
        return hash(self.word+self.tag)

    def __str__(self):
        return "%s : %s" % (self.word, self.tag)


class Tag(object):
    NOUN = "NN"
    VERB = "VB"
    ADJ = "JJ"
    ADV = "RB"