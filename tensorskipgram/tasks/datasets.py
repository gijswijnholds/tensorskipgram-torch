"""Placeholder for the various datasets that we evaluate on."""
from tensorskipgram.tasks.task import SimilarityTask, DisambiguationTask
from tensorskipgram.tasks.task import SimilaritySample
from tensorskipgram.tasks.task import Tag, WordTag
import numpy as np
from typing import List, Callable


def get_intransitive_nouns(s1, s2) -> List[str]:
    return [s1[0], s2[0]]


def get_intransitive_verbs(s1, s2) -> List[str]:
    return [s1[1], s2[1]]


def load_intransitive_sentence_data(path: str, splitter: str, processLine: Callable) -> List[SimilaritySample]:
    with open(path, 'r') as f:
        lines = f.readlines()

    realLines = lines[1:]
    tsData = {}
    # take only the last five elements (subject1, verb1, subject2, verb2, score)
    for line in realLines:
        line = line.strip()
        (subj1, verb1, subj2, verb2, score) = processLine(line.split(splitter))
        tsData.setdefault(((subj1, verb1), (subj2, verb2)), []).append(float(score))
    avgTsData = {k: np.mean(tsData[k]) for k in tsData}
    return [(sent1, sent2, avgScore) for ((sent1, sent2), avgScore)
            in avgTsData.items()]


def get_transitive_nouns(s1, s2) -> List[str]:
    return [s1[0], s1[2], s2[0], s2[2]]


def get_transitive_verbs(s1, s2) -> List[str]:
    return [s1[1], s2[1]]


def load_transitive_sentence_data(path: str, splitter: str, processLine: Callable) -> List[SimilaritySample]:
    with open(path, 'r') as f:
        lines = f.readlines()

    realLines = lines[1:]
    tsData = {}

    # take only the last seven elements (subject1, verb1, object1, subject2, verb2, object2, score)
    for line in realLines:
        line = line.strip()
        (subj1, verb1, obj1, subj2, verb2, obj2, score) = processLine(line.split(splitter))
        tsData.setdefault(((subj1, verb1, obj1),
                           (subj2, verb2, obj2)),
                          []).append(float(score))

    avgTsData = {k: np.mean(tsData[k]) for k in tsData}

    return [(sent1, sent2, avgScore) for ((sent1, sent2), avgScore)
            in avgTsData.items()]


def get_ellipsis_nouns(s1, s2) -> List[str]:
    # if you want only subjects/objects, uncomment this
    # return [s1[0], s1[2], s1[4], s2[0], s2[2], s2[4]]
    return [s1[0], s1[2], s1[3], s1[4], s1[5], s1[6],
            s2[0], s2[2], s2[3], s2[4], s2[5], s2[6]]


def get_ellipsis_verbs(s1, s2) -> List[str]:
    return [s1[1], s2[1]]


def load_ellipsis_sentence_data(path: str, splitter: str, processLine: Callable) -> List[SimilaritySample]:
    with open(path, 'r') as f:
        lines = f.readlines()

    realLines = lines[1:]
    rawEllData = {}

    # participant verb subject object landmark input hilo
    # take only the 2-5 elements (verb, subject, object, landmark, input)
    for line in realLines:
        line = line.strip()
        (sent1, sent2, score) = processLine(line.split(splitter))
        rawEllData.setdefault((sent1, sent2), []).append(score)

    avgEllData = {k: np.mean(rawEllData[k]) for k in rawEllData}

    return [(s1, s2, avgScore) for ((s1, s2), avgScore)
            in avgEllData.items()]


def lemmatise(lemmamap, word):
    try:
        newWord = lemmamap[word]
    except KeyError:
        newWord = word
    return newWord


class ML2008(SimilarityTask):
    pass


def create_ml2008(ml2008_path: str = 'ML2008/ML2008.txt') -> ML2008:
    def process_line_ml2008(ln):
        lemmaMap_ml2008 = {'papers': 'Papers', 'sniffing': 'Sniffing'}
        (verb1, subj, verb2, score) = ln[1:5]
        verb1WT = WordTag(lemmatise(lemmaMap_ml2008, verb1), Tag.VERB)
        subjWT = WordTag(lemmatise(lemmaMap_ml2008, subj), Tag.NOUN)
        verb2WT = WordTag(lemmatise(lemmaMap_ml2008, verb2), Tag.VERB)
        return (subjWT, verb1WT, subjWT, verb2WT, score)
    name = "ML2008"
    data = load_intransitive_sentence_data(ml2008_path, ' ', process_line_ml2008)
    return ML2008(name, data, get_intransitive_nouns, get_intransitive_verbs)


exp2prefix = {"ML2008": "ML08-", "ML2010": "ML10-", "GS2011": "GS-",
              "KS2013": "SK-", "KS2014": "KS-", "ELLDIS": "ELLDIS-",
              "ELLSIM": "ELLSIM-"}


class ML2010(SimilarityTask):
    pass


def create_ml2010(ml2010_path: str = 'ML2010/ML2010.txt') -> ML2010:
    def process_line_ml2010(ln):
        lemmaMap_ml2010 = {}
        (obj1, verb1, obj2, verb2, score) = ln[3:]
        if ln[1] != 'verbobjects':
            return ("", "", "", "", 0.0)
        verb1WT = WordTag(lemmatise(lemmaMap_ml2010, verb1), Tag.VERB)
        obj1WT = WordTag(lemmatise(lemmaMap_ml2010, obj1), Tag.NOUN)
        verb2WT = WordTag(lemmatise(lemmaMap_ml2010, verb2), Tag.VERB)
        obj2WT = WordTag(lemmatise(lemmaMap_ml2010, obj2), Tag.NOUN)
        return (obj1WT, verb1WT, obj2WT, verb2WT, score)
    name = "ML2010"
    data = load_intransitive_sentence_data(ml2010_path, ' ', process_line_ml2010)
    data = [d for d in data if not isinstance(d[0][0], str)]
    return ML2010(name, data, get_intransitive_nouns, get_intransitive_verbs)


class GS2011(SimilarityTask):
    pass


class GS2011Dis(DisambiguationTask):
    pass


def create_gs2011(gs2011_path: str = 'GS2011/GS2011data.txt') -> GS2011:
    def process_line_gs2011(ln):
        lemmaMap_gs2011 = {'papers': 'Papers', 'sniffing': 'Sniffing'}
        (verb1, subj, obj, verb2, score) = ln[1:6]
        verb1WT = WordTag(lemmatise(lemmaMap_gs2011, verb1), Tag.VERB)
        subjWT = WordTag(lemmatise(lemmaMap_gs2011, subj), Tag.NOUN)
        objWT = WordTag(lemmatise(lemmaMap_gs2011, obj), Tag.NOUN)
        verb2WT = WordTag(lemmatise(lemmaMap_gs2011, verb2), Tag.VERB)
        return (subjWT, verb1WT, objWT, subjWT, verb2WT, objWT, score)
    name = "GS2011"
    data = load_transitive_sentence_data(gs2011_path, ' ', process_line_gs2011)
    return GS2011(name, data, get_transitive_nouns, get_transitive_verbs)


class KS2013(SimilarityTask):
    pass


class KS2013Dis(DisambiguationTask):
    pass


def create_ks2013(ks2013_path: str = 'KS2013/KS2013-CoNLL.txt') -> KS2013:
    def process_line_ks2013(ln):
        lemmaMap_ks2013 = {'englishman': 'english'}
        (subj, verb1, verb2, obj, score) = ln[3], ln[4], ln[5], ln[7], ln[8]
        verb1WT = WordTag(lemmatise(lemmaMap_ks2013, verb1), Tag.VERB)
        subjWT = WordTag(lemmatise(lemmaMap_ks2013, subj), Tag.NOUN)
        objWT = WordTag(lemmatise(lemmaMap_ks2013, obj), Tag.NOUN)
        verb2WT = WordTag(lemmatise(lemmaMap_ks2013, verb2), Tag.VERB)
        return (subjWT, verb1WT, objWT, subjWT, verb2WT, objWT, score)
    name = "KS2013"
    data = load_transitive_sentence_data(ks2013_path, ' ', process_line_ks2013)
    return KS2013(name, data, get_transitive_nouns, get_transitive_verbs)


class KS2014(SimilarityTask):
    pass


def create_ks2014(ks2014_path: str = 'KS2014/KS2014.txt') -> KS2014:
    def process_line_ks2014(ln):
        (subj1, verb1, obj1, subj2, verb2, obj2, score) = ln[1:]
        subj1WT = WordTag(subj1, Tag.NOUN)
        verb1WT = WordTag(verb1, Tag.VERB)
        obj1WT = WordTag(obj1, Tag.NOUN)
        subj2WT = WordTag(subj2, Tag.NOUN)
        verb2WT = WordTag(verb2, Tag.VERB)
        obj2WT = WordTag(obj2, Tag.NOUN)
        return (subj1WT, verb1WT, obj1WT, subj2WT, verb2WT, obj2WT, score)
    name = "KS2014"
    data = load_transitive_sentence_data(ks2014_path, ' ', process_line_ks2014)
    return KS2014(name, data, get_transitive_nouns, get_transitive_verbs)


class ELLDIS(SimilarityTask):
    pass


class ELLDISDis(DisambiguationTask):
    pass


def create_elldis(elldis_path: str = 'WS2018/ELLDIS_CORRECTED.txt') -> ELLDIS:
    def process_line_elldis(ln):
        lemmaMap_elldis = {'airlines': 'Airlines', 'papers': 'Papers', 'sniffing': 'Sniffing'}
        sent1, sent2, score = ln[1], ln[2], float(ln[3])
        (s1, v1, o1, and1, s2, aux1, aux2) = sent1.split(' ')
        (s3, v3, o3, and2, s4, aux3, aux4) = sent2.split(' ')
        s1WT = WordTag(lemmatise(lemmaMap_elldis, s1), Tag.NOUN)
        s2WT = WordTag(lemmatise(lemmaMap_elldis, s2), Tag.NOUN)
        s3WT = WordTag(lemmatise(lemmaMap_elldis, s3), Tag.NOUN)
        s4WT = WordTag(lemmatise(lemmaMap_elldis, s4), Tag.NOUN)
        o1WT = WordTag(lemmatise(lemmaMap_elldis, o1), Tag.NOUN)
        o3WT = WordTag(lemmatise(lemmaMap_elldis, o3), Tag.NOUN)
        v1WT = WordTag(lemmatise(lemmaMap_elldis, v1), Tag.VERB)
        v3WT = WordTag(lemmatise(lemmaMap_elldis, v3), Tag.VERB)
        finalTriple = ((s1WT, v1WT, o1WT, and1, s2WT, aux1, aux2),
                       (s3WT, v3WT, o3WT, and2, s4WT, aux3, aux4), score)
        return finalTriple
    name = "ELLDIS"
    data = load_ellipsis_sentence_data(elldis_path, '\t', process_line_elldis)
    return ELLDIS(name, data, get_ellipsis_nouns, get_ellipsis_verbs)


class ELLSIM(SimilarityTask):
    pass


def create_ellsim(ellsim_path: str = 'WS2018/ELLSIM_CORRECTED.txt') -> ELLDIS:
    def process_line_ellsim(ln):
        lemmaMap_ellsim = {}
        sent1, sent2, score = ln[1], ln[2], float(ln[3])
        (s1, v1, o1, and1, s2, aux1, aux2) = sent1.split(' ')
        (s3, v3, o3, and2, s4, aux3, aux4) = sent2.split(' ')
        s1WT = WordTag(lemmatise(lemmaMap_ellsim, s1), Tag.NOUN)
        s2WT = WordTag(lemmatise(lemmaMap_ellsim, s2), Tag.NOUN)
        s3WT = WordTag(lemmatise(lemmaMap_ellsim, s3), Tag.NOUN)
        s4WT = WordTag(lemmatise(lemmaMap_ellsim, s4), Tag.NOUN)
        o1WT = WordTag(lemmatise(lemmaMap_ellsim, o1), Tag.NOUN)
        o3WT = WordTag(lemmatise(lemmaMap_ellsim, o3), Tag.NOUN)
        v1WT = WordTag(lemmatise(lemmaMap_ellsim, v1), Tag.VERB)
        v3WT = WordTag(lemmatise(lemmaMap_ellsim, v3), Tag.VERB)
        finalTriple = ((s1WT, v1WT, o1WT, and1, s2WT, aux1, aux2),
                       (s3WT, v3WT, o3WT, and2, s4WT, aux3, aux4), score)
        return finalTriple
    name = "ELLSIM"
    data = load_ellipsis_sentence_data(ellsim_path, '\t', process_line_ellsim)
    return ELLSIM(name, data, get_ellipsis_nouns, get_ellipsis_verbs)
