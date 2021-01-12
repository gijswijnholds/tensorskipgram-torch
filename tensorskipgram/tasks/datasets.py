"""Placeholder for the various datasets that we evaluate on."""
from tensorskipgram.tasks.task import (Tag, WordTag, SimilaritySample,
                                       WordSimilaritySample, SimilarityTask,
                                       DisambiguationTask, WordSimilarityTask,
                                       ClassSample)
from tensorskipgram.tasks.util import paragaps_noun_map, paragaps_verb_map
import numpy as np
from typing import List, Callable
from collections import defaultdict

def get_wordsim_nouns(s1, s2) -> List[str]:
    return [s1, s2]


def get_wordsim_verbs(s1, s2) -> List[str]:
    return [s1, s2]


def load_wordsim_data(path, splitter, processLine, skipFirstLine=False):
    with open(path, 'r') as f:
        lines = f.readlines()

    if skipFirstLine:
        lines = lines[1:]

    wsData = []
    # take only the first three elements (w1, w2, score)
    for line in lines:
        line = line.strip()
        (wt1, wt2, score) = processLine(line.split(splitter))
        wsData.append((wt1, wt2, float(score)))

    return wsData


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


class MENVerb(WordSimilarityTask):
    pass


def create_men_verb(path: str = 'MEN/MEN_dataset_lemma_form_full') -> MENVerb:
    men_tag = {"-n": Tag.NOUN, "-j": Tag.ADJ, "-v": Tag.VERB}
    def process_line_men(ln):
        (w1, w2, score) = ln[:3]
        if w1[-2:] != '-v' or w2[-2:] != '-v':
            return ("", "", 0.0)
        wt1 = WordTag(w1[:-2], men_tag[w1[-2:]])
        wt2 = WordTag(w2[:-2], men_tag[w2[-2:]])
        return (wt1, wt2, score)
    name = "MEN-Verb"
    data = load_wordsim_data(path, ' ', process_line_men, skipFirstLine=False)
    data = [d for d in data if not isinstance(d[0], str)]
    return MENVerb(name, data, get_wordsim_nouns, get_wordsim_verbs)



class SimLexVerb(WordSimilarityTask):
    pass


def create_simlex_verb(path: str = 'SimLex-999/SimLex-999.txt') -> SimLexVerb:
    posTagMap_simlex = {'A': Tag.ADJ, 'N': Tag.NOUN, 'V': Tag.VERB}
    lemmaMap_simlex = {'teeth': 'tooth', 'men': 'man', 'august': 'August',
                       'rattle': 'Rattle'}
    def process_line_simlex(ln):
        (w1, w2, pos, score) = ln[:4]
        wt1 = WordTag(lemmatise(lemmaMap_simlex, w1), posTagMap_simlex[pos])
        wt2 = WordTag(lemmatise(lemmaMap_simlex, w2), posTagMap_simlex[pos])
        return (wt1, wt2, score)
    name = "SimLex-Verb"
    data = load_wordsim_data(path, '\t', process_line_simlex, skipFirstLine=True)[777:]
    return SimLexVerb(name, data, get_wordsim_nouns, get_wordsim_verbs)


class VerbSim(WordSimilarityTask):
    pass


def create_verbsim(path: str = 'VerbSim/200601-GWC-130verbpairs.txt') -> VerbSim:
    def process_line_verbsim(ln):
        (rank, w1, w2, score) = ln[:5]
        if w2 == 'figure out':
            w2 = 'calculate'
        wt1 = WordTag(w1, Tag.VERB)
        wt2 = WordTag(w2, Tag.VERB)
        return (wt1, wt2, float(score))
    name = "VerbSim"
    data = load_wordsim_data(path, '\t', process_line_verbsim, skipFirstLine=True)
    return VerbSim(name, data, get_wordsim_nouns, get_wordsim_verbs)


class SimVerbDev(WordSimilarityTask):
    pass


def create_simverbdev(path: str = 'SIMVERB3500/SimVerb-500-dev.txt') -> SimVerbDev:
    def process_line_simverbdev(ln):
        (w1, w2, pos, score, rel) = ln[:5]
        wt1 = WordTag(w1, Tag.VERB)
        wt2 = WordTag(w2, Tag.VERB)
        return (wt1, wt2, score)
    name = "SimVerbDev"
    data = load_wordsim_data(path, '\t', process_line_simverbdev, skipFirstLine=False)
    return SimVerbDev(name, data, get_wordsim_nouns, get_wordsim_verbs)


class SimVerbTest(WordSimilarityTask):
    pass


def create_simverbtest(path: str = 'SIMVERB3500/SimVerb-3000-test.txt') -> SimVerbTest:
    def process_line_simverbdev(ln):
        (w1, w2, pos, score, rel) = ln[:5]
        wt1 = WordTag(w1, Tag.VERB)
        wt2 = WordTag(w2, Tag.VERB)
        return (wt1, wt2, score)
    name = "SimVerbTest"
    data = load_wordsim_data(path, '\t', process_line_simverbdev, skipFirstLine=False)
    return SimVerbTest(name, data, get_wordsim_nouns, get_wordsim_verbs)



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


def create_elldis_dis(elldis_path: str = 'WS2018/ELLDIS_CORRECTED.txt') -> ELLDISDis:
    elldis_base = create_elldis(elldis_path)
    name = "ELLDISDis"
    data_dict = {}
    for (s1, s2, sc) in elldis_base.data:
        data_dict.setdefault(s1, []).append((s2, sc))
    data_dict = {d: data_dict[d] for d in data_dict if len(data_dict[d]) == 2}
    data = []
    for s1 in data_dict:
        (s2, sc2), (s3, sc3) = data_dict[s1]
        if sc2 > sc3:
            data.append((s1, s2, s3))
        elif sc3 > sc2:
            data.append((s1, s3, s2))
    return ELLDISDis(name, data, get_ellipsis_nouns, get_ellipsis_verbs)

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


def load_paragaps_data(path: str, process_line: Callable) -> List[ClassSample]:
    with open(path, 'r') as f:
        lines = f.readlines()

    real_lines = lines[1:]
    pargaps_data = []
    for i in range(0, len(real_lines), 2):
        ln1, ln2 = real_lines[i], real_lines[i+1]
        ln1, ln2 = ln1.strip().split()[1:], ln2.strip().split()[1:]
        assert ln1[0] == ln2[0]
        (sent1, lab1), (sent2, lab2) = process_line(ln1), process_line(ln2)
        landmark_sent = sent1[:3] + sent1[4:]
        if lab1 == 1:
            good_sent, bad_sent = sent1, sent2
        elif lab2 == 1:
            good_sent, bad_sent = sent2, sent1
        sn1 = landmark_sent
        sn2 = landmark_sent[:2] + (good_sent[3],) + landmark_sent[3:]
        sn3 = landmark_sent[:2] + (bad_sent[3],) + landmark_sent[3:]
        pargaps_data.append((sn1, sn2, sn3))
    return pargaps_data


class PARAGAPS(DisambiguationTask):
    pass


def get_paragaps_nouns(s1, s2) -> List[str]:
    return [s1[1], s1[4], s2[1], s2[4]]


def get_paragaps_verbs(s1, s2) -> List[str]:
    return [s1[2], s1[-1], s2[2], s2[-1]]


def create_paragaps(paragaps_path: str = 'PARGAP/pargaps_2020.txt') -> PARAGAPS:
    def process_line_paragaps(ln):
        (adj1, noun1, verb1, verb2, adj2, noun2, verb3), label = ln[:-1], int(ln[-1])
        adj1WT = WordTag(lemmatise(paragaps_noun_map, adj1), Tag.ADJ)
        noun1WT = WordTag(lemmatise(paragaps_noun_map, noun1), Tag.NOUN)
        verb1WT = WordTag(lemmatise(paragaps_verb_map, verb1), Tag.VERB)
        verb2WT = WordTag(lemmatise(paragaps_verb_map, verb2), Tag.VERB)
        adj2WT = WordTag(lemmatise(paragaps_noun_map, adj2), Tag.ADJ)
        noun2WT = WordTag(lemmatise(paragaps_noun_map, noun2), Tag.NOUN)
        verb3WT = WordTag(lemmatise(paragaps_verb_map, verb3), Tag.VERB)
        return (adj1WT, noun1WT, verb1WT, verb2WT, adj2WT, noun2WT, verb3WT), label
    name = "PARAGAPS"
    data = load_paragaps_data(paragaps_path, process_line_paragaps)
    return PARAGAPS(name, data, get_paragaps_nouns, get_paragaps_verbs)
