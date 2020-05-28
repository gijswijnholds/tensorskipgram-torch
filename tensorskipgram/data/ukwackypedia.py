import os
import multiprocessing as mp
from multiprocessing import Pool
from typing import Tuple, List, Callable, Optional
from tqdm import tqdm
from tensorskipgram.data.util import dump_obj_fn
from collections import Counter


class UKWackypedia(object):
    def __init__(self, root_folder: str, transform: Optional[Callable] = None,
                 reduce: bool = False):
        self.file_list = self.load_file_list(root_folder)
        self.transform = transform
        self.reduce = reduce

    def load_file_list(self, folder):
        # folder should be /import/gijs-shared/gijs/corpora/ukwackypedia_split
        return [os.path.join(folder, fn) for fn in os.listdir(folder)]

    def parse_file(self, fn: str):
        print("Opening file...")
        all_text = open(fn, 'r', encoding='utf-8').read()
        print("Cleaning full text...")
        all_sents = all_text.split('</s>\n<s>')
        all_sents[0] = all_sents[0].split('<s>')[1]
        all_sents[-1] = all_sents[-1].split('</s>')[0]
        print("Parsing sentences...")
        if self.transform:
            final_sents = [self.transform(s) for s in tqdm(all_sents)]
            if self.reduce:
                final_sents = [w for s in final_sents for w in s]
        print("Done parsing file!")
        return final_sents

    def parse_all_files(self, num_workers: int):
        pool = mp.Pool(num_workers)
        all_results = pool.map(self.parse_file, self.file_list)
        all_results_flat = [r for rs in all_results for r in rs]
        return all_results_flat


def parse_sentence_base(s: str):
    return [tuple(wls.split('\t')) for wls in s.strip().split('\n')]


def get_verb_args(s: str):
    wls = [word_info.split('\t') for word_info in s.strip().split('\n')]
    wls = [wl for wl in wls if len(wl) == 6]
    vargs = []
    for token, lemma, pos, token_id, head_id, dep in wls:
        if pos[:1] == 'V':
            subj_lem, obj_lem = None, None
            for token1, lemma1, pos1, token_id1, head_id1, dep1 in wls:
                if head_id1 == token_id and dep1 == 'SBJ':
                    subj_lem = lemma1
                elif head_id1 == token_id and dep1 == 'OBJ':
                    obj_lem = lemma1
                elif subj_lem and obj_lem:
                    vargs.append((lemma, subj_lem, obj_lem))
                    break
    return vargs


def merge_verb_triples(all_triples_dict):
    verb_dict = {}
    for (v, s, o) in all_triples_dict:
        if v in verb_dict:
            if (s, o) in verb_dict[v]:
                verb_dict[v][(s, o)] += all_triples_dict[(v, s, o)]
            else:
                verb_dict[v][(s, o)] = all_triples_dict[(v, s, o)]
        else:
            verb_dict[v] = {(s, o): all_triples_dict[(v, s, o)]}
    return verb_dict


if __name__ == '__main__':
    my_folder = '/import/gijs-shared/gijs'
    print("Creating corpus reader...")
    my_verb_corpus = UKWackypedia(os.path.join(my_folder,
                                               'corpora/ukwackypedia_split'),
                                  transform=get_verb_args, reduce=True)
    print("Processing corpus files...")
    all_vargs = my_verb_corpus.parse_all_files(num_workers=40)
    print("Processing counts...")
    all_vargs_dict = dict(Counter(all_vargs))
    print("Merging counts by verb...")
    all_verbs_dict = merge_verb_triples(all_vargs_dict)
    print("Saving verb dict...")
    dump_obj_fn(all_verbs_dict,
                os.path.join(my_folder,
                             'verb_data/verb_counts_all_corpus_verbs_dict.p'))
