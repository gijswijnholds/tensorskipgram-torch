import os
import multiprocessing as mp
from multiprocessing import Pool
from typing import Tuple, List, Callable, Optional
from tqdm import tqdm
from tensorskipgram.data.util import dump_obj_fn
from collections import Counter


class UKWackypedia(object):
    def __init__(self, root_folder: str, transform: Optional[Callable] = None, reduce: bool = False):
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
    def parse_all_files(self, num_workers: int, multi: bool = False):
        if multi:
            pool = mp.Pool(num_workers)
            all_results = pool.map(self.parse_file, self.file_list)
            all_results_flat = [r for rs in all_results for r in rs]
            return all_results_flat
        else:
            all_results = []
            num_fn = len(self.file_list)
            for i, fn in enumerate(self.file_list):
                print(f'Processing file {i}/{num_fn}...')
                all_results.append(self.parse_file(fn))
            return all_results


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


my_verb_corpus = UKWackypedia('/import/gijs-shared/gijs/corpora/ukwackypedia_split',
                              transform=get_verb_args, reduce=True)
test_fn = my_verb_corpus.file_list[0]
all_vargs = my_verb_corpus.parse_all_files(multi=True)
all_vargs_dict = dict(Counter(all_vargs))
dump_obj_fn(all_vargs_dict,
            '/import/gijs-shared/gijs/verb_data/verb_counts_all_corpus_verbs.p')
