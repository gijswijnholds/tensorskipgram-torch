"""Main code for extracting dependency data from a parsed corpus."""
from collections import Counter
from typing import List
from tensorskipgram.preprocessing.util import dump_obj_fn
from tensorskipgram.preprocessing.training_data_creator import Preprocessor, DataCreator
from tensorskipgram.preprocessing.ukwackypedia \
    import UKWackypedia, get_verb_args, merge_verb_triples
from tensorskipgram.config \
    import (ukwackypedia_split_folder, svo_triples_fn, verblist_fn,
            noun_space_fn, preproc_fn, subj_data_fn, obj_data_fn)
from tensorskipgram.tasks.datasets \
    import (create_men_verb, create_simlex_verb, create_verbsim, create_simverbdev,
            create_simverbtest)
from tensorskipgram.config import (menverb_path, simlex_path, verbsim_path,
                                   simverbdev_path, simverbtest_path, relpron_path)
from tensorskipgram.tasks.datasets \
    import (create_ml2008, create_ml2010, create_gs2011, create_ks2013,
            create_ks2014, create_elldis, create_ellsim)
from tensorskipgram.config import (ml2008_path, ml2010_path, gs2011_path, ks2013_path,
                                   ks2014_path, elldis_path, ellsim_path)


def extract_svo_triples(corpus_folder: str, out_fn: str) -> None:
    """Extract and save all intances of (s,v,o) triples from a corpus."""
    print("Creating corpus reader...")
    my_verb_corpus = UKWackypedia(corpus_folder,
                                  transform=get_verb_args)
    print("Processing corpus files...")
    all_vargs = my_verb_corpus.parse_all_files(num_workers=40)
    print("Processing counts...")
    all_vargs_dict = dict(Counter(all_vargs))
    print("Merging counts by verb...")
    all_verbs_dict = merge_verb_triples(all_vargs_dict)
    print("Saving verb dict...")
    dump_obj_fn(all_verbs_dict, out_fn)


def get_relpron_verbs(relpron_fn: str) -> List[str]:
    with open(relpron_fn, 'r') as in_file:
        lines = in_file.readlines()
    verbs = []
    for ln in lines:
        ln = ln.strip().split()
        rel_type = ln[0]
        if rel_type == 'SBJ':
            verbs.append(ln[4][:-2])
        elif rel_type == 'OBJ':
            verbs.append(ln[5][:-2])
    return verbs


def extract_verbs(out_fn) -> None:
    """Extract all verbs and save to text file."""
    dataset_verbwts = (create_men_verb(menverb_path).verbs +
                       create_verbsim(verbsim_path).verbs +
                       create_simlex_verb(simlex_path).verbs +
                       create_simverbdev(simverbdev_path).verbs +
                       create_simverbtest(simverbtest_path).verbs +
                       create_ml2008(ml2008_path).verbs +
                       create_ml2010(ml2010_path).verbs +
                       create_gs2011(gs2011_path).verbs +
                       create_ks2013(ks2013_path).verbs +
                       create_ks2014(ks2014_path).verbs +
                       create_elldis(elldis_path).verbs +
                       create_ellsim(ellsim_path).verbs)
    dataset_verbs = [v.word for v in dataset_verbwts]
    all_verbs = list(set(dataset_verbs + get_relpron_verbs(relpron_path)))
    with open(out_fn, 'w') as out_file:
        out_file.write('\n'.join(all_verbs))


def prepare_training_data_nouns() -> None:
    """Prepare the training data for a regular noun skipgram model."""
    pass


def prepare_training_data_verbs(preprocessor_fn: str, space_fn: str,
                                triples_fn: str, verbs_fn: str,
                                subj_fn: str, obj_fn: str,
                                neg_samples: int) -> None:
    """Create and prepare training data for a matrix verb skipgram model."""
    preproc = Preprocessor(preprocessor_fn, space_fn,
                           triples_fn, verbs_fn)
    data_creator = DataCreator(preproc, subj_fn, obj_fn)
    data_creator.setup(neg_samples=neg_samples)


def main():
    extract_svo_triples(ukwackypedia_split_folder, svo_triples_fn)
    extract_verbs(verblist_fn)
    prepare_training_data_verbs(preproc_fn, noun_space_fn, svo_triples_fn,
                                verblist_fn, subj_data_fn, obj_data_fn, 10)
