"""The main code for extracting dependency data from a parsed corpus, and
indexing it for use in a tensor-skipgram model."""
from collections import Counter
from tensorskipgram.data.util import dump_obj_fn
from tensorskipgram.data.preprocessing import Preprocessor
from tensorskipgram.data.ukwackypedia \
    import UKWackypedia, get_verb_args, merge_verb_triples
from tensorskipgram.data.config \
    import ukwackypedia_split_folder, svo_triples_fn, verblist_fn
from tensorskipgram.tasks.datasets \
    import create_ml2008, create_ml2010, create_gs2011, create_ks2013
from tensorskipgram.tasks.datasets \
    import create_ks2014, create_elldis, create_ellsim

Path = str
Fn = str


def extract_svo_triples(corpus_folder: Path, out_fn: Fn) -> None:
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


def extract_verbs(out_fn) -> None:
    """Extract all verbs and save to text file."""
    all_verbs = list(set(create_ml2008().verbs + create_ml2010().verbs
                         + create_gs2011().verbs + create_ks2013().verbs
                         + create_ks2014().verbs + create_elldis().verbs
                         + create_ellsim().verbs))
    with open(out_fn, 'w') as out_file:
        out_file.write('\n'.join(all_verbs))


def prepare_training_data_nouns() -> None:
    """Prepare the training data for a regular noun skipgram model."""
    pass


def prepare_training_data_verbs() -> None:
    """Extract and save all intances of (s,v,o) triples from a corpus."""
    pass


if __name__ == '__main__':
    # Step 1: take a corpus and extract svo triples, store them away
    extract_svo_triples(ukwackypedia_split_folder, svo_triples_fn)
    extract_verbs(verblist_fn)
    # Step 2: index them properly, using an indexer that we have somehow obtained?
    # Step 3: store the non-indexed and indexed data
