"""The main code for extracting dependency data from a parsed corpus, and
indexing it for use in a tensor-skipgram model."""
from collections import Counter
from tensorskipgram.data.util import dump_obj_fn
from tensorskipgram.data.training_data_creator import Preprocessor, DataCreator
from tensorskipgram.data.ukwackypedia \
    import UKWackypedia, get_verb_args, merge_verb_triples
from tensorskipgram.data.config \
    import ukwackypedia_split_folder, svo_triples_fn, verblist_fn
from tensorskipgram.data.config \
    import noun_space_fn, preproc_fn, subj_data_fn, obj_data_fn
from tensorskipgram.tasks.datasets \
    import create_ml2008, create_ml2010, create_gs2011, create_ks2013
from tensorskipgram.tasks.datasets \
    import create_ks2014, create_elldis, create_ellsim


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


def prepare_training_data_verbs(preprocessor_fn: str, space_fn: str,
                                triples_fn: str, verbs_fn: str,
                                subj_fn: str, obj_fn: str) -> None:
    """Create and prepare training data for a matrix verb skipgram model."""
    preproc = Preprocessor(preprocessor_fn, space_fn,
                           triples_fn, verbs_fn)
    data_creator = DataCreator(preproc, subj_fn, obj_fn)
    data_creator.setup()


if __name__ == '__main__':
    extract_svo_triples(ukwackypedia_split_folder, svo_triples_fn)
    extract_verbs(verblist_fn)
    prepare_training_data_verbs(preproc_fn, noun_space_fn, svo_triples_fn,
                                verblist_fn, subj_data_fn, obj_data_fn)
