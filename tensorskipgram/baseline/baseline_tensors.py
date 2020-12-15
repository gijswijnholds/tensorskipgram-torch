import numpy as np
from tqdm import tqdm
import torch
from typing import List, Tuple
from tensorskipgram.preprocessing.training_data_creator import Preprocessor
from tensorskipgram.config import (preproc_fn, svo_triples_fn, verblist_fn,
                                   noun_space_fn, preproc_gaps_fn,
                                   verblist_with_gaps_fn, relational_mats_out_fn,
                                   kronecker_mats_out_fn, bert_mats_out_fn,
                                   bert_in_context_mats_out_fn, bert_space_fn,
                                   kronecker_bert_mats_out_fn)
from tensorskipgram.evaluation.spaces import VectorSpace
from transformers import BertTokenizer, BertModel


def lemmatise_verb(v):
    """Fix some verbs that did not come up in the corpus."""
    if v == 'author':
        return 'write'
    elif v == 'win-over':
        return 'persuade'
    else:
        return v


def create_relational_tensors(space_fn, preproc_filename, triples_fn, verbs_fn, out_fn):
    """Create verb matrices based on their subject/object counts."""
    # noun_space = VectorSpace(name="skipgram100", path=space_fn)
    noun_space = VectorSpace(name="bert", path=space_fn)
    verbs = [ln.strip() for ln in open(verbs_fn, 'r').readlines()]
    preprocessor = Preprocessor(preproc_filename, space_fn, triples_fn, verbs_fn).preproc
    verb_counts = preprocessor['verb']['v2c']
    lower_to_upper = preprocessor['l2u']
    out_file = open(out_fn, 'w')
    print("Creating Relational verb matrices!")
    for verb in tqdm(verbs):
        verb = lemmatise_verb(verb)
        verb_mat = np.zeros((768, 768))
        for (s, o) in verb_counts[verb]:
            verb_mat += verb_counts[verb][(s, o)] * np.outer(noun_space.embed(lower_to_upper[s]),
                                                             noun_space.embed(lower_to_upper[o]))
        # verb_mat = np.sum([verb_counts[verb][(s, o)] *
        #                    np.outer(noun_space.embed(lower_to_upper[s]),
        #                             noun_space.embed(lower_to_upper[o]))
        #                    for (s, o) in verb_counts[verb]], axis=0)
        # verb_mat_txt = ' '.join([str(i) for i in verb_mat.reshape(10000)])
        verb_mat_txt = ' '.join([str(i) for i in verb_mat.reshape(589824)])
        out_file.write(f'{verb}\t{verb_mat_txt}\n')
    out_file.close()
    print("Done creating Relational verb matrices!")


def create_kronecker_tensors(space_fn, preproc_filename, triples_fn, verbs_fn, out_fn):
    # noun_space = VectorSpace(name="skipgram100", path=space_fn)
    noun_space = VectorSpace(name="bert", path=space_fn)
    verbs = [ln.strip() for ln in open(verbs_fn, 'r').readlines()]
    preprocessor = Preprocessor(preproc_filename, space_fn, triples_fn, verbs_fn).preproc
    lower_to_upper = preprocessor['l2u']
    out_file = open(out_fn, 'w')
    print("Creating Kronecker verb matrices!")
    for verb in tqdm(verbs):
        verb = lemmatise_verb(verb)
        verb_mat = np.outer(noun_space.embed(lower_to_upper[verb]), noun_space.embed(lower_to_upper[verb]))
        # verb_mat_txt = ' '.join([str(i) for i in verb_mat.reshape(10000)])
        verb_mat_txt = ' '.join([str(i) for i in verb_mat.reshape(589824)])
        out_file.write(f'{verb}\t{verb_mat_txt}\n')
    out_file.close()
    print("Done creating Kronecker verb matrices!")


class BERT():
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model


def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")
    model.eval()
    return BERT(tokenizer, model)


def get_bert_vec(bert, word):
    return torch.sum(bert.model(**bert.tokenizer(word, return_tensors="pt", add_special_tokens=False))[0][0], axis=0).detach().numpy()


def get_bert_vecs(bert: BERT, words: List[str]):
    with torch.no_grad():
        tokens = bert.tokenizer(words, return_tensors="pt", padding=True,
                                add_special_tokens=False)
        atmask = tokens['attention_mask']
        atmask = atmask.reshape(atmask.shape + (1,))
        result = bert.model(tokens['input_ids'])[0]
        result_final = torch.div(torch.sum(atmask*result, axis=1), torch.sum(atmask, axis=1))
    return dict(zip(words, result_final.detach().numpy()))


def get_bert_svo_vecs(bert, subj, verb, obj):
    subj_slice = len(bert.tokenizer(subj)['input_ids'])-2
    verb_slice = len(bert.tokenizer(verb)['input_ids'])-2
    svo_vecs = bert.model(**bert.tokenizer(f'{subj} {verb} {obj}', return_tensors="pt"))[0][0][1:-1]
    subj_vecs, verb_vecs, obj_vecs = svo_vecs[:subj_slice], svo_vecs[subj_slice:subj_slice+verb_slice], svo_vecs[subj_slice+verb_slice:]
    subj_vec, obj_vec = torch.sum(subj_vecs, axis=0).detach().numpy(), torch.sum(obj_vecs, axis=0).detach().numpy()
    return subj_vec, obj_vec


def create_relational_tensors_from_bert_in_context(space_fn, preproc_filename, triples_fn, verbs_fn, out_fn):
    """Create verb matrices based on their subject/object counts, coming from context-BERT."""
    bert = load_bert()
    verbs = [ln.strip() for ln in open(verbs_fn, 'r').readlines()]
    preprocessor = Preprocessor(preproc_filename, space_fn, triples_fn, verbs_fn).preproc
    verb_counts = preprocessor['verb']['v2c']
    out_file = open(out_fn, 'w')
    print("Creating Relational verb matrices in context with BERT!")
    with torch.no_grad():
        for verb in tqdm(verbs):
            verb = lemmatise_verb(verb)
            verb_mat = np.sum([verb_counts[verb][(s, o)] *
                               np.outer(*get_bert_svo_vecs(bert, s, verb, o))
                               for (s, o) in verb_counts[verb]], axis=0)
            verb_mat_txt = ' '.join([str(i) for i in verb_mat.reshape(589824)])
            out_file.write(f'{verb}\t{verb_mat_txt}\n')
    out_file.close()
    print("Done creating Relational verb matrices in context with BERT!")


def main() -> None:
    # create_relational_tensors(noun_space_fn, preproc_gaps_fn, svo_triples_fn, verblist_with_gaps_fn, relational_mats_out_fn)
    # create_kronecker_tensors(noun_space_fn, preproc_gaps_fn, svo_triples_fn, verblist_with_gaps_fn, kronecker_mats_out_fn)
    # create_relational_tensors(bert_space_fn, preproc_gaps_fn, svo_triples_fn, verblist_with_gaps_fn, bert_mats_out_fn)
    create_kronecker_tensors(bert_space_fn, preproc_gaps_fn, svo_triples_fn, verblist_with_gaps_fn, kronecker_bert_mats_out_fn)
    # create_relational_tensors_from_bert(noun_space_fn, preproc_gaps_fn, svo_triples_fn, verblist_with_gaps_fn, bert_mats_out_fn)
    # create_relational_tensors_from_bert_in_context(noun_space_fn, preproc_gaps_fn, svo_triples_fn, verblist_with_gaps_fn, bert_in_context_mats_out_fn)
