import torch
from torch import LongTensor, FloatTensor
from torch.nn import Embedding, Linear
from tensorskipgram.evaluation.data import SentenceData


class SentenceEmbedderSimilarity(torch.nn.Module):
    def __init__(self, noun_matrix, subj_verb_cube, obj_verb_cube, hidden_size: int):
        super(SentenceEmbedderSimilarity, self).__init__()
        self.noun_embedding = Embedding.from_pretrained(noun_matrix)
        self.embed_size = self.noun_embedding.embedding_dim
        self.hidden_size = hidden_size
        self.subj_verb_embedding = Embedding.from_pretrained(subj_verb_cube)
        self.obj_verb_embedding = Embedding.from_pretrained(obj_verb_cube)
        self.predictor_h = Linear(4*self.embed_size, self.hidden_size)
        self.predictor_o = Linear(self.hidden_size, 5)

    def forward(self,
                X_sentence1: SentenceData,
                X_sentence2: SentenceData) -> FloatTensor:
        """Forward a sentence with verbs in it.

        The input will be a pair of the leftover words in a sentence and the
        verb-args list. For the verb-args list, we apply the matrix for each
        verb to it's arguments, and add in the vectors of the leftover words.
        """
        """
        :param X_sentence1: [batch_size, num_samples]
        :param X_sentence2: [batch_size, num_samples]
        :return: similarity estimation: [batch_size, num_samples]
        """
        emb_sentence1 = self.forward_sentence(*X_sentence1)
        emb_sentence2 = self.forward_sentence(*X_sentence2)

        mult_dist = torch.mul(emb_sentence1, emb_sentence2)
        abs_dist = torch.abs(torch.add(emb_sentence1, -emb_sentence2))
        vec_dist = torch.cat((emb_sentence1, emb_sentence2, mult_dist, abs_dist))

        out = torch.nn.Sigmoid()(self.predictor_h(vec_dist))
        out = torch.nn.LogSoftmax(dim=-1)(self.predictor_o(out))
        return out

    def embed_verb_args(self, verb_args, arg=None):
        assert arg in ['subj', 'obj', 'transsubj', 'transobj']
        if len(verb_args) == 0:
            verb_mats = torch.tensor([])
            arg_vecs = torch.tensor([])
        elif arg == 'subj':
            verb_mats = self.subj_verb_embedding(verb_args[:, 0])
            arg_vecs = self.noun_embedding(verb_args[:, 1])
        elif arg == 'obj':
            verb_mats = self.obj_verb_embedding(verb_args[:, 0])
            arg_vecs = self.noun_embedding(verb_args[:, 1])
        elif arg == 'transsubj':
            verb_mats = self.subj_verb_embedding(verb_args[:, 0])
            arg_vecs = self.noun_embedding(verb_args[:, 1])
        elif arg == 'transobj':
            verb_mats = self.obj_verb_embedding(verb_args[:, 0])
            arg_vecs = self.noun_embedding(verb_args[:, 2])
        return verb_mats, arg_vecs

    def forward_sentence(self,
                         words: LongTensor,
                         verb_subj: LongTensor,
                         verb_obj: LongTensor,
                         verb_trans: LongTensor) -> FloatTensor:
        if words.dim() == 2:
            words, verb_subj, verb_obj, verb_trans = words[0], verb_subj[0], verb_obj[0], verb_trans[0]
        word_sum = self.noun_embedding(words).sum(dim=0)
        subj_verb_mats, subj_vecs = self.embed_verb_args(verb_subj, arg='subj')
        obj_verb_mats, obj_vecs = self.embed_verb_args(verb_obj, arg='obj')
        trans_verb_subj_mats, trans_subj_vecs = self.embed_verb_args(verb_trans, arg='transsubj')
        trans_verb_obj_mats, trans_obj_vecs = self.embed_verb_args(verb_trans, arg='transobj')
        verb_mats = torch.cat((subj_verb_mats, obj_verb_mats,
                               trans_verb_subj_mats, trans_verb_obj_mats))
        verb_mats = verb_mats.view(-1, self.embed_size, self.embed_size)
        arg_vecs = torch.cat((subj_vecs, obj_vecs,
                              trans_subj_vecs, trans_obj_vecs))
        arg_vecs = arg_vecs.view(-1, self.embed_size)
        verb_arg_vecs = torch.bmm(verb_mats, arg_vecs.unsqueeze(-1)).squeeze()
        verb_arg_sum = verb_arg_vecs.sum(dim=0)
        return word_sum + verb_arg_sum
