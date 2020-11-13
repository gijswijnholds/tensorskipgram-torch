import torch
from torch import LongTensor, FloatTensor
from torch.nn import Embedding, Linear, Parameter
from tensorskipgram.evaluation.data import SentenceData
from tensorskipgram.evaluation.trainer import map_label_to_target


class SentenceEmbedderVec(torch.nn.Module):
    def __init__(self, noun_matrix, flex_nouns: bool):
        super(SentenceEmbedderVec, self).__init__()
        self.noun_embedding = Embedding.from_pretrained(noun_matrix)
        self.embed_size = self.noun_embedding.embedding_dim
        if flex_nouns:
            self.noun_embedding.weight.requires_grad = True

    def forward(self,
                words: LongTensor) -> FloatTensor:
        if isinstance(words, list):
            words = words[0]
        # word_sum = self.noun_embedding(words).mean(dim=0)
        word_sum = torch.prod(self.noun_embedding(words), dim=0)
        # word_sum = self.noun_embedding(all_words).mean(dim=0)
        return word_sum


class SentenceEmbedder(torch.nn.Module):
    def __init__(self, noun_matrix, subj_verb_cube, obj_verb_cube,
                 flex_nouns: bool, flex_verbs: bool):
        super(SentenceEmbedder, self).__init__()
        self.noun_embedding = Embedding.from_pretrained(noun_matrix)
        self.embed_size = self.noun_embedding.embedding_dim
        self.subj_verb_embedding = Embedding.from_pretrained(subj_verb_cube)
        self.obj_verb_embedding = Embedding.from_pretrained(obj_verb_cube)
        self.modulator = Parameter(torch.Tensor(1))
        self.modulator.data.uniform_(-1, 1)
        if flex_nouns:
            self.noun_embedding.weight.requires_grad = True
        if flex_verbs:
            self.subj_verb_embedding.weight.requires_grad = True
            self.obj_verb_embedding.weight.requires_grad = True

    def embed_verb_args(self, verb_args: LongTensor, arg=None):
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

    def forward(self,
                words: LongTensor,
                verb_subj: LongTensor,
                verb_obj: LongTensor,
                verb_trans: LongTensor) -> FloatTensor:
        if words.dim() == 2:
            words, verb_subj, verb_obj, verb_trans = words[0], verb_subj[0], verb_obj[0], verb_trans[0]
        word_sum = self.noun_embedding(words).mean(dim=0)
        # return word_sum
        subj_verb_mats, subj_vecs = self.embed_verb_args(verb_subj, arg='subj')
        obj_verb_mats, obj_vecs = self.embed_verb_args(verb_obj, arg='obj')
        trans_verb_subj_mats, trans_subj_vecs = self.embed_verb_args(
            verb_trans, arg='transsubj')
        trans_verb_obj_mats, trans_obj_vecs = self.embed_verb_args(
            verb_trans, arg='transobj')
        verb_mats = torch.cat((subj_verb_mats, obj_verb_mats,
                               trans_verb_subj_mats, trans_verb_obj_mats))
        verb_mats = verb_mats.view(-1, self.embed_size, self.embed_size)
        arg_vecs = torch.cat((subj_vecs, obj_vecs,
                              trans_subj_vecs, trans_obj_vecs))
        arg_vecs = arg_vecs.view(-1, self.embed_size)
        verb_arg_vecs = torch.bmm(verb_mats, arg_vecs.unsqueeze(-1)).squeeze()
        verb_arg_vecs = torch.nn.Tanh()(verb_arg_vecs)
        if verb_arg_vecs.dim() == 1:
            verb_arg_vecs = verb_arg_vecs.unsqueeze(0)
        if len(verb_arg_vecs) == 0:
            output = word_sum
        else:
            output = verb_arg_sum
            # verb_arg_sum = verb_arg_vecs.mean(dim=0)
            # word_sum = torch.nn.Sigmoid()(self.modulator) * word_sum
            # output = torch.stack([word_sum, verb_arg_sum]).mean(dim=0)
        return output



class SimilarityLinear(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int):
        super(SimilarityLinear, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.activation = torch.nn.Sigmoid()
        self.predictor_h = Linear(4 * self.embed_size, self.hidden_size)
        # self.predictor_h2 = Linear(self.hidden_size, self.hidden_size)
        self.predictor_o = Linear(self.hidden_size, 5)

    def forward(self, sent1vec, sent2vec) -> FloatTensor:
        mult_dist = torch.mul(sent1vec, sent2vec)
        abs_dist = torch.abs(torch.add(sent1vec, -sent2vec))
        vec_dist = torch.cat((sent1vec, sent2vec, mult_dist, abs_dist))
        out = self.activation(self.predictor_h(vec_dist))
        # out = self.activation(self.predictor_h2(out))
        # out = torch.nn.Sigmoid()(self.predictor_h(vec_dist))
        out = torch.nn.LogSoftmax(dim=-1)(self.predictor_o(out))
        return out


class SimilarityEntailment(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int):
        super(SimilarityEntailment, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.activation = torch.nn.Tanh()
        # self.activation = torch.nn.ReLU() worked ok, higher train acc.
        self.predictor_h = Linear(4 * self.embed_size, self.hidden_size)
        # self.predictor_h2 = Linear(self.hidden_size, self.hidden_size)
        self.predictor_o = Linear(self.hidden_size, 3)

    def forward(self, sent1vec, sent2vec) -> FloatTensor:
        mult_dist = torch.mul(sent1vec, sent2vec)
        abs_dist = torch.abs(torch.add(sent1vec, -sent2vec))
        vec_dist = torch.cat((sent1vec, sent2vec, mult_dist, abs_dist))
        out = self.activation(self.predictor_h(vec_dist))
        # out = self.activation(self.predictor_h2(out))
        # out = torch.nn.Sigmoid()(self.predictor_h(vec_dist))
        out = self.predictor_o(out)
        return out


class SimilarityLinear2(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int):
        super(SimilarityLinear2, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.activation = torch.nn.Tanh()
        self.predictor_h = Linear(4 * self.embed_size, self.hidden_size)
        self.predictor_o = Linear(self.hidden_size, 1)

    def forward(self, sent1vec, sent2vec) -> FloatTensor:
        mult_dist = torch.mul(sent1vec, sent2vec)
        abs_dist = torch.abs(torch.add(sent1vec, -sent2vec))
        vec_dist = torch.cat((sent1vec, sent2vec, mult_dist, abs_dist))
        out = self.activation(self.predictor_h(vec_dist))
        # out = torch.nn.Sigmoid()(self.predictor_h(vec_dist))
        out = torch.nn.Sigmoid()(self.predictor_o(out))
        return out


class SimilarityDot(torch.nn.Module):
    def __init__(self):
        super(SimilarityDot, self).__init__()

    def forward(self, sent1vec, sent2vec) -> FloatTensor:
        dot = torch.sum(sent1vec * sent2vec)
        output = dot / (torch.norm(sent1vec) * torch.norm(sent2vec))
        # output = (((output*0.5)+0.5)*4)+1
        return output
        # output = min(torch.tensor([5.]), (((output*0.5)+0.5)*4)+1)
        # return torch.log(map_label_to_target(output, 5)[0])


class VecSimDot(torch.nn.Module):
    def __init__(self, noun_matrix, hidden_size: int, flex_nouns: bool):
        super(VecSimDot, self).__init__()
        self.sentence_embedder = SentenceEmbedderVec(noun_matrix, flex_nouns)
        self.hidden_size = hidden_size
        self.similarity = SimilarityDot()

    def forward(self,
                X_sentence1: SentenceData,
                X_sentence2: SentenceData) -> FloatTensor:
        emb_sentence1 = self.sentence_embedder(X_sentence1)
        emb_sentence2 = self.sentence_embedder(X_sentence2)
        output = self.similarity(emb_sentence1, emb_sentence2)
        return output


class VecSimLinear(torch.nn.Module):
    def __init__(self, noun_matrix, hidden_size: int, flex_nouns: bool):
        super(VecSimLinear, self).__init__()
        self.sentence_embedder = SentenceEmbedderVec(noun_matrix, flex_nouns)
        self.embed_size = self.sentence_embedder.noun_embedding.embedding_dim
        self.hidden_size = hidden_size
        self.similarity = SimilarityLinear(self.embed_size, self.hidden_size)

    def forward(self,
                X_sentence1: SentenceData,
                X_sentence2: SentenceData) -> FloatTensor:
        emb_sentence1 = self.sentence_embedder(X_sentence1)
        emb_sentence2 = self.sentence_embedder(X_sentence2)
        output = self.similarity(emb_sentence1, emb_sentence2)
        return output


class VecSimLinear2(torch.nn.Module):
    def __init__(self, noun_matrix, hidden_size: int, flex_nouns: bool):
        super(VecSimLinear2, self).__init__()
        self.sentence_embedder = SentenceEmbedderVec(noun_matrix, flex_nouns)
        self.embed_size = self.sentence_embedder.noun_embedding.embedding_dim
        self.hidden_size = hidden_size
        self.similarity = SimilarityLinear2(self.embed_size, self.hidden_size)

    def forward(self,
                X_sentence1: SentenceData,
                X_sentence2: SentenceData) -> FloatTensor:
        emb_sentence1 = self.sentence_embedder(X_sentence1)
        emb_sentence2 = self.sentence_embedder(X_sentence2)
        output = self.similarity(emb_sentence1, emb_sentence2)
        return output


class VecSimEntailment(torch.nn.Module):
    def __init__(self, noun_matrix, hidden_size: int, flex_nouns: bool):
        super(VecSimEntailment, self).__init__()
        self.sentence_embedder = SentenceEmbedderVec(noun_matrix, flex_nouns)
        self.embed_size = self.sentence_embedder.noun_embedding.embedding_dim
        self.hidden_size = hidden_size
        self.similarity = SimilarityEntailment(self.embed_size, self.hidden_size)

    def forward(self,
                X_sentence1: SentenceData,
                X_sentence2: SentenceData) -> FloatTensor:
        emb_sentence1 = self.sentence_embedder(X_sentence1)
        emb_sentence2 = self.sentence_embedder(X_sentence2)
        output = self.similarity(emb_sentence1, emb_sentence2)
        return output


class SentenceEmbedderSimilarity(torch.nn.Module):
    def __init__(self, noun_matrix, subj_verb_cube, obj_verb_cube,
                 hidden_size: int, flex_nouns: bool, flex_verbs: bool):
        super(SentenceEmbedderSimilarity, self).__init__()
        self.sentence_embedder = SentenceEmbedder(noun_matrix, subj_verb_cube,
                                                  obj_verb_cube, flex_nouns,
                                                  flex_verbs)
        self.embed_size = self.sentence_embedder.noun_embedding.embedding_dim
        self.hidden_size = hidden_size
        self.similarity = SimilarityLinear(self.embed_size, self.hidden_size)

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
        emb_sentence1 = self.sentence_embedder(*X_sentence1)
        emb_sentence2 = self.sentence_embedder(*X_sentence2)
        output = self.similarity(emb_sentence1, emb_sentence2)
        return output


class SentenceEmbedderSimilarity2(torch.nn.Module):
    def __init__(self, noun_matrix, subj_verb_cube, obj_verb_cube,
                 hidden_size: int, flex_nouns: bool, flex_verbs: bool):
        super(SentenceEmbedderSimilarity2, self).__init__()
        self.sentence_embedder = SentenceEmbedder(noun_matrix, subj_verb_cube,
                                                  obj_verb_cube, flex_nouns,
                                                  flex_verbs)
        self.embed_size = self.sentence_embedder.noun_embedding.embedding_dim
        self.hidden_size = hidden_size
        self.similarity = SimilarityLinear2(self.embed_size, self.hidden_size)

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
        emb_sentence1 = self.sentence_embedder(*X_sentence1)
        emb_sentence2 = self.sentence_embedder(*X_sentence2)
        output = self.similarity(emb_sentence1, emb_sentence2)
        return output



class SentenceEmbedderSimilarityDot(torch.nn.Module):
    def __init__(self, noun_matrix, subj_verb_cube, obj_verb_cube,
                 hidden_size: int, flex_nouns: bool, flex_verbs: bool):
        super(SentenceEmbedderSimilarityDot, self).__init__()
        self.sentence_embedder = SentenceEmbedder(noun_matrix, subj_verb_cube,
                                                  obj_verb_cube, flex_nouns,
                                                  flex_verbs)
        self.embed_size = self.sentence_embedder.noun_embedding.embedding_dim
        self.hidden_size = hidden_size
        self.similarity = SimilarityDot()

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
        emb_sentence1 = self.sentence_embedder(*X_sentence1)
        emb_sentence2 = self.sentence_embedder(*X_sentence2)
        output = self.similarity(emb_sentence1, emb_sentence2).unsqueeze(0)
        return output


class SentenceEmbedderEntailment(torch.nn.Module):
    def __init__(self, noun_matrix, subj_verb_cube, obj_verb_cube,
                 hidden_size: int, flex_nouns: bool, flex_verbs: bool):
        super(SentenceEmbedderEntailment, self).__init__()
        self.sentence_embedder = SentenceEmbedder(noun_matrix, subj_verb_cube,
                                                  obj_verb_cube, flex_nouns,
                                                  flex_verbs)
        self.embed_size = self.sentence_embedder.noun_embedding.embedding_dim
        self.hidden_size = hidden_size
        self.similarity = SimilarityEntailment(self.embed_size, self.hidden_size)

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
        emb_sentence1 = self.sentence_embedder(*X_sentence1)
        emb_sentence2 = self.sentence_embedder(*X_sentence2)
        output = self.similarity(emb_sentence1, emb_sentence2)
        return output
