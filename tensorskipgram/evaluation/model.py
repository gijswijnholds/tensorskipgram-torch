import torch
from torch.nn import Embedding


class SentenceEmbedder(torch.nn.Module):
    def __init__(self, noun_matrix, subj_verb_cube, obj_verb_cube):
        super(SentenceEmbedder, self).__init__()
        self.noun_embedding = Embedding.from_pretrained(noun_matrix)
        self.subj_verb_embedding = Embedding.from_pretrained(subj_verb_cube)
        self.obj_verb_embedding = Embedding.from_pretrained(obj_verb_cube)

    def forward(self, split_sentence):
        """Forward a sentence with verbs in it.

        The input will be a pair of the leftover words in a sentence and the
        verb-args list. For the verb-args list, we apply the matrix for each
        verb to it's arguments, and add in the vectors of the leftover words.
        """
        pass
