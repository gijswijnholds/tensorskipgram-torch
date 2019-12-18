import torch
from torch import FloatTensor, Tensor
from torch.nn import Embedding


class Skipgram(torch.nn.Module):
    def __init__(self, vocab_size:int, context_vocab_size:int, embed_size:int):
        super(Skipgram, self).__init__()
        self.target_embedding = Embedding(vocab_size, embed_size)
        self.context_embedding = Embedding(context_vocab_size, embed_size)

    def forward(self, X_target: Tensor, X_context: Tensor) -> FloatTensor:
        """
        :param X_target: [batch_size]
        :param X_context: [batch_size]
        :return:
        """
        target_vecs = self.target_embedding(X_target)
        context_vecs = self.context_embedding(X_context)
        batch_dot = torch.sum(target_vecs * context_vecs, dim=1)
        return batch_dot

class MatrixSkipgram(torch.nn.Module):
    def __init__(self, noun_vocab_size: int, functor_vocab_size: int,
                 context_vocab_size: int, embed_size: int,
                 nounMatrix: FloatTensor):
        super(MatrixSkipgram, self).__init__()
        assert nounMatrix.shape[0] == noun_vocab_size
        assert nounMatrix.shape[1] == embed_size
        self.embed_size = embed_size
        self.argument_embedding = Embedding.from_pretrained(nounMatrix)
        self.functor_embedding = Embedding(functor_vocab_size, embed_size*embed_size)
        self.context_embedding = Embedding(context_vocab_size, embed_size)

    def forward(self, X_argument: Tensor,
                X_functor: Tensor,
                X_context: Tensor) -> FloatTensor:
        """
        :param X_argument: [batch_size]
        :param X_functor: [batch_size]
        :param X_context: [batch_size]
        :return:
        """
        arg_vecs = self.argument_embedding(X_argument)
        func_vecs = self.functor_embedding(X_functor)
        context_vecs = self.context_embedding(X_context)
        func_mats = func_vecs.view(-1, self.embed_size, self.embed_size)

        funcarg_vecs = torch.matmul(func_mats, arg_vecs.unsqueeze(2)).squeeze()

        batch_dot = torch.sum(funcarg_vecs * context_vecs, dim=1)
        return batch_dot


loss_fn = torch.nn.BCEWithLogitsLoss()

# nounMatrix = torch.rand(50,100)
# myMatSG = MatrixSkipgram(50, 10, 20, 100, nounMatrix)
