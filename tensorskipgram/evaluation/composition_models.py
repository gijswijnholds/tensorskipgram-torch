from tensorskipgram.evaluation.spaces import Vector, VectorSpace, MatrixSpace


class CompositionModel(object):
    def __init__(self, name: str, vector_space: VectorSpace, matrix_space: MatrixSpace):
        self._name = name
        self._vector_space = vector_space
        self._matrix_space = matrix_space

    @property
    def name(self):
        return self._name

    def __call__(self, sentence):
        pass


class IntransitiveModel(CompositionModel):
    def __call__(self, sentence: str) -> Vector:
        arg, verb = sentence


class TransitiveModel(CompositionModel):
    def __call__(self, sentence: str) -> Vector:
        subj, verb, obj = sentence


class EllipsisModel(CompositionModel):
    def __call__(self, sentence: str) -> Vector:
        subj, verb, obj, coord, subj, does, too = sentence
