from typing import List, Callable, Union
from tensorskipgram.evaluation.spaces import Vector, Matrix, VectorSpace, MatrixSpace


class CompositionModel(object):
    def __init__(self, name: str, vector_space: VectorSpace, matrix_space: MatrixSpace,
                 composer: Callable[[List[Union[Vector, Matrix]]], Vector]):
        self._name = name
        self._vector_space = vector_space
        self._matrix_space = matrix_space
        self._composer = composer

    @property
    def name(self):
        return self._name

    def __call__(self, sentence):
        pass


class CompositionModelMid(object):
    def __init__(self, name: str, vector_space: VectorSpace,
                 matrix_space1: MatrixSpace, matrix_space2: MatrixSpace,
                 composer: Callable[[List[Union[Vector, Matrix]]], Vector],
                 alpha: float):
        self._name = name
        self._vector_space = vector_space
        self._matrix_space1 = matrix_space1
        self._matrix_space2 = matrix_space2
        self._composer = composer
        self._alpha = alpha

    @property
    def name(self):
        return self._name

    def __call__(self, vecs: List[Vector], mat1: Matrix, mat2: Matrix):
        pass


class CompositionModelLate(object):
    def __init__(self, name: str, vector_space: VectorSpace,
                 matrix_space1: MatrixSpace, matrix_space2: MatrixSpace,
                 composer: Callable[[List[Union[Vector, Matrix]]], Vector],
                 alpha: float):
        self._name = name
        self._vector_space = vector_space
        self._matrix_space1 = matrix_space1
        self._matrix_space2 = matrix_space2
        self._composer = composer
        self._alpha = alpha

    @property
    def name(self):
        return self._name

    def __call__(self, sentence):
        pass


class CompositionModelTwo(object):
    def __init__(self, name: str, vector_space: VectorSpace,
                 matrix_space1: MatrixSpace, matrix_space2: MatrixSpace,
                 composer1: Callable[[List[Union[Vector, Matrix]]], Vector],
                 composer2: Callable[[List[Union[Vector, Matrix]]], Vector],
                 alpha: float):
        self._name = name
        self._vector_space = vector_space
        self._matrix_space1 = matrix_space1
        self._matrix_space2 = matrix_space2
        self._composer1 = composer1
        self._composer2 = composer2
        self._alpha = alpha

    @property
    def name(self):
        return self._name

    def __call__(self, sentence):
        pass


class IntransitiveModel(CompositionModel):
    def __call__(self, sentence: str) -> Vector:
        arg, verb = sentence
        arg_vec = self._vector_space.embed(arg)
        verb_mat = self._matrix_space.embed(verb)
        return self._composer([arg_vec, verb_mat])


class TransitiveModel(CompositionModel):
    def __call__(self, sentence: str) -> Vector:
        subj, verb, obj = sentence
        subj_vec = self._vector_space.embed(subj)
        verb_mat = self._matrix_space.embed(verb)
        obj_vec = self._vector_space.embed(obj)
        return self._composer([subj_vec, verb_mat, obj_vec])


class EllipsisModel(CompositionModel):
    def __call__(self, sentence: str) -> Vector:
        subj, verb, obj, coord, subj2, does, too = sentence
        subj_vec = self._vector_space.embed(subj)
        verb_mat = self._matrix_space.embed(verb)
        obj_vec = self._vector_space.embed(obj)
        subj2_vec = self._vector_space.embed(subj2)
        self._composer([subj_vec, verb_mat, obj_vec, subj2_vec])


class IntransitiveModelMid(CompositionModelMid):
    def __call__(self, sentence: str) -> Vector:
        arg, verb = sentence
        arg_vec = self._vector_space.embed(arg)
        verb_mat1 = self._matrix_space1.embed(verb)
        verb_mat2 = self._matrix_space2.embed(verb)
        verb_mat = self._alpha * verb_mat1 + ((1 - self._alpha) * verb_mat2)
        return self._composer([arg_vec, verb_mat])


class TransitiveModelMid(CompositionModelMid):
    def __call__(self, sentence: str) -> Vector:
        subj, verb, obj = sentence
        subj_vec = self._vector_space.embed(subj)
        verb_mat1 = self._matrix_space1.embed(verb)
        verb_mat2 = self._matrix_space2.embed(verb)
        obj_vec = self._vector_space.embed(obj)
        verb_mat = self._alpha * verb_mat1 + ((1 - self._alpha) * verb_mat2)
        return self._composer([subj_vec, verb_mat, obj_vec])


class EllipsisModelMid(CompositionModelMid):
    def __call__(self, sentence: str) -> Vector:
        subj, verb, obj, coord, subj2, does, too = sentence
        subj_vec = self._vector_space.embed(subj)
        verb_mat1 = self._matrix_space1.embed(verb)
        verb_mat2 = self._matrix_space2.embed(verb)
        obj_vec = self._vector_space.embed(obj)
        subj2_vec = self._vector_space.embed(subj2)
        verb_mat = self._alpha * verb_mat1 + ((1 - self._alpha) * verb_mat2)
        return self._composer([subj_vec, verb_mat, obj_vec, subj2_vec])


class IntransitiveModelLate(CompositionModelLate):
    def __call__(self, sentence: str) -> Vector:
        arg, verb = sentence
        arg_vec = self._vector_space.embed(arg)
        verb_mat1 = self._matrix_space1.embed(verb)
        verb_mat2 = self._matrix_space2.embed(verb)
        comp1 = self._composer([arg_vec, verb_mat1])
        comp2 = self._composer([arg_vec, verb_mat2])
        return self._alpha * comp1 + ((1 - self._alpha) * comp2)


class TransitiveModelLate(CompositionModelLate):
    def __call__(self, sentence: str) -> Vector:
        subj, verb, obj = sentence
        subj_vec = self._vector_space.embed(subj)
        verb_mat1 = self._matrix_space1.embed(verb)
        verb_mat2 = self._matrix_space2.embed(verb)
        obj_vec = self._vector_space.embed(obj)
        comp1 = self._composer([subj_vec, verb_mat1, obj_vec])
        comp2 = self._composer([subj_vec, verb_mat2, obj_vec])
        return self._alpha * comp1 + ((1 - self._alpha) * comp2)


class EllipsisModelLate(CompositionModelLate):
    def __call__(self, sentence: str) -> Vector:
        subj, verb, obj, coord, subj2, does, too = sentence
        subj_vec = self._vector_space.embed(subj)
        verb_mat1 = self._matrix_space1.embed(verb)
        verb_mat2 = self._matrix_space2.embed(verb)
        obj_vec = self._vector_space.embed(obj)
        subj2_vec = self._vector_space.embed(subj2)
        comp1 = self._composer([subj_vec, verb_mat1, obj_vec, subj2_vec])
        comp2 = self._composer([subj_vec, verb_mat2, obj_vec, subj2_vec])
        return self._alpha * comp1 + ((1 - self._alpha) * comp2)


class IntransitiveModelTwo(CompositionModelTwo):
    def __call__(self, sentence: str) -> Vector:
        arg, verb = sentence
        arg_vec = self._vector_space.embed(arg)
        verb_mat1 = self._matrix_space1.embed(verb)
        verb_mat2 = self._matrix_space2.embed(verb)
        comp1 = self._composer1([arg_vec, verb_mat1])
        comp2 = self._composer2([arg_vec, verb_mat2])
        return self._alpha * comp1 + ((1 - self._alpha) * comp2)


class TransitiveModelTwo(CompositionModelTwo):
    def __call__(self, sentence: str) -> Vector:
        subj, verb, obj = sentence
        subj_vec = self._vector_space.embed(subj)
        verb_mat1 = self._matrix_space1.embed(verb)
        verb_mat2 = self._matrix_space2.embed(verb)
        obj_vec = self._vector_space.embed(obj)
        comp1 = self._composer1([subj_vec, verb_mat1, obj_vec])
        comp2 = self._composer2([subj_vec, verb_mat2, obj_vec])
        return self._alpha * comp1 + ((1 - self._alpha) * comp2)


class EllipsisModelTwo(CompositionModelTwo):
    def __call__(self, sentence: str) -> Vector:
        subj, verb, obj, coord, subj2, does, too = sentence
        subj_vec = self._vector_space.embed(subj)
        verb_mat1 = self._matrix_space1.embed(verb)
        verb_mat2 = self._matrix_space2.embed(verb)
        obj_vec = self._vector_space.embed(obj)
        subj2_vec = self._vector_space.embed(subj2)
        comp1 = self._composer1([subj_vec, verb_mat1, obj_vec, subj2_vec])
        comp2 = self._composer2([subj_vec, verb_mat2, obj_vec, subj2_vec])
        return self._alpha * comp1 + ((1 - self._alpha) * comp2)
