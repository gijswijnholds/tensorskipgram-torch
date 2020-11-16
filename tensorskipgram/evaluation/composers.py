"""Define ways of composing vectors/matrices together into a sentence embedding."""
import numpy as np
from typing import List, Union, Tuple, Callable
from tensorskipgram.evaluation.spaces import Vector, Matrix


"""Intransitive Models."""


def cat_intrans(vects: List[Union[Vector, Matrix]]) -> Vector:
    subj_vec, verb_mat = vects
    return np.einsum('i,ij->j', subj_vec, verb_mat)


"""Transitive Models."""


def copy_object(vects: List[Union[Vector, Matrix]]) -> Vector:
    subj_vec, verb_mat, obj_vec = vects
    return np.einsum('i,ij->j', subj_vec, verb_mat) * obj_vec


def copy_subject(vects: List[Union[Vector, Matrix]]) -> Vector:
    subj_vec, verb_mat, obj_vec = vects
    return subj_vec * np.einsum('i,ij->j', obj_vec, verb_mat)


def copy_object_sum(vects: List[Union[Vector, Matrix]]) -> Vector:
    subj_vec, verb_mat, obj_vec = vects
    return np.einsum('i,ij->j', subj_vec, verb_mat) + obj_vec


def copy_subject_sum(vects: List[Union[Vector, Matrix]]) -> Vector:
    subj_vec, verb_mat, obj_vec = vects
    return subj_vec + np.einsum('i,ij->j', obj_vec, verb_mat)


def cat_subject(vects: List[Union[Vector, Matrix]]) -> Vector:
    subj_vec, verb_mat = vects[0], vects[1]
    return np.einsum('i,ij->j', subj_vec, verb_mat)


def cat_object(vects: List[Union[Vector, Matrix]]) -> Vector:
    verb_mat, obj_vec = vects[1], vects[2]
    return np.einsum('i,ij->j', obj_vec, verb_mat)


def frobenius_add(vects: List[Union[Vector, Matrix]]) -> Vector:
    return copy_subject(vects) + copy_object(vects)


def frobenius_mult(vects: List[Union[Vector, Matrix]]) -> Vector:
    return copy_subject(vects) * copy_object(vects)


"""Ellipsis Models."""


def ell_wrapper(vects: List[Union[Vector, Matrix]],
                trans_model: Callable[List[Union[Vector, Matrix]], Vector],
                coordinator: Callable[Tuple[Vector, Vector], Vector]) -> Vector:
    subj_vec, verb_mat, obj_vec, subj2_vec = vects
    return coordinator(trans_model([subj_vec, verb_mat, obj_vec]),
                       trans_model([subj2_vec, verb_mat, obj_vec]))


def add(v1, v2):
    return v1+v2


def mult(v1, v2):
    return v1*v2


def ell_cat_subject_sum(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, cat_subject, add)


def ell_cat_subject_mult(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, cat_subject, mult)


def ell_cat_object_sum(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, cat_object, add)


def ell_cat_object_mult(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, cat_object, mult)


"""Two Map Transitive Models."""


def cat_argument(vects: List[Union[Vector, Matrix]], alpha: float) -> Vector:
    subj_vec, subj_mat, obj_mat, obj_vec = vects
    comp_subj = cat_subject([subj_vec, subj_mat, obj_vec])
    comp_obj = cat_object([subj_vec, obj_mat, obj_vec])
    return alpha * comp_subj + (1 - alpha) * comp_obj


def copy_argument(vects: List[Union[Vector, Matrix]], alpha: float) -> Vector:
    subj_vec, subj_mat, obj_mat, obj_vec = vects
    comp_subj = copy_object([subj_vec, subj_mat, obj_vec])
    comp_obj = copy_subject([subj_vec, obj_mat, obj_vec])
    return alpha * comp_subj + (1 - alpha) * comp_obj


def copy_argument_sum(vects: List[Union[Vector, Matrix]], alpha: float) -> Vector:
    subj_vec, subj_mat, obj_mat, obj_vec = vects
    comp_subj = copy_object_sum([subj_vec, subj_mat, obj_vec])
    comp_obj = copy_subject_sum([subj_vec, obj_mat, obj_vec])
    return alpha * comp_subj + (1 - alpha) * comp_obj
