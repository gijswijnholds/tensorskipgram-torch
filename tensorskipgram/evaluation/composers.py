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
                trans_model: Callable[[List[Union[Vector, Matrix]]], Vector],
                coordinator: Callable[[Vector, Vector], Vector]) -> Vector:
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


def ell_copy_subject_sum(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, copy_subject, add)


def ell_copy_subject_mult(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, copy_subject, mult)


def ell_copy_subject_sum_sum(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, copy_subject_sum, add)


def ell_copy_subject_sum_mult(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, copy_subject_sum, mult)


def ell_copy_object_sum(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, copy_object, add)


def ell_copy_object_mult(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, copy_object, mult)


def ell_copy_object_sum_sum(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, copy_object_sum, add)


def ell_copy_object_sum_mult(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, copy_object_sum, mult)


def ell_frobenius_add_sum(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, frobenius_add, add)


def ell_frobenius_add_mult(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, frobenius_add, mult)


def ell_frobenius_mult_sum(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, frobenius_mult, add)


def ell_frobenius_mult_mult(vects: List[Union[Vector, Matrix]]) -> Vector:
    return ell_wrapper(vects, frobenius_mult, mult)
