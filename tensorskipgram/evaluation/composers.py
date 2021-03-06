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


"""Parasitic Gap Models."""
def paragaps_basic(vects: List[Union[Vector, Matrix]]) -> Vector:
    noun1_vec, verb1_mat, noun2_vec, verb2_mat = vects
    return cat_intrans(vects[:2]) + cat_intrans(vects[2:])


""" Copying meta models for ellipsis. """
def normal(vects: List[Union[Vector, Matrix]],
           trans_model: Callable[[List[Union[Vector, Matrix]]], Vector],
           coordinator: Callable[[Vector, Vector], Vector]) -> Vector:
    subj_vec, verb_mat, obj_vec, subj2_vec = vects
    return coordinator(trans_model([subj_vec, verb_mat, obj_vec]),
                       trans_model([subj2_vec, verb_mat, obj_vec]))


def cogebraa(vects: List[Union[Vector, Matrix]],
             trans_model: Callable[[List[Union[Vector, Matrix]]], Vector],
             coordinator: Callable[[Vector, Vector], Vector]) -> Vector:
    subj_vec, verb_mat, obj_vec, subj2_vec = vects
    return coordinator(trans_model([subj_vec, verb_mat, obj_vec]), subj2_vec)


def cogebrab(vects: List[Union[Vector, Matrix]],
             trans_model: Callable[[List[Union[Vector, Matrix]]], Vector],
             coordinator: Callable[[Vector, Vector], Vector]) -> Vector:
    subj_vec, verb_mat, obj_vec, subj2_vec = vects
    return coordinator(subj_vec, trans_model([subj2_vec, verb_mat, obj_vec]))


def cofree(vects: List[Union[Vector, Matrix]],
           trans_model: Callable[[List[Union[Vector, Matrix]]], Vector],
           coordinator: Callable[[Vector, Vector], Vector]) -> Vector:
    subj_vec, verb_mat, obj_vec, subj2_vec = vects
    return (coordinator(trans_model([subj_vec, verb_mat, obj_vec]), subj2_vec) +
            coordinator(subj_vec, trans_model([subj2_vec, verb_mat, obj_vec])))


trans_composers = [copy_object, copy_subject, frobenius_mult, frobenius_add]
coordinators = [mult, add]

# normal_models = [('normal-' + comp.__name__ + '-' + c.__name__, lambda vects: normal(vects, comp, c)) for comp in trans_composers for c in coordinators]
# cogebraa_models = [('cogebraa-' + comp.__name__ + '-' + c.__name__, lambda vects: cogebraa(vects, comp, c)) for comp in trans_composers for c in coordinators]
# cogebrab_models = [('cogebrab-' + comp.__name__ + '-' + c.__name__, lambda vects: cogebrab(vects, comp, c)) for comp in trans_composers for c in coordinators]
# cofree_models = [('cofree-' + comp.__name__ + '-' + c.__name__, lambda vects: cofree(vects, comp, c)) for comp in trans_composers for c in coordinators]
normal_models = [('normal-copy_object-add', lambda vects: normal(vects, copy_object, add)),
                 ('normal-copy_subject-add', lambda vects: normal(vects, copy_subject, add)),
                 ('normal-frobenius_add-add', lambda vects: normal(vects, frobenius_add, add)),
                 ('normal-frobenius_mult-add', lambda vects: normal(vects, frobenius_mult, add)),
                 ('normal-copy_object-mult', lambda vects: normal(vects, copy_object, mult)),
                 ('normal-copy_subject-mult', lambda vects: normal(vects, copy_subject, mult)),
                 ('normal-frobenius_add-mult', lambda vects: normal(vects, frobenius_add, mult)),
                 ('normal-frobenius_mult-mult', lambda vects: normal(vects, frobenius_mult, mult))]
cogebraa_models = [('cogebraa-copy_object-add', lambda vects: cogebraa(vects, copy_object, add)),
                 ('cogebraa-copy_subject-add', lambda vects: cogebraa(vects, copy_subject, add)),
                 ('cogebraa-frobenius_add-add', lambda vects: cogebraa(vects, frobenius_add, add)),
                 ('cogebraa-frobenius_mult-add', lambda vects: cogebraa(vects, frobenius_mult, add)),
                 ('cogebraa-copy_object-mult', lambda vects: cogebraa(vects, copy_object, mult)),
                 ('cogebraa-copy_subject-mult', lambda vects: cogebraa(vects, copy_subject, mult)),
                 ('cogebraa-frobenius_add-mult', lambda vects: cogebraa(vects, frobenius_add, mult)),
                 ('cogebraa-frobenius_mult-mult', lambda vects: cogebraa(vects, frobenius_mult, mult))]
cogebrab_models = [('cogebrab-copy_object-add', lambda vects: cogebrab(vects, copy_object, add)),
                 ('cogebrab-copy_subject-add', lambda vects: cogebrab(vects, copy_subject, add)),
                 ('cogebrab-frobenius_add-add', lambda vects: cogebrab(vects, frobenius_add, add)),
                 ('cogebrab-frobenius_mult-add', lambda vects: cogebrab(vects, frobenius_mult, add)),
                 ('cogebrab-copy_object-mult', lambda vects: cogebrab(vects, copy_object, mult)),
                 ('cogebrab-copy_subject-mult', lambda vects: cogebrab(vects, copy_subject, mult)),
                 ('cogebrab-frobenius_add-mult', lambda vects: cogebrab(vects, frobenius_add, mult)),
                 ('cogebrab-frobenius_mult-mult', lambda vects: cogebrab(vects, frobenius_mult, mult))]
cofree_models = [('cofree-copy_object-add', lambda vects: cofree(vects, copy_object, add)),
                 ('cofree-copy_subject-add', lambda vects: cofree(vects, copy_subject, add)),
                 ('cofree-frobenius_add-add', lambda vects: cofree(vects, frobenius_add, add)),
                 ('cofree-frobenius_mult-add', lambda vects: cofree(vects, frobenius_mult, add)),
                 ('cofree-copy_object-mult', lambda vects: cofree(vects, copy_object, mult)),
                 ('cofree-copy_subject-mult', lambda vects: cofree(vects, copy_subject, mult)),
                 ('cofree-frobenius_add-mult', lambda vects: cofree(vects, frobenius_add, mult)),
                 ('cofree-frobenius_mult-mult', lambda vects: cofree(vects, frobenius_mult, mult))]
