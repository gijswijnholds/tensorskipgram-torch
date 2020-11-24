from tensorskipgram.evaluation.composers import cat_intrans, cat_subject, cat_object
from tensorskipgram.evaluation.composers \
    import copy_subject, copy_object, copy_subject_sum, copy_object_sum
from tensorskipgram.evaluation.composers import frobenius_add, frobenius_mult
from tensorskipgram.evaluation.composers \
    import cat_argument, copy_argument, copy_argument_sum
from tensorskipgram.evaluation.composers \
    import ell_cat_subject_sum, ell_cat_subject_mult, ell_cat_object_sum, ell_cat_object_mult
from tensorskipgram.evaluation.spaces import VectorSpace, MatrixSpace
from tensorskipgram.evaluation.composition_models import (CompositionModel,
                                                          CompositionModelMid,
                                                          CompositionModelLate,
                                                          CompositionModelTwo,
                                                          IntransitiveModel,
                                                          TransitiveModel,
                                                          EllipsisModel,
                                                          IntransitiveModelMid,
                                                          TransitiveModelMid,
                                                          EllipsisModelMid,
                                                          IntransitiveModelLate,
                                                          TransitiveModelLate,
                                                          EllipsisModelLate,
                                                          IntransitiveModelTwo,
                                                          TransitiveModelTwo,
                                                          EllipsisModelTwo)
from tensorskipgram.config import noun_space_fn, model_path_subj_conc, model_path_obj_conc

skipgram_space = VectorSpace(name="skipgram100", path=noun_space_fn)
skipgram_subj_mats = MatrixSpace(name="skipgram_subj_mat", path=model_path_subj_conc)
skipgram_obj_mats = MatrixSpace(name="skipgram_obj_mat", path=model_path_obj_conc)


def make_concrete_model(name, model_class: CompositionModel, composer, setting):
    assert setting in ['subj', 'obj']
    if setting == 'subj':
        mats = skipgram_subj_mats
    elif setting == 'obj':
        mats = skipgram_obj_mats
    return model_class(name, skipgram_space, mats, composer)


def make_concrete_model_mid(name, model_class: CompositionModelMid, composer, alpha: float):
    return model_class(name, skipgram_space, skipgram_subj_mats, skipgram_obj_mats, composer, alpha)


def make_concrete_model_late(name, model_class: CompositionModelLate, composer, alpha: float):
    return model_class(name, skipgram_space, skipgram_subj_mats, skipgram_obj_mats, composer, alpha)


def make_concrete_model_two(name, model_class: CompositionModelTwo, composer_subj, composer_obj, alpha: float):
    return model_class(name, skipgram_space, skipgram_subj_mats, skipgram_obj_mats, composer_subj, composer_obj, alpha)


alphas = [a/10. for a in range(0, 11)]

""" Intransitive Models """
cat_intrans_model_subj = make_concrete_model("CAT-sv", IntransitiveModel, cat_intrans, setting='subj')
cat_intrans_model_obj = make_concrete_model("CAT-sv", IntransitiveModel, cat_intrans, setting='obj')

cat_intrans_models_mid = [make_concrete_model_mid("CAT-sv", IntransitiveModelMid, cat_intrans, alpha=a)
                          for a in alphas]
cat_intrans_models_late = [make_concrete_model_late("CAT-sv", IntransitiveModelLate, cat_intrans, alpha=a)
                           for a in alphas]

""" Transitive Models """

cat_subject_models_mid = [make_concrete_model_mid("CATS-svo", TransitiveModelMid, cat_subject, alpha=a)
                          for a in alphas]
cat_object_models_mid = [make_concrete_model_mid("CATO-svo", TransitiveModelMid, cat_subject, alpha=a)
                         for a in alphas]
copy_subject_models_mid = [make_concrete_model_mid("copy-subject-svo", TransitiveModelMid, copy_subject, alpha=a)
                           for a in alphas]
copy_object_models_mid = [make_concrete_model_mid("copy-object-svo", TransitiveModelMid, copy_object, alpha=a)
                          for a in alphas]
frobenius_add_models_mid = [make_concrete_model_mid("frobenius-add-svo", TransitiveModelMid, frobenius_add, alpha=a)
                            for a in alphas]
frobenius_mult_models_mid = [make_concrete_model_mid("frobenius-mult-svo", TransitiveModelMid, frobenius_mult, alpha=a)
                             for a in alphas]

trans_models_mid = (cat_subject_models_mid + cat_object_models_mid + copy_subject_models_mid +
                    copy_object_models_mid + frobenius_add_models_mid + frobenius_mult_models_mid)

cat_subject_models_late = [make_concrete_model_late("CATS-svo", TransitiveModelLate, cat_subject, alpha=a)
                           for a in alphas]
cat_object_models_late = [make_concrete_model_late("CATO-svo", TransitiveModelLate, cat_subject, alpha=a)
                          for a in alphas]
copy_subject_models_late = [make_concrete_model_late("copy-subject-svo", TransitiveModelLate, copy_subject, alpha=a)
                            for a in alphas]
copy_object_models_late = [make_concrete_model_late("copy-object-svo", TransitiveModelLate, copy_object, alpha=a)
                           for a in alphas]
frobenius_add_models_late = [make_concrete_model_late("frobenius-add-svo", TransitiveModelLate, frobenius_add, alpha=a)
                             for a in alphas]
frobenius_mult_models_late = [make_concrete_model_late("frobenius-mult-svo", TransitiveModelLate, frobenius_mult, alpha=a)
                              for a in alphas]

trans_models_late = (cat_subject_models_late + cat_object_models_late + copy_subject_models_late +
                     copy_object_models_late + frobenius_add_models_late + frobenius_mult_models_late)

copy_argument_models = [make_concrete_model_two("copy_argument-svo", TransitiveModelTwo,
                        copy_subject, copy_object, a) for a in alphas]
copy_argument_sum_models = [make_concrete_model_two("copy-argument-sum-svo", TransitiveModelTwo,
                            copy_subject_sum, copy_object_sum, a) for a in alphas]
cat_argument_models = [make_concrete_model_two("cat-argument-svo", TransitiveModelTwo,
                       cat_subject, cat_object, a) for a in alphas]

trans_models_two = copy_argument_models + copy_argument_sum_models + cat_argument_models


""" Ellipsis Models """
ell_cat_subject_sum_model = make_concrete_model_mid("ell-cat-subject-sum-svos", EllipsisModelMid, ell_cat_subject_sum, alpha=0.5)
ell_cat_subject_mult_model = make_concrete_model_mid("ell-cat-subject-mult-svos", EllipsisModelMid, ell_cat_subject_mult, alpha=0.5)

ell_cat_object_sum_model = make_concrete_model_mid("ell-cat-object-sum-svos", EllipsisModelMid, ell_cat_object_sum, alpha=0.5)
ell_cat_object_mult_model = make_concrete_model_mid("ell-cat-object-mult-svos", EllipsisModelMid, ell_cat_object_mult, alpha=0.5)


ell_models = [ell_cat_subject_sum_model, ell_cat_subject_mult_model,
              ell_cat_object_sum_model, ell_cat_object_mult_model]
