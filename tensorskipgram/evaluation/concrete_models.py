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
                                                          CompositionModelEarly,
                                                          CompositionModelMid,
                                                          CompositionModelTwo,
                                                          IntransitiveModel,
                                                          TransitiveModel,
                                                          EllipsisModel,
                                                          IntransitiveModelEarly,
                                                          TransitiveModelEarly,
                                                          EllipsisModelEarly,
                                                          IntransitiveModelMid,
                                                          TransitiveModelMid,
                                                          EllipsisModelMid,
                                                          TransitiveModelTwo,
                                                          EllipsisModelTwo)
from tensorskipgram.config import noun_space_fn, model_path_subj_conc, model_path_obj_conc

skipgram_space = VectorSpace(name="skipgram100", path=noun_space_fn)
skipgram_subj_mats = MatrixSpace(name="skipgram_subj_mat", path=model_path_subj_conc)
skipgram_obj_mats = MatrixSpace(name="skipgram_obj_mat", path=model_path_obj_conc)
# relational_mats = MatrixSpace(name="relational_mat", path=model_path_relational)
#
# def make_concrete_model(name, model_class: CompositionModel, composer, setting):
#     return model_class(name, skipgram_space, relational_mats, composer)


def make_concrete_model_early(name, model_class: CompositionModelEarly, composer, alpha: float):
    return model_class(name, skipgram_space, skipgram_subj_mats, skipgram_obj_mats, composer, alpha)


def make_concrete_model_mid(name, model_class: CompositionModelMid, composer, alpha: float):
    return model_class(name, skipgram_space, skipgram_subj_mats, skipgram_obj_mats, composer, alpha)


def make_concrete_model_two(name, model_class: CompositionModelTwo, composer_subj, composer_obj, alpha: float):
    return model_class(name, skipgram_space, skipgram_subj_mats, skipgram_obj_mats, composer_subj, composer_obj, alpha)


alphas = [a/10. for a in range(0, 11)]

""" Intransitive Models """
cat_intrans_models_early = [make_concrete_model_early("CAT-sv", IntransitiveModelEarly, cat_intrans, alpha=a)
                            for a in alphas]
cat_intrans_models_mid = [make_concrete_model_mid("CAT-sv", IntransitiveModelMid, cat_intrans, alpha=a)
                          for a in alphas]


""" Transitive Models """

cat_subject_models_early = [make_concrete_model_early("CATS-svo", TransitiveModelEarly, cat_subject, alpha=a)
                            for a in alphas]
cat_object_models_early = [make_concrete_model_early("CATO-svo", TransitiveModelEarly, cat_subject, alpha=a)
                           for a in alphas]
copy_subject_models_early = [make_concrete_model_early("copy-subject-svo", TransitiveModelEarly, copy_subject, alpha=a)
                             for a in alphas]
copy_object_models_early = [make_concrete_model_early("copy-object-svo", TransitiveModelEarly, copy_object, alpha=a)
                            for a in alphas]
frobenius_add_models_early = [make_concrete_model_early("frobenius-add-svo", TransitiveModelEarly, frobenius_add, alpha=a)
                              for a in alphas]
frobenius_mult_models_early = [make_concrete_model_early("frobenius-mult-svo", TransitiveModelEarly, frobenius_mult, alpha=a)
                               for a in alphas]

trans_models_early = (cat_subject_models_early + cat_object_models_early + copy_subject_models_early +
                      copy_object_models_early + frobenius_add_models_early + frobenius_mult_models_early)

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

copy_argument_models = [make_concrete_model_two("copy_argument-svo", TransitiveModelTwo,
                        copy_subject, copy_object, a) for a in alphas]
copy_argument_sum_models = [make_concrete_model_two("copy-argument-sum-svo", TransitiveModelTwo,
                            copy_subject_sum, copy_object_sum, a) for a in alphas]
cat_argument_models = [make_concrete_model_two("cat-argument-svo", TransitiveModelTwo,
                       cat_subject, cat_object, a) for a in alphas]

trans_models_two = copy_argument_models + copy_argument_sum_models + cat_argument_models

TODO:
trans_models_late = [(m1, m2) for m1 in X for m2 in Y]

""" Ellipsis Models """
ell_cat_subject_sum_models_early = [make_concrete_model_early("ell-cat-subject-sum-svos", EllipsisModelEarly, ell_cat_subject_sum, alpha=a)
                                    for a in alphas]
ell_cat_subject_mult_models_early = [make_concrete_model_early("ell-cat-subject-mult-svos", EllipsisModelEarly, ell_cat_subject_mult, alpha=a)
                                     for a in alphas]

ell_cat_object_sum_models_early = [make_concrete_model_early("ell-cat-object-sum-svos", EllipsisModelEarly, ell_cat_object_sum, alpha=a)
                                   for a in alphas]
ell_cat_object_mult_models_early = [make_concrete_model_early("ell-cat-object-mult-svos", EllipsisModelEarly, ell_cat_object_mult, alpha=a)
                                    for a in alphas]


ell_models_early = (ell_cat_subject_sum_models_early + ell_cat_subject_mult_models_early +
                    ell_cat_object_sum_models_early + ell_cat_object_mult_models_early)


ell_cat_subject_sum_models_mid = [make_concrete_model_mid("ell-cat-subject-sum-svos", EllipsisModelMid, ell_cat_subject_sum, alpha=a)
                                  for a in alphas]
ell_cat_subject_mult_models_mid = [make_concrete_model_mid("ell-cat-subject-mult-svos", EllipsisModelMid, ell_cat_subject_mult, alpha=a)
                                   for a in alphas]

ell_cat_object_sum_models_mid = [make_concrete_model_mid("ell-cat-object-sum-svos", EllipsisModelMid, ell_cat_object_sum, alpha=a)
                                 for a in alphas]
ell_cat_object_mult_models_mid = [make_concrete_model_mid("ell-cat-object-mult-svos", EllipsisModelMid, ell_cat_object_mult, alpha=a)
                                  for a in alphas]


ell_models_mid = (ell_cat_subject_sum_models_mid + ell_cat_subject_mult_models_mid +
                  ell_cat_object_sum_models_mid + ell_cat_object_mult_models_mid)


# catArgModelsMiddle = [TransitiveModelTensorSeparateMiddle("Cat Argument_%s" % a, "CatArg_%s" % a, cat_subject, cat_object, a) for a in alphas]

# catArgModelsLate = [TransitiveModelTensorSeparateLate("Cat Argument_%s" % a, "CatArg_%s" % a, cat_subject, cat_object, a) for a in alphas]
