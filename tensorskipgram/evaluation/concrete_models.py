from tensorskipgram.evaluation.composers import cat_intrans, cat_subject, cat_object
from tensorskipgram.evaluation.composers \
    import copy_subject, copy_object, copy_subject_sum, copy_object_sum
from tensorskipgram.evaluation.composers import frobenius_add, frobenius_mult
from tensorskipgram.evaluation.composers \
    import (ell_cat_subject_sum, ell_cat_subject_mult, ell_cat_object_sum,
            ell_cat_object_mult, ell_copy_subject_sum, ell_copy_subject_mult,
            ell_copy_object_sum, ell_copy_object_mult, ell_frobenius_add_sum,
            ell_frobenius_add_mult, ell_frobenius_mult_sum, ell_frobenius_mult_mult,
            ell_copy_subject_sum_sum, ell_copy_object_sum_sum,
            ell_copy_subject_sum_mult, ell_copy_object_sum_mult)
from tensorskipgram.evaluation.spaces import VectorSpace, MatrixSpace
from tensorskipgram.evaluation.composition_models \
    import (CompositionModel, CompositionModelEarly, CompositionModelMid,
            CompositionModelTwo, IntransitiveModel, TransitiveModel, EllipsisModel,
            IntransitiveModelEarly, TransitiveModelEarly, EllipsisModelEarly,
            IntransitiveModelMid, TransitiveModelMid, EllipsisModelMid,
            TransitiveModelTwo, EllipsisModelTwo)
from tensorskipgram.config import noun_space_fn, model_path_subj_conc, model_path_obj_conc
from tensorskipgram.config import model_out_path_subj_gaps, model_out_path_obj_gaps
from tensorskipgram.config import model_out_path_subj_gaps2, model_out_path_obj_gaps2
from tensorskipgram.evaluation.composition_models import ParagapsModel
from tensorskipgram.evaluation.composers import paragaps_basic
from tensorskipgram.config import model_out_path_subj_gapss, model_out_path_obj_gapss
skipgram_space = VectorSpace(name="skipgram100", path=noun_space_fn)
# skipgram_subj_mats = MatrixSpace(name="skipgram_subj_mat", path=model_path_subj_conc)
# skipgram_obj_mats = MatrixSpace(name="skipgram_obj_mat", path=model_path_obj_conc)
# skipgram_subj_mats = MatrixSpace(name="skipgram_subj_mat", path=model_out_path_subj_gaps2)
# skipgram_obj_mats = MatrixSpace(name="skipgram_obj_mat", path=model_out_path_obj_gaps2)

def make_concrete_model(name, model_class: CompositionModel, composer, setting: str):
    assert setting in ['subj', 'obj']
    if setting == 'subj':
        return model_class(name, skipgram_space, skipgram_subj_mats, composer)
    elif setting == 'obj':
        return model_class(name, skipgram_space, skipgram_obj_mats, composer)


def make_concrete_model_early(name, model_class: CompositionModelEarly, composer, alpha: float):
    return model_class(name, skipgram_space, skipgram_subj_mats, skipgram_obj_mats, composer, alpha)


def make_concrete_model_mid(name, model_class: CompositionModelMid, composer, alpha: float):
    return model_class(name, skipgram_space, skipgram_subj_mats, skipgram_obj_mats, composer, alpha)


def make_concrete_model_two(name, model_class: CompositionModelTwo, composer_subj, composer_obj, alpha: float):
    return model_class(name, skipgram_space, skipgram_subj_mats, skipgram_obj_mats, composer_subj, composer_obj, alpha)


alphas = [a/10. for a in range(0, 11)]

""" Intransitive Models """
cat_intrans_model_subj = make_concrete_model("CAT-sv", IntransitiveModel, cat_intrans, setting='subj')
cat_intrans_model_obj = make_concrete_model("CAT-sv", IntransitiveModel, cat_intrans, setting='obj')

cat_intrans_models_early = [make_concrete_model_early("CAT-sv", IntransitiveModelEarly, cat_intrans, alpha=a)
                            for a in alphas]
cat_intrans_models_mid = [make_concrete_model_mid("CAT-sv", IntransitiveModelMid, cat_intrans, alpha=a)
                          for a in alphas]

intrans_models_late = [('CAT-sv', cat_intrans_model_subj, cat_intrans_model_obj)]


""" Transitive Models """

cat_subject_models_early = [make_concrete_model_early("CATS-svo", TransitiveModelEarly, cat_subject, alpha=a)
                            for a in alphas]
cat_object_models_early = [make_concrete_model_early("CATO-svo", TransitiveModelEarly, cat_object, alpha=a)
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
cat_object_models_mid = [make_concrete_model_mid("CATO-svo", TransitiveModelMid, cat_object, alpha=a)
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

copy_argument_models = [make_concrete_model_two("copy-argument-svo", TransitiveModelTwo,
                        copy_object, copy_subject, a) for a in alphas]
copy_argument_sum_models = [make_concrete_model_two("copy-argument-sum-svo", TransitiveModelTwo,
                            copy_object_sum, copy_subject_sum, a) for a in alphas]
copy_argument_inv_models = [make_concrete_model_two("copy-argument-inv-svo", TransitiveModelTwo,
                                copy_subject, copy_object, a) for a in alphas]
copy_argument_inv_sum_models = [make_concrete_model_two("copy-argument-inv-sum-svo", TransitiveModelTwo,
                                    copy_subject_sum, copy_object_sum, a) for a in alphas]
cat_argument_models = [make_concrete_model_two("cat-argument-svo", TransitiveModelTwo,
                       cat_subject, cat_object, a) for a in alphas]
cat_argument_inv_models = [make_concrete_model_two("cat-argument-inv-svo", TransitiveModelTwo,
                           cat_object, cat_subject, a) for a in alphas]

trans_models_two = (copy_argument_models + copy_argument_sum_models + cat_argument_models +
                    cat_argument_inv_models + copy_argument_inv_models +
                    copy_argument_inv_sum_models)


""" Transitive Late Models. """

cat_subject_subj = make_concrete_model("CATS-svo", TransitiveModel, cat_subject, setting='subj')
cat_subject_obj = make_concrete_model("CATS-svo", TransitiveModel, cat_subject, setting='obj')

cat_object_subj = make_concrete_model("CATO-svo", TransitiveModel, cat_object, setting='subj')
cat_object_obj = make_concrete_model("CATO-svo", TransitiveModel, cat_object, setting='obj')

copy_subject_subj = make_concrete_model("copy-subject-svo", TransitiveModel, copy_subject, setting='subj')
copy_subject_obj = make_concrete_model("copy-subject-svo", TransitiveModel, copy_subject, setting='obj')
copy_subject_sum_obj = make_concrete_model("copy-subject-sum-svo", TransitiveModel, copy_subject_sum, setting='subj')

copy_object_subj = make_concrete_model("copy-object-svo", TransitiveModel, copy_object, setting='subj')
copy_object_obj = make_concrete_model("copy-object-svo", TransitiveModel, copy_object, setting='obj')
copy_object_sum_subj = make_concrete_model("copy-object-sum-svo", TransitiveModel, copy_object_sum, setting='subj')

frobenius_add_subj = make_concrete_model("frobenius-add-svo", TransitiveModel, frobenius_add, setting='subj')
frobenius_add_obj = make_concrete_model("frobenius-add-svo", TransitiveModel, frobenius_add, setting='obj')

frobenius_mult_subj = make_concrete_model("frobenius-mult-svo", TransitiveModel, frobenius_mult, setting='subj')
frobenius_mult_obj = make_concrete_model("frobenius-mult-svo", TransitiveModel, frobenius_mult, setting='obj')

trans_models_late = [('cat-subject-svo', cat_subject_subj, cat_subject_obj),
                     ('cat-object-svo', cat_object_subj, cat_object_obj),
                     ('copy-subject-svo', copy_subject_subj, copy_subject_obj),
                     ('copy-object-svo', copy_object_subj, copy_object_obj),
                     ('cat-argument-svo', cat_subject_subj, cat_object_obj),
                     ('copy-argument-svo', copy_object_subj, copy_subject_obj),
                     ('copy-argument-sum-svo', copy_object_sum_subj, copy_subject_sum_obj),
                     ('frobenius-add-svo', frobenius_add_subj, frobenius_add_obj),
                     ('frobenius-mult-svo', frobenius_mult_subj, frobenius_mult_obj)]

""" Ellipsis Models """
ell_cat_subject_sum_models_early = [make_concrete_model_early("ell-cat-subject-sum-svos", EllipsisModelEarly, ell_cat_subject_sum, alpha=a)
                                    for a in alphas]
ell_cat_subject_mult_models_early = [make_concrete_model_early("ell-cat-subject-mult-svos", EllipsisModelEarly, ell_cat_subject_mult, alpha=a)
                                     for a in alphas]

ell_cat_object_sum_models_early = [make_concrete_model_early("ell-cat-object-sum-svos", EllipsisModelEarly, ell_cat_object_sum, alpha=a)
                                   for a in alphas]
ell_cat_object_mult_models_early = [make_concrete_model_early("ell-cat-object-mult-svos", EllipsisModelEarly, ell_cat_object_mult, alpha=a)
                                    for a in alphas]


ell_copy_subject_sum_models_early = [make_concrete_model_early("ell-copy-subject-sum-svos", EllipsisModelEarly, ell_copy_subject_sum, alpha=a)
                                    for a in alphas]
ell_copy_subject_mult_models_early = [make_concrete_model_early("ell-copy-subject-mult-svos", EllipsisModelEarly, ell_copy_subject_mult, alpha=a)
                                     for a in alphas]

ell_copy_object_sum_models_early = [make_concrete_model_early("ell-copy-object-sum-svos", EllipsisModelEarly, ell_copy_object_sum, alpha=a)
                                   for a in alphas]
ell_copy_object_mult_models_early = [make_concrete_model_early("ell-copy-object-mult-svos", EllipsisModelEarly, ell_copy_object_mult, alpha=a)
                                    for a in alphas]

ell_frobenius_add_sum_models_early = [make_concrete_model_early("ell-frobenius-add-sum-svos", EllipsisModelEarly, ell_frobenius_add_sum, alpha=a)
                                   for a in alphas]
ell_frobenius_add_mult_models_early = [make_concrete_model_early("ell-frobenius-add-mult-svos", EllipsisModelEarly, ell_frobenius_add_mult, alpha=a)
                                    for a in alphas]

ell_frobenius_mult_sum_models_early = [make_concrete_model_early("ell-frobenius-mult-sum-svos", EllipsisModelEarly, ell_frobenius_mult_sum, alpha=a)
                                       for a in alphas]
ell_frobenius_mult_mult_models_early = [make_concrete_model_early("ell-frobenius-mult-mult-svos", EllipsisModelEarly, ell_frobenius_mult_mult, alpha=a)
                                        for a in alphas]


ell_models_early = (ell_cat_subject_sum_models_early + ell_cat_subject_mult_models_early +
                    ell_cat_object_sum_models_early + ell_cat_object_mult_models_early +
                    ell_copy_subject_sum_models_early + ell_copy_subject_mult_models_early +
                    ell_copy_object_sum_models_early + ell_copy_object_mult_models_early +
                    ell_frobenius_add_sum_models_early + ell_frobenius_add_mult_models_early +
                    ell_frobenius_mult_sum_models_early + ell_frobenius_mult_mult_models_early)


ell_cat_subject_sum_models_mid = [make_concrete_model_mid("ell-cat-subject-sum-svos", EllipsisModelMid, ell_cat_subject_sum, alpha=a)
                                  for a in alphas]
ell_cat_subject_mult_models_mid = [make_concrete_model_mid("ell-cat-subject-mult-svos", EllipsisModelMid, ell_cat_subject_mult, alpha=a)
                                   for a in alphas]

ell_cat_object_sum_models_mid = [make_concrete_model_mid("ell-cat-object-sum-svos", EllipsisModelMid, ell_cat_object_sum, alpha=a)
                                 for a in alphas]
ell_cat_object_mult_models_mid = [make_concrete_model_mid("ell-cat-object-mult-svos", EllipsisModelMid, ell_cat_object_mult, alpha=a)
                                  for a in alphas]


ell_copy_subject_sum_models_mid = [make_concrete_model_mid("ell-copy-subject-sum-svos", EllipsisModelMid, ell_copy_subject_sum, alpha=a)
                                   for a in alphas]
ell_copy_subject_mult_models_mid = [make_concrete_model_mid("ell-copy-subject-mult-svos", EllipsisModelMid, ell_copy_subject_mult, alpha=a)
                                    for a in alphas]

ell_copy_object_sum_models_mid = [make_concrete_model_mid("ell-copy-object-sum-svos", EllipsisModelMid, ell_copy_object_sum, alpha=a)
                                  for a in alphas]
ell_copy_object_mult_models_mid = [make_concrete_model_mid("ell-copy-object-mult-svos", EllipsisModelMid, ell_copy_object_mult, alpha=a)
                                   for a in alphas]


ell_frobenius_add_sum_models_mid = [make_concrete_model_mid("ell-frobenius-add-sum-svos", EllipsisModelMid, ell_frobenius_add_sum, alpha=a)
                                    for a in alphas]
ell_frobenius_add_mult_models_mid = [make_concrete_model_mid("ell-frobenius-add-mult-svos", EllipsisModelMid, ell_frobenius_add_mult, alpha=a)
                                     for a in alphas]

ell_frobenius_mult_sum_models_mid = [make_concrete_model_mid("ell-frobenius-mult-sum-svos", EllipsisModelMid, ell_frobenius_mult_sum, alpha=a)
                                     for a in alphas]
ell_frobenius_mult_mult_models_mid = [make_concrete_model_mid("ell-frobenius-mult-mult-svos", EllipsisModelMid, ell_frobenius_mult_mult, alpha=a)
                                      for a in alphas]


ell_models_mid = (ell_cat_subject_sum_models_mid + ell_cat_subject_mult_models_mid +
                  ell_cat_object_sum_models_mid + ell_cat_object_mult_models_mid +
                  ell_copy_subject_sum_models_mid + ell_copy_subject_mult_models_mid +
                  ell_copy_object_sum_models_mid + ell_copy_object_mult_models_mid +
                  ell_frobenius_add_sum_models_mid + ell_frobenius_add_mult_models_mid +
                  ell_frobenius_mult_sum_models_mid + ell_frobenius_mult_mult_models_mid)


""" Ellipsis Two Models. """

ell_copy_argument_sum_models = [make_concrete_model_two("copy-argument-sum-svos", EllipsisModelTwo,
                                ell_copy_object_sum, ell_copy_subject_sum, a) for a in alphas]
ell_copy_argument_mult_models = [make_concrete_model_two("copy-argument-mult-svos", EllipsisModelTwo,
                                 ell_copy_object_mult, ell_copy_subject_mult, a) for a in alphas]

ell_copy_argument_sum_sum_models = [make_concrete_model_two("copy-argument-sum-sum-svos", EllipsisModelTwo,
                                    ell_copy_object_sum_sum, ell_copy_subject_sum_sum, a) for a in alphas]
ell_copy_argument_sum_mult_models = [make_concrete_model_two("copy-argument-sum-mult-svos", EllipsisModelTwo,
                                     ell_copy_object_sum_mult, ell_copy_subject_sum_mult, a) for a in alphas]


ell_copy_argument_inv_sum_models = [make_concrete_model_two("copy-argument-inv-sum-svos", EllipsisModelTwo,
                                    ell_copy_subject_sum, ell_copy_object_sum, a) for a in alphas]
ell_copy_argument_inv_mult_models = [make_concrete_model_two("copy-argument-inv-mult-svos", EllipsisModelTwo,
                                     ell_copy_subject_mult, ell_copy_object_mult, a) for a in alphas]

ell_copy_argument_inv_sum_sum_models = [make_concrete_model_two("copy-argument-inv-sum-sum-svos", EllipsisModelTwo,
                                        ell_copy_subject_sum_sum, ell_copy_object_sum_sum, a) for a in alphas]
ell_copy_argument_inv_sum_mult_models = [make_concrete_model_two("copy-argument-inv-sum-mult-svos", EllipsisModelTwo,
                                         ell_copy_object_mult, ell_copy_subject_mult, a) for a in alphas]


ell_cat_argument_sum_models = [make_concrete_model_two("cat-argument-sum-svos", EllipsisModelTwo,
                               ell_cat_subject_sum, ell_cat_object_sum, a) for a in alphas]
ell_cat_argument_mult_models = [make_concrete_model_two("cat-argument-mult-svos", EllipsisModelTwo,
                                ell_cat_subject_mult, ell_cat_object_mult, a) for a in alphas]

ell_cat_argument_inv_sum_models = [make_concrete_model_two("cat-argument-inv-sum-svos", EllipsisModelTwo,
                                   ell_cat_object_sum, ell_cat_subject_sum, a) for a in alphas]
ell_cat_argument_inv_mult_models = [make_concrete_model_two("cat-argument-inv-mult-svos", EllipsisModelTwo,
                                    ell_cat_object_mult, ell_cat_subject_mult, a) for a in alphas]


ell_models_two = (ell_copy_argument_sum_models + ell_copy_argument_mult_models +
                  ell_copy_argument_sum_sum_models + ell_copy_argument_sum_mult_models +
                  ell_copy_argument_inv_sum_models + ell_copy_argument_inv_mult_models +
                  ell_copy_argument_inv_sum_sum_models + ell_copy_argument_inv_sum_mult_models +
                  ell_cat_argument_sum_models + ell_cat_argument_mult_models +
                  ell_cat_argument_inv_sum_models + ell_cat_argument_inv_mult_models)

""" Ellipsis Late Models. """

ell_cat_subject_sum_subj = make_concrete_model("CATS-sum-svos", EllipsisModel, ell_cat_subject_sum, setting='subj')
ell_cat_subject_sum_obj = make_concrete_model("CATS-sum-svos", EllipsisModel, ell_cat_subject_sum, setting='obj')

ell_cat_object_sum_subj = make_concrete_model("CATO-sum-svos", EllipsisModel, ell_cat_object_sum, setting='subj')
ell_cat_object_sum_obj = make_concrete_model("CATO-sum-svos", EllipsisModel, ell_cat_object_sum, setting='obj')

ell_copy_subject_sum_subj = make_concrete_model("copy-subject-sum-svos", EllipsisModel, ell_copy_subject_sum, setting='subj')
ell_copy_subject_sum_obj = make_concrete_model("copy-subject-sum-svos", EllipsisModel, ell_copy_subject_sum, setting='obj')

ell_copy_object_sum_subj = make_concrete_model("copy-object-sum-svos", EllipsisModel, ell_copy_object_sum, setting='subj')
ell_copy_object_sum_obj = make_concrete_model("copy-object-sum-svos", EllipsisModel, ell_copy_object_sum, setting='obj')


ell_copy_subject_sum_sum_obj = make_concrete_model("copy-subject-sum-sum-svos", EllipsisModel, ell_copy_subject_sum_sum, setting='obj')
ell_copy_object_sum_sum_subj = make_concrete_model("copy-subject-sum-sum-svos", EllipsisModel, ell_copy_object_sum_sum, setting='subj')


ell_frobenius_add_sum_subj = make_concrete_model("frobenius-add-sum-svos", EllipsisModel, ell_frobenius_add_sum, setting='subj')
ell_frobenius_add_sum_obj = make_concrete_model("frobenius-add-sum-svos", EllipsisModel, ell_frobenius_add_sum, setting='obj')

ell_frobenius_mult_sum_subj = make_concrete_model("frobenius-mult-sum-svos", EllipsisModel, ell_frobenius_mult_sum, setting='subj')
ell_frobenius_mult_sum_obj = make_concrete_model("frobenius-mult-sum-svos", EllipsisModel, ell_frobenius_mult_sum, setting='obj')


ell_models_late_sum = [('cat-subject-sum-svos', ell_cat_subject_sum_subj, ell_cat_subject_sum_obj),
                       ('cat-object-sum-svos', ell_cat_object_sum_subj, ell_cat_object_sum_obj),
                       ('copy-subject-sum-svos', ell_copy_subject_sum_subj, ell_copy_subject_sum_obj),
                       ('copy-object-sum-svos', ell_copy_object_sum_subj, ell_copy_object_sum_obj),
                       ('cat-argument-sum-svos', ell_cat_subject_sum_subj, ell_cat_object_sum_obj),
                       ('copy-argument-sum-svos', ell_copy_object_sum_subj, ell_copy_subject_sum_obj),
                       ('copy-argument-sum-sum-svos', ell_copy_object_sum_sum_subj, ell_copy_subject_sum_sum_obj),
                       ('frobenius-add-sum-svos', ell_frobenius_add_sum_subj, ell_frobenius_add_sum_obj),
                       ('frobenius-mult-sum-svos', ell_frobenius_mult_sum_subj, ell_frobenius_mult_sum_obj)]


ell_cat_subject_mult_subj = make_concrete_model("CATS-mult-svos", EllipsisModel, ell_cat_subject_mult, setting='subj')
ell_cat_subject_mult_obj = make_concrete_model("CATS-mult-svos", EllipsisModel, ell_cat_subject_mult, setting='obj')

ell_cat_object_mult_subj = make_concrete_model("CATO-mult-svos", EllipsisModel, ell_cat_object_mult, setting='subj')
ell_cat_object_mult_obj = make_concrete_model("CATO-mult-svos", EllipsisModel, ell_cat_object_mult, setting='obj')

ell_copy_subject_mult_subj = make_concrete_model("copy-subject-mult-svos", EllipsisModel, ell_copy_subject_mult, setting='subj')
ell_copy_subject_mult_obj = make_concrete_model("copy-subject-mult-svos", EllipsisModel, ell_copy_subject_mult, setting='obj')

ell_copy_object_mult_subj = make_concrete_model("copy-object-mult-svos", EllipsisModel, ell_copy_object_mult, setting='subj')
ell_copy_object_mult_obj = make_concrete_model("copy-object-mult-svos", EllipsisModel, ell_copy_object_mult, setting='obj')


ell_copy_subject_sum_mult_obj = make_concrete_model("copy-subject-sum-mult-svos", EllipsisModel, ell_copy_subject_sum_mult, setting='obj')
ell_copy_object_sum_mult_subj = make_concrete_model("copy-subject-sum-mult-svos", EllipsisModel, ell_copy_object_sum_mult, setting='subj')


ell_frobenius_add_mult_subj = make_concrete_model("frobenius-add-mult-svos", EllipsisModel, ell_frobenius_add_mult, setting='subj')
ell_frobenius_add_mult_obj = make_concrete_model("frobenius-add-mult-svos", EllipsisModel, ell_frobenius_add_mult, setting='obj')

ell_frobenius_mult_mult_subj = make_concrete_model("frobenius-mult-mult-svos", EllipsisModel, ell_frobenius_mult_mult, setting='subj')
ell_frobenius_mult_mult_obj = make_concrete_model("frobenius-mult-mult-svos", EllipsisModel, ell_frobenius_mult_mult, setting='obj')


ell_models_late_mult = [('cat-subject-mult-svos', ell_cat_subject_mult_subj, ell_cat_subject_mult_obj),
                        ('cat-object-mult-svos', ell_cat_object_mult_subj, ell_cat_object_mult_obj),
                        ('copy-subject-mult-svos', ell_copy_subject_mult_subj, ell_copy_subject_mult_obj),
                        ('copy-object-mult-svos', ell_copy_object_mult_subj, ell_copy_object_mult_obj),
                        ('cat-argument-mult-svos', ell_cat_subject_mult_subj, ell_cat_object_mult_obj),
                        ('copy-argument-mult-svos', ell_copy_object_mult_subj, ell_copy_subject_mult_obj),
                        ('copy-argument-sum-mult-svos', ell_copy_object_sum_mult_subj, ell_copy_subject_sum_mult_obj),
                        ('frobenius-add-mult-svos', ell_frobenius_add_mult_subj, ell_frobenius_add_mult_obj),
                        ('frobenius-mult-mult-svos', ell_frobenius_mult_mult_subj, ell_frobenius_mult_mult_obj)]

ell_models_late = ell_models_late_sum + ell_models_late_mult
# catArgModelsMiddle = [TransitiveModelTensorSeparateMiddle("Cat Argument_%s" % a, "CatArg_%s" % a, cat_subject, cat_object, a) for a in alphas]

# catArgModelsLate = [TransitiveModelTensorSeparateLate("Cat Argument_%s" % a, "CatArg_%s" % a, cat_subject, cat_object, a) for a in alphas]

paragaps_model_basic_subj = make_concrete_model("basic", ParagapsModel, paragaps_basic, setting='subj')
paragaps_model_basic_obj = make_concrete_model("basic", ParagapsModel, paragaps_basic, setting='obj')
