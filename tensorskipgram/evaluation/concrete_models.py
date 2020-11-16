from tensorskipgram.evaluation.composers import cat_intrans, cat_subject, cat_object
from tensorskipgram.evaluation.composers \
    import copy_subject, copy_object, copy_subject_sum, copy_object_sum
from tensorskipgram.evaluation.composers import frobenius_add, frobenius_mult
from tensorskipgram.evaluation.composers \
    import cat_argument, copy_argument, copy_argument_sum
from tensorskipgram.evaluation.composers import ell_wrapper
from tensorskipgram.evaluation.spaces import VectorSpace, MatrixSpace
from tensorskipgram.evaluation.composition_models \
    import IntransitiveModel, TransitiveModel, EllipsisModel
from tensorskipgram.evaluation.composition_models \
    import IntransitiveTwoModel, TransitiveTwoModel, EllipsisTwoModel
from tensorskipgram.evaluation.composition_models \
    import IntransitiveTwoModelMix, TransitiveTwoModelMix, EllipsisTwoModelMix
from tensorskipgram.config import noun_space_fn, model_path_subj_conc, model_path_obj_conc

skipgram_space = VectorSpace(name="skipgram100", path=noun_space_fn)
skipgram_subj_mats = MatrixSpace(name="skipgram_subj_mat", path=model_path_subj_conc)
skipgram_obj_mats = MatrixSpace(name="skipgram_obj_mat", path=model_path_obj_conc)


def make_concrete_model(name, model_class, composer):
    return model_class(name, skipgram_space, skipgram_subj_mats, skipgram_obj_mats)


def make_concrete_mix_model(name, model_class, composer_subj, composer_obj, alpha: float):
    return model_class(name, skipgram_space, skipgram_subj_mats, skipgram_obj_mats, alpha)


""" Intransitive Models """
cat_intrans_model = make_concrete_model("CAT-sv", IntransitiveTwoModel, cat_intrans)


""" Transitive Models """

cat_subject_model = make_concrete_model("CATS-svo", TransitiveTwoModel, cat_subject)
cat_object_model = make_concrete_model("CATO-svo", TransitiveTwoModel, cat_subject)
copy_subject_model = make_concrete_model("copy-subject-svo", TransitiveTwoModel, copy_subject)
copy_object_model = make_concrete_model("copy-object-svo", TransitiveTwoModel, copy_object)
frobenius_add_model = make_concrete_model("frobenius-add-svo", TransitiveTwoModel, frobenius_add)
frobenius_mult_model = make_concrete_model("frobenius-mult-svo", TransitiveTwoModel, frobenius_mult)

alphas = [a/10. for a in range(0, 11)]
copy_argument_models = [make_concrete_mix_model("copy_argument-svo", TransitiveTwoModelMix,
                        copy_subject, copy_object, a) for a in alphas]
copy_argument_sum_models = [make_concrete_mix_model("copy-argument-sum-svo", TransitiveTwoModelMix,
                            copy_subject_sum, copy_object_sum, a) for a in alphas]
cat_argument_models = [make_concrete_mix_model("cat-argument-svo", TransitiveTwoModelMix,
                       cat_subject, cat_object, a) for a in alphas]

""" Ellipsis Models """
