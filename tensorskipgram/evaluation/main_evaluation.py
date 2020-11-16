"""Evaluate some models on evaluation tasks."""
from tensorskipgram.tasks.datasets \
    import create_ml2008, create_ml2010, create_gs2011, create_ks2013
from tensorskipgram.tasks.datasets \
    import create_ks2014, create_elldis, create_ellsim
from tensorskipgram.evaluation.concrete_models import cat_intrans_model
from tensorskipgram.evaluation.concrete_models import cat_subject_model, cat_object_model
from tensorskipgram.evaluation.concrete_models import ell_models
from tensorskipgram.evaluation.evaluator import evaluate_model_on_task


def evaluate_intransitive_models() -> None:
    """Load all tasks, and models, and compute spearman correlations."""
    ml2008, ml2010 = create_ml2008(), create_ml2010()
    result_dict = {}
    result_dict[ml2008.name] = {m.name: evaluate_model_on_task(m, ml2008)
                                for m in [cat_intrans_model]}
    result_dict[ml2010.name] = {m.name: evaluate_model_on_task(m, ml2010)
                                for m in [cat_intrans_model]}
    return result_dict


def evaluate_transitive_models() -> None:
    """Load all tasks, and models, and compute spearman correlations."""
    gs2011, ks2013, ks2014 = create_gs2011(), create_ks2013(), create_ks2014()
    result_dict = {}
    trans_models = [cat_subject_model, cat_object_model]
    result_dict[gs2011.name] = {m.name: evaluate_model_on_task(m, gs2011)
                                for m in trans_models}
    result_dict[ks2013.name] = {m.name: evaluate_model_on_task(m, ks2013)
                                for m in trans_models}
    result_dict[ks2014.name] = {m.name: evaluate_model_on_task(m, ks2014)
                                for m in trans_models}
    return result_dict


def evaluate_ellipsis_models() -> None:
    """Load all tasks, and models, and compute spearman correlations."""
    elldis, ellsim = create_elldis(), create_ellsim()
    result_dict = {}
    result_dict[elldis.name] = {m.name: evaluate_model_on_task(m, elldis)
                                for m in ell_models}
    result_dict[ellsim.name] = {m.name: evaluate_model_on_task(m, ellsim)
                                for m in ell_models}
    return result_dict


def evaluate_all_models() -> None:
    """Load all tasks, and models, and compute spearman correlations."""
    intrans_results = evaluate_intransitive_models()
    trans_results = evaluate_transitive_models()
    ellipsis_results = evaluate_ellipsis_models()
    return intrans_results, trans_results, ellipsis_results


def main() -> None:
    intrans_results, trans_results, ellipsis_results = evaluate_all_models()
