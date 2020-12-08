"""Evaluate some models on evaluation tasks."""
from tqdm import tqdm
from typing import Dict
from tensorskipgram.tasks.datasets import create_paragaps
from tensorskipgram.config import paragaps_path
from tensorskipgram.tasks.datasets \
    import (create_ml2008, create_ml2010, create_gs2011, create_ks2013,
            create_ks2014, create_elldis, create_ellsim)
from tensorskipgram.evaluation.concrete_models \
    import (cat_intrans_models_early, cat_intrans_models_mid, intrans_models_late,
            trans_models_early, trans_models_mid, trans_models_two, trans_models_late,
            ell_models_early, ell_models_mid, ell_models_late, alphas)
from tensorskipgram.evaluation.evaluator \
    import (evaluate_model_on_task, evaluate_model_on_task_late_fusion)
from tensorskipgram.config import (ml2008_path, ml2010_path, gs2011_path, ks2013_path,
                                   ks2014_path, elldis_path, ellsim_path)
from tensorskipgram.evaluation.concrete_models \
    import paragaps_model_basic_subj, paragaps_model_basic_obj


def evaluate_intransitive_models() -> None:
    """Load all tasks, and models, and compute spearman correlations."""
    ml2008, ml2010 = create_ml2008(ml2008_path), create_ml2010(ml2010_path)
    intrans_models = cat_intrans_models_early + cat_intrans_models_mid
    result_dict = {}
    result_dict[ml2008.name + '-early-mid'] = {m.name: evaluate_model_on_task(m, ml2008)
                                              for m in tqdm(intrans_models)}
    result_dict[ml2008.name + '-late'] = {n + f'-alpha-{a}': evaluate_model_on_task_late_fusion(m1, m2, ml2008, a)
                                          for (n, m1, m2) in tqdm(intrans_models_late) for a in alphas}
    result_dict[ml2010.name + '-early-mid'] = {m.name: evaluate_model_on_task(m, ml2010)
                                              for m in tqdm(intrans_models)}
    result_dict[ml2010.name + '-late'] = {n + f'-alpha-{a}': evaluate_model_on_task_late_fusion(m1, m2, ml2010, a)
                                          for (n, m1, m2) in tqdm(intrans_models_late) for a in alphas}
    return result_dict


def evaluate_transitive_models() -> None:
    """Load all tasks, and models, and compute spearman correlations."""
    gs2011, ks2013, ks2014 = create_gs2011(gs2011_path), create_ks2013(ks2013_path), create_ks2014(ks2014_path)
    trans_models = trans_models_early + trans_models_mid + trans_models_two
    result_dict = {}
    result_dict[gs2011.name + '-early-mid'] = {m.name: evaluate_model_on_task(m, gs2011)
                                               for m in tqdm(trans_models)}
    result_dict[gs2011.name + '-late'] = {n + f'-alpha-{a}': evaluate_model_on_task_late_fusion(m1, m2, gs2011, a)
                                          for (n, m1, m2) in tqdm(trans_models_late) for a in alphas}
    result_dict[ks2013.name + '-early-mid'] = {m.name: evaluate_model_on_task(m, ks2013)
                                               for m in tqdm(trans_models)}
    result_dict[ks2013.name + '-late'] = {n + f'-alpha-{a}': evaluate_model_on_task_late_fusion(m1, m2, ks2013, a)
                                          for (n, m1, m2) in tqdm(trans_models_late) for a in alphas}
    result_dict[ks2014.name + '-early-mid'] = {m.name: evaluate_model_on_task(m, ks2014)
                                               for m in tqdm(trans_models)}
    result_dict[ks2014.name + '-late'] = {n + f'-alpha-{a}': evaluate_model_on_task_late_fusion(m1, m2, ks2014, a)
                                          for (n, m1, m2) in tqdm(trans_models_late) for a in alphas}
    return result_dict


def evaluate_ellipsis_models() -> None:
    """Load all tasks, and models, and compute spearman correlations."""
    elldis, ellsim = create_elldis(elldis_path), create_ellsim(ellsim_path)
    ell_models = ell_models_early + ell_models_mid
    result_dict = {}
    result_dict[elldis.name + '-early-mid'] = {m.name: evaluate_model_on_task(m, elldis)
                                               for m in tqdm(ell_models)}
    result_dict[elldis.name + '-late'] = {n + f'-alpha-{a}': evaluate_model_on_task_late_fusion(m1, m2, elldis, a)
                                          for (n, m1, m2) in tqdm(ell_models_late) for a in alphas}
    result_dict[ellsim.name + '-early-mid'] = {m.name: evaluate_model_on_task(m, ellsim)
                                               for m in tqdm(ell_models)}
    result_dict[ellsim.name + '-late'] = {n + f'-alpha-{a}': evaluate_model_on_task_late_fusion(m1, m2, ellsim, a)
                                          for (n, m1, m2) in tqdm(ell_models_late) for a in alphas}
    return result_dict


def evaluate_all_models() -> None:
    """Load all tasks, and models, and compute spearman correlations."""
    intrans_results = evaluate_intransitive_models()
    trans_results = evaluate_transitive_models()
    ellipsis_results = evaluate_ellipsis_models()
    return intrans_results, trans_results, ellipsis_results


def get_max_results(result_dict: Dict) -> Dict:
    """Compute the highest results for a dictionary of results."""
    return {k: round(result_dict[k][max(result_dict[k], key=lambda d: result_dict[k][d][0])][0], 3) for k in result_dict}


def main() -> None:
    """Run all experiments on all tasks, and print the best results."""
    intrans_results, trans_results, ellipsis_results = evaluate_all_models()
    max_res_intrans = get_max_results(intrans_results)
    max_res_trans = get_max_results(trans_results)
    max_res_ellipsis = get_max_results(ellipsis_results)
    for k in max_res_intrans.keys():
        print(k, max_res_intrans[k])
    for k in max_res_trans.keys():
        print(k, max_res_trans[k])
    for k in max_res_ellipsis.keys():
        print(k, max_res_ellipsis[k])
