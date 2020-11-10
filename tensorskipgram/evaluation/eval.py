import os
import torch
from tensorskipgram.evaluation.data import SICKPreprocessor, SICKDataset, SICKDatasetNouns
from tensorskipgram.evaluation.model \
    import SentenceEmbedderSimilarity, SentenceEmbedderSimilarity2, SentenceEmbedderSimilarityDot, SentenceEmbedderEntailment
from tensorskipgram.evaluation.model \
    import VecSimDot, VecSimLinear, VecSimLinear2, VecSimEntailment
from tensorskipgram.evaluation.trainer import *
from tensorskipgram.evaluation.paths import *
from tensorskipgram.evaluation.eval_load import *
from tensorskipgram.data.util import load_obj_fn, dump_obj_fn
from torch.utils.data import DataLoader


def eval_vecs_vs_mats_relatedness(num_epochs=10):
    vec_model = VecSimLinear(noun_matrix, 50, flex_nouns=False)
    vec_loss_fn = torch.nn.KLDivLoss()
    vec_opt = torch.optim.Adam(vec_model.parameters(), lr=0.01)
    sick_noun_dataloader = DataLoader(sick_noun_dataset_train, shuffle=True,
                                      batch_size=1)
    vec_results = train_epochs('vec_model_relatedness', vec_model,
                               sick_noun_dataloader, sick_noun_dataset_train,
                               sick_noun_dataset_dev, sick_noun_dataset_test,
                               vec_loss_fn, vec_opt, 'cpu', num_epochs)
    # max(vec_results[1]), max(vec_results[2]), max(vec_results[3])
    # 10 epochs:
    # (0.8096670635042296, 0.7098549911085893, 0.7157704446369832)
    # 100 epochs:
    # (0.9531877518787207, 0.7133568854585155, 0.7157704446369832)
    mat_model = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube,
                                           verb_obj_cube, 50, flex_nouns=False,
                                           flex_verbs=False)
    mat_loss_fn = torch.nn.KLDivLoss()
    mat_opt = torch.optim.Adam(mat_model.parameters(), lr=0.01)
    sick_dataloader = DataLoader(sick_dataset_train, shuffle=True,
                                 batch_size=1)
    mat_results = train_epochs('mat_model_relatedness', mat_model,
                               sick_dataloader, sick_dataset_train,
                               sick_dataset_dev, sick_dataset_test,
                               mat_loss_fn, mat_opt, 'cpu', num_epochs)
    # leaving out verbs and arguments
    # max(mat_results[1]), max(mat_results[2]), max(mat_results[3])
    # 10 epochs:
    # (0.6137628622780154, 0.49017651662727346, 0.4749577324259667)
    # 100 epochs:
    # (0.8678987510034879, 0.5018126196163354, 0.4858539505764354)
    # with verbs and arguments:
    # 10 epochs:
    # (0.8420799577963197, 0.40137649426774213, 0.47185027950105096)
    # 100 epochs:
    # (0.9735609555258457, 0.40137649426774213, 0.47185027950105096)
    return vec_results, mat_results


# Have a model for just vectors, and one with the matrices as well
def eval_vecs_vs_mats_entailment(num_epochs=10):
    vec_model = VecSimEntailment(noun_matrix, 50, flex_nouns=False)
    vec_opt = torch.optim.Adam(vec_model.parameters(), lr=0.00075)
    vec_loss_fn = torch.nn.CrossEntropyLoss()
    sick_noun_dataloader_e = DataLoader(sick_noun_dataset_e_train,
                                        shuffle=True, batch_size=1)
    vec_results = train_epochs_e('vec_model_fix_all_tanh', vec_model,
                                 sick_noun_dataloader_e,
                                 sick_noun_dataset_e_train,
                                 sick_noun_dataset_e_dev,
                                 sick_noun_dataset_e_test,
                                 vec_loss_fn, vec_opt, 'cpu', num_epochs)
    # max(vec_results[1]), max(vec_results[2]), max(vec_results[3])
    # 10 epochs:
    # (0.7690921378688894, 0.7313131313131314, 0.7244190786791683)
    # 100 epochs:
    # (0.9952692047758505, 0.7414141414141414, 0.7458214431308602)
    mat_model = SentenceEmbedderEntailment(noun_matrix, verb_subj_cube,
                                           verb_obj_cube, 50, flex_nouns=False,
                                           flex_verbs=False)
    mat_opt = torch.optim.Adam(mat_model.parameters(), lr=0.00075)
    mat_loss_fn = torch.nn.CrossEntropyLoss()
    sick_dataloader_e = DataLoader(sick_dataset_e_train, shuffle=True,
                                   batch_size=1)
    mat_results = train_epochs_e('mat_model_fix_all_tanh', mat_model,
                                 sick_dataloader_e, sick_dataset_e_train,
                                 sick_dataset_e_dev, sick_dataset_e_test,
                                 mat_loss_fn, mat_opt, 'cpu', num_epochs)
    # leaving out verbs and arguments
    # max(mat_results[1]), max(mat_results[2]), max(mat_results[3])
    # 10 epochs:
    # (0.7044379364721784, 0.696969696969697, 0.6646962902568284)
    # 100 epochs:
    # (0.9538184275737779, 0.6868686868686869, 0.6860986547085202)
    # with verbs and arguments:
    # 10 epochs:
    # (0.7726965532777652, 0.6505050505050505, 0.6443130860171219)
    # 100 epochs:
    # (0.9817526469925659, 0.6505050505050505, 0.6469629025682837)
    return vec_results, mat_results


def eval_vecs_vs_mats_relatedness_relational(num_epochs=10):
    # noun_matrix, verb_subj_cube, verb_obj_cube = recreate_representations_relational()
    noun_matrix, verb_subj_cube, verb_obj_cube = reload_representations_relational()
    mat_model = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube,
                                           verb_obj_cube, 50, flex_nouns=False,
                                           flex_verbs=False)
    mat_loss_fn = torch.nn.KLDivLoss()
    mat_opt = torch.optim.Adam(mat_model.parameters(), lr=0.01)
    sick_dataloader = DataLoader(sick_dataset_train, shuffle=True,
                                 batch_size=1)
    mat_results = train_epochs('mat_model_relatedness', mat_model,
                               sick_dataloader, sick_dataset_train,
                               sick_dataset_dev, sick_dataset_test,
                               mat_loss_fn, mat_opt, 'cpu', num_epochs)
    # leaving out verbs and arguments
    # max(mat_results[1]), max(mat_results[2]), max(mat_results[3])
    # with verbs and arguments:
    # max(mat_results[1]), max(mat_results[2]), max(mat_results[3])
    # 10 epochs:
    # (0.5309653253302434, 0.4135493108761163, 0.4046116905839026)
    # 100 epochs, With adagrad:
    # (0.6085218165182982, 0.39965360390427257, 0.4792003996023485)
    # 100 epochs:
    # (0.8246867658243521, 0.43214200771875505, 0.46620026805819204)
    # KRONECKER:
    # 10 epochs:
    # (0.5352080956347902, 0.41644946062153876, 0.4490823266371673)
    # 100 epochs:
    # (0.785493002939924, 0.49350520499206646, 0.49657922512770514)
    return mat_results

# embedder_model_fix_all = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=False, flex_verbs=False)
# opt_fix_all = torch.optim.Adam(embedder_model_fix_all.parameters(), lr=0.01)
# sick_dataloader = DataLoader(sick_dataset_train, shuffle=True, batch_size=1)
# loss_fn = torch.nn.KLDivLoss()

# embedder_model_fix_all

# results = train_epochs('verb_model_fix_all_sigmoid', embedder_model_fix_all, sick_dataloader, sick_dataset_train, sick_dataset_dev, sick_dataset_test, loss_fn, opt_fix_all, 'cpu', 15)


# embedder_model_fix_all = SentenceEmbedderEntailment(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=False, flex_verbs=False)
# opt_fix_all = torch.optim.Adam(embedder_model_fix_all.parameters(), lr=0.001)
# loss_fn = torch.nn.CrossEntropyLoss()
# sick_dataloader_e = DataLoader(sick_dataset_e_train, shuffle=True, batch_size=1)

# results = train_epochs_e('verb_model_fix_all_sigmoid', embedder_model_fix_all, sick_dataloader_e, sick_dataset_e_train, sick_dataset_e_dev, sick_dataset_e_test, loss_fn, opt_fix_all, 'cpu', 10)
# results = train_epochs_dot('verb_model2', embedder_model_fix_all, sick_dataloader, sick_dataset_train, sick_dataset_dev, sick_dataset_test, loss_fn, opt_fix_all, 'cpu', 5)
# evaluate_e(embedder_model_fix_all, sick_dataset_e_dev)
# embedder_model_flex_verbs = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=False, flex_verbs=True)
# embedder_model_flex_nouns = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=True, flex_verbs=False)
# embedder_model_flex_all = SentenceEmbedderSimilarity(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=True, flex_verbs=True)
# loss_fn = torch.nn.KLDivLoss()
# opt_fix_all = torch.optim.Adam(embedder_model_fix_all.parameters(), lr=0.0075)
# sick_dataloader = DataLoader(sick_dataset_train, shuffle=True, batch_size=1)
# model_results = {}
# model_results['fix_all'] = \
#     train_epochs(embedder_model_fix_all, sick_dataloader,
#                  sick_dataset_dev, sick_dataset_test, loss_fn,
#                  opt_fix_all, 'cpu', 5)
# dot_embedder_model_fix_all = SentenceEmbedderSimilarityDot(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=False, flex_verbs=False)
# dot_embedder_model_flex_verbs = SentenceEmbedderSimilarityDot(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=False, flex_verbs=True)
# dot_embedder_model_flex_nouns = SentenceEmbedderSimilarityDot(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=True, flex_verbs=False)
# dot_embedder_model_flex_all = SentenceEmbedderSimilarityDot(noun_matrix, verb_subj_cube, verb_obj_cube, 50, flex_nouns=True, flex_verbs=True)
# opt_flex_verbs_dot = torch.optim.Adam(dot_embedder_model_flex_verbs.parameters(), lr=0.01)
# opt_flex_nouns_dot = torch.optim.Adam(dot_embedder_model_flex_nouns.parameters(), lr=0.01)
# opt_flex_all_dot = torch.optim.Adam(dot_embedder_model_flex_all.parameters(), lr=0.01)
# loss_fn = torch.nn.MSELoss()
# sick_dataloader = DataLoader(sick_dataset_train, shuffle=True, batch_size=1)
# sick_dataloader_filter = DataLoader(sick_dataset_train_filter, shuffle=True, batch_size=1)

# model_results = {}
# model_configs = [('dot_flex_nouns', dot_embedder_model_flex_nouns, opt_flex_nouns_dot, 'nofilter'),
#                  ('dot_flex_verbs', dot_embedder_model_flex_verbs, opt_flex_verbs_dot, 'filter'),
#                  ('dot_flex_all', dot_embedder_model_flex_all, opt_flex_all_dot, 'nofilter')]
#

# for (name, model, opt, filterbool) in model_configs:
#     if filterbool == 'filter':
#         loader = sick_dataloader_filter
#     elif filterbool == 'nofilter':
#         loader = sick_dataloader
#     model_results[name] = train_epochs_dot(name, model, loader,
#                                            sick_dataset_train,
#                                            sick_dataset_dev,
#                                            sick_dataset_test,
#                                            loss_fn, opt, 'cpu', 10)
# model_results['dot_flex_nouns'] = \
#     train_epochs_dot('dot_flex_nouns', dot_embedder_model_flex_nouns, sick_dataloader,
#                      sick_dataset_train, sick_dataset_dev, sick_dataset_test, loss_fn,
#                      opt_flex_nouns_dot, 'cpu', 5)
# model_results['dot_flex_verbs'] = \
#     train_epochs_dot('dot_flex_verbs', dot_embedder_model_flex_verbs, sick_dataloader_filter,
#                      sick_dataset_dev, sick_dataset_test, loss_fn,
#                      opt_flex_verbs_dot, 'cpu', 5)
# model_results['dot_flex_all'] = \
#     train_epochs_dot('dot_flex_all', dot_embedder_model_flex_all, sick_dataloader,
#                      sick_dataset_dev, sick_dataset_test, loss_fn,
#                      opt_flex_all_dot, 'cpu', 5)

# epoch_losses, evals, tests = train_epochs_dot(dot_embedder_model_flex_verbs, sick_dataloader, sick_dataset_dev, sick_dataset_test, loss_fn, opt_fix_all, 'cpu', 5)



# epoch_losses, evals, tests = train_epochs(embedder_model_flex_verbs, sick_dataloader, sick_dataset_dev, sick_dataset_test, loss_fn, opt_fix_all, 'cpu', 5)


# epoch_loss1 = train_epoch(embedder_model_fix_all, sick_dataloader, loss_fn, opt_fix_all, 'cpu', 1)

# epoch_loss = train_epoch_dot(dot_embedder_model_flex_verbs, sick_dataloader, loss_fn, opt_flex_verbs, 'cpu', 1)

# epoch_losses, evals, tests = train_epochs(embedder_model_flex_verbs, sick_dataloader, sick_dataset_dev, sick_dataset_test, loss_fn, opt_fix_all, 'cpu', 5)

# model_configs = [('fix_all', embedder_model_fix_all, opt_fix_all),
#                  ('flex_nouns', embedder_model_flex_nouns, opt_flex_nouns),
#                  ('flex_verbs', embedder_model_flex_verbs, opt_flex_verbs),
#                  ('flex_all', embedder_model_flex_all, opt_flex_all)]
# num_epochs = 1

# model_results = {'fix_all': (), 'flex_nouns': (), 'flex_verbs': (), 'flex_all': ()}
#
# for (name, model, opt) in model_configs:
#     epoch_losses = []
#     evals = [evaluate(model, sick_dataset_dev)]
#     tests = [evaluate(model, sick_dataset_test)]
#     for i in range(num_epochs):
#         epoch_losses.append(train_epoch(model, sick_dataloader, loss_fn, opt, 'cpu', i+1))
#         eval = evaluate(model, sick_dataset_dev)
#         evals.append(eval)
#         test = evaluate(model, sick_dataset_test)
#         tests.append(test)
#         print(f"Eval: {eval},    Test: {test}   Model: {name}")
#     model_results[name] = (epoch_losses, evals, tests)
#




# vec_mats = {'noun_matrix': noun_matrix, 'verb_subj_cube': verb_subj_cube, 'verb_obj_cube': verb_obj_cube}
# dump_obj_fn(vec_mats, model_data_fn)
#
static_model = VecSimDot(noun_matrix, 100, flex_nouns=False)
evaluate_dot(static_model, sick_noun_dataset_train)
evaluate_dot(static_model, sick_noun_dataset_dev)
evaluate_dot(static_model, sick_noun_dataset_test)


# flex_model = VecSimDot(noun_matrix, 100, flex_nouns=True)
# evaluate_dot(flex_model, sick_noun_dataset_train)
# evaluate_dot(flex_model, sick_noun_dataset_dev)
# evaluate_dot(flex_model, sick_noun_dataset_test)

# static_model_linear = VecSimLinear(noun_matrix, 50, flex_nouns=False)
# static_model_linear = VecSimLinear2(noun_matrix, 50, flex_nouns=False)
# loss_fn = torch.nn.KLDivLoss()
# evaluate(static_model_linear, sick_noun_dataset_train)

# loss_fn = torch.nn.MSELoss()
# opt_flex = torch.optim.Adam(flex_model.parameters(), lr=0.01)
# opt_linear = torch.optim.Adam(static_model_linear.parameters(), lr=0.01)
# sick_noun_dataloader = DataLoader(sick_noun_dataset_train, shuffle=True, batch_size=1)
# results = train_epochs('static_model_linear', static_model_linear, sick_noun_dataloader, sick_noun_dataset_train, sick_noun_dataset_dev, sick_noun_dataset_test, loss_fn, opt_linear, 'cpu', 15)
# results = train_epochs_dot('flex_model', flex_model, sick_noun_dataloader,
#                            sick_noun_dataset_train, sick_noun_dataset_dev, sick_noun_dataset_test, loss_fn,
                           # opt_flex, 'cpu', 5)
# loss_fn = torch.nn.KLDivLoss()

# opt_flex_verbs = torch.optim.Adam(embedder_model_flex_verbs.parameters(), lr=0.0075)
# opt_flex_nouns = torch.optim.Adam(embedder_model_flex_nouns.parameters(), lr=0.0075)
# opt_flex_all = torch.optim.Adam(embedder_model_flex_all.parameters(), lr=0.0075)
# opt_fix_all = torch.optim.Adam(embedder_model_fix_all.parameters(), lr=0.0075)
# opt_flex_verbs_dot = torch.optim.Adam(dot_embedder_model_flex_verbs.parameters(), lr=0.0075)
# opt_flex_all_dot = torch.optim.Adam(dot_embedder_model_flex_all.parameters(), lr=0.0075)



# Results for dot-embedders:
# model_results
# {'dot_flex_verbs': ([0.2568059854315886, 0.16635961296924584, 0.1469770215114483, 0.13252882478274255, 0.12003039491614557], [0.35621489269527107, 0.4301284986668193, 0.46760520386335586, 0.4602146445818622, 0.49370054543020736, 0.4742499963810401], [0.3963879245983484, 0.5012396364528565, 0.5031630659020132, 0.5104925562376401, 0.5169757525872021, 0.5236893009561678]), 'dot_flex_all': ([0.26230038825111307, 0.15247191474587676, 0.13153560618438734, 0.1136442863362579, 0.10272791943611262], [0.35621489269527107, 0.5247930896103069, 0.561895183558694, 0.5428356768676464, 0.5737065446369748, 0.5645925996064798], [0.3963879245983484, 0.5543708367870215, 0.566286093808108, 0.5804495948523976, 0.5983446930552828, 0.5937246679738968]), 'dot_flex_nouns': ([0.2482688429470698, 0.15317785698118458, 0.12527591080410627, 0.11160531853802587, 0.10241876648204801], [0.35621489269527107, 0.5195124325580049, 0.5463827503647728, 0.5509934577606579, 0.5579033435616314, 0.563516513141701], [0.3963879245983484, 0.5206404793778241, 0.5532054610251935, 0.5607130509202083, 0.5701995619157101, 0.5713027181921023])}
