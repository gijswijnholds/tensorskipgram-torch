# tensorskipgram-torch

This repository contains code to learn multi-linear maps for verbs, accompanying the paper

Gijs Wijnholds and Mehrnoosh Sadrzadeh. *Representation Learning for Type-Driven Composition.* CoNLL 2020.

If you use any of this code or the available representations, or you made any other use of this repository, please cite

```
@inproceedings{wijnholds-etal-2020-representation,
    title = "Representation Learning for Type-Driven Composition",
    author = "Wijnholds, Gijs  and
      Sadrzadeh, Mehrnoosh  and
      Clark, Stephen",
    booktitle = "Proceedings of the 24th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.conll-1.24",
    pages = "313--324",
    abstract = "This paper is about learning word representations using grammatical type information. We use the syntactic types of Combinatory Categorial Grammar to develop multilinear representations, i.e. maps with n arguments, for words with different functional types. The multilinear maps of words compose with each other to form sentence representations. We extend the skipgram algorithm from vectors to multi- linear maps to learn these representations and instantiate it on unary and binary maps for transitive verbs. These are evaluated on verb and sentence similarity and disambiguation tasks and a subset of the SICK relatedness dataset. Our model performs better than previous type- driven models and is competitive with state of the art representation learning methods such as BERT and neural sentence encoders.",
}
```

## Code

In order to recreate the experiments, you will need to perform three steps, as outlined in `main.py`:

1. Preprocess a corpus of choice to extract SVO triples. In our case we use UKWackyPedia, but you can implement your corpus as well. This code is in the *preprocessing* folder.
2. Train matrices. This code you find in the *training* folder.
3. Evaluate the matrices on a selection of tasks. This is done in the *evaluation* folder.

## Models

We publish the models of our proposed method, which are a pair of matrix representations per verb, each one of which multiplies with the vectors for one of the verb's arguments to predict the other argument. The models cover a list of 1160 verbs, that were extracted from the evaluation datasets. Of course, you can retrain the model on another set of verbs.

We do *not* provide the verb cube model or the ablation matrix model, as they underperform and are too large to reasonably store online.

| Model Name      | Dimensions | Link to Embeddings       |
| --------------- |:----------:| :----------------------: |
| Noun (skipgram) | 100        | [link][skipgram_vectors] |
| *Mat* x Subj    | 100x100    | [link][mat_subj_matrices]|
| *Mat* x Obj     | 100x100    | [link][mat_obj_matrices] |
| Relational      | 100x100    | [link][rel_matrices] |
| Kronecker       | 100x100    | [link][kron_matrices] |
[skipgram_vectors]: https://ln2.sync.com/dl/9fbd93010/dd7rbij3-7vb8zkh2-87xf8mwk-wb9xvta6
[mat_subj_matrices]: https://ln2.sync.com/dl/dcafc01e0/2fvmq7tb-d2e6nnn7-7vnhviya-99kkb4bx
[mat_obj_matrices]: https://ln2.sync.com/dl/cc5276da0/gphdbdp4-tkwem5ci-f3mg5fvk-at6r2h9b
[rel_matrices]: https://ln2.sync.com/dl/cc5276da0/gphdbdp4-tkwem5ci-f3mg5fvk-at6r2h9b
[kron_matrices]: https://ln2.sync.com/dl/cc5276da0/gphdbdp4-tkwem5ci-f3mg5fvk-at6r2h9b


## Evaluation Results

We display the results that we obtain on multiple evaluation datasets in the table below:

| Dataset   | Spearman rho | Human agreement |
| --------- |:------------:|:---------------:|
| ML2008    |     0.19     |      0.66       |
| ML2010    |     0.55     |      0.71       |
| GS2011    |     0.54     |      0.74       |
| KS2013a   |     0.37     |      0.58       |
| KS2013b   |     0.75     |      0.75       |
| ELLDIS    |     0.56     |      0.58       |
| ELLSIM    |     0.76     |      0.43       |

On SICK-R, we achieve a Pearson correlation score of 0.70.
