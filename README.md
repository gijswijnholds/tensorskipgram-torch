# tensorskipgram-torch

This repository contains code to learn multi-linear maps for verbs, accompanying the paper

Gijs Wijnholds and Mehrnoosh Sadrzadeh. *Representation Learning for Type-Driven Composition.* CoNLL 2020.

If you use any of this code or the available representations, or you made any other use of this repository, please cite

```
@inproceedings{wijnholds2019representation,
  title = "Representation Learning for Type-Driven Composition",
  author = "Gijs Wijnholds and Mehrnoosh Sadrzadeh and Stephen Clark",
  year = "2020",
  booktitle={Proceedings of the 2020 Conference on Natural Language Learning (CoNLL) Volume 1 (Long Papers)},
  publisher={Association for Computational Linguistics}
}
```

## Code

## Models

We publish the models of our proposed method, which are a pair of matrix representations per verb, each one of which multiplies with the vectors for one of the verb's arguments to predict the other argument. The models cover a list of 1160 verbs, that were extracted from the evaluation datasets. Of course, you can retrain the model on another set of verbs.

We do *not* provide the verb cube model or the ablation matrix model, as they underperform and are too large to reasonably store online.

| Model Name    | Dimensions | Vectors                  | Tensors                  |
| ------------- |:----------:| :----------------------: | :----------------------: |
| Mat x Subj    | 100x100    | [link][skipgram_vectors] | [link][mat_subj_matrices]|
| Mat x Obj     | 100x100    | [link][skipgram_vectors] | [link][mat_obj_matrices] |

[skipgram_vectors]: https://ln2.sync.com/dl/9fbd93010/dd7rbij3-7vb8zkh2-87xf8mwk-wb9xvta6
[mat_subj_matrices]: https://ln2.sync.com/dl/dcafc01e0/2fvmq7tb-d2e6nnn7-7vnhviya-99kkb4bx
[mat_obj_matrices]: https://ln2.sync.com/dl/cc5276da0/gphdbdp4-tkwem5ci-f3mg5fvk-at6r2h9b

## Evaluation Results

We display the results that we obtain on multiple evaluation datasets in the table below:

| Dataset   | Spearman rho | Human agreement |
| --------- |:------------:|:---------------:|
| ML2008    |     xx       |       xx        |
| ML2010    |     xx       |       xx        |
| GS2011    |     xx       |       xx        |
| KS2013a   |     xx       |       xx        |
| KS2013b   |     xx       |       xx        |
| ELLDIS    |     xx       |       xx        |
| ELLSIM    |     xx       |       xx        |
