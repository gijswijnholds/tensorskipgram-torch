from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import numpy as np
from tensorskipgram.evaluation.spaces import Vector
from typing import Callable


def cosine_sim(emb1, emb2) -> float:
    """Cosine similarity between two embeddings."""
    return cosine_similarity([emb1, emb2])[0][1]


def evaluate_model_on_task(model: Callable[[str], Vector], task):
    """Calculate spearman rho between actual and predicted (cosine) similarity scores."""
    preds, trues = zip(*[(cosine_sim(model(s1), model(s2)), sc) for (s1, s2, sc) in task.data])
    return spearmanr(preds, trues)


def evaluate_model_on_task_late_fusion(model1: Callable[[str], Vector],
                                       model2: Callable[[str], Vector],
                                       task, alpha: float):
    """Evaluate a model a above but using late fusion."""
    """This means we compute two separate models, then fuse those together..."""
    def compute_late_fusion(s1, s2):
        cosim1 = cosine_sim(model1(s1), model1(s2))
        cosim2 = cosine_sim(model2(s1), model2(s2))
        return alpha * cosim1 + ((1 - alpha) * cosim2)
    preds, trues = zip(*[(compute_late_fusion(s1, s2), sc) for (s1, s2, sc) in task.data])
    return spearmanr(preds, trues)
