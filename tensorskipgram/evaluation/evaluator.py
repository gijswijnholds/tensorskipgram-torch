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


def evaluate_model_on_disambiguation_task(model: Callable[[str], Vector], task, strict: bool=False):
    """Calculate disambiguation accuracy (s1 should be more similar to s2 than to s3?)."""
    def compute_disambiguation(s1, s2, s3) -> float:
        vec1, vec2, vec3 = model(s1), model(s2), model(s3)
        cosim1, cosim2 = cosine_sim(vec1, vec2), cosine_sim(vec1, vec3)
        if cosim1 == cosim2:
            if strict:
                return 0.
            else:
                return 1.
        elif cosim1 > cosim2:
            return 1.
        else:
            return 0.
    preds = [compute_disambiguation(s1, s2, s3) for (s1, s2, s3) in task.data]
    return sum(preds)/float(len(task.data))
