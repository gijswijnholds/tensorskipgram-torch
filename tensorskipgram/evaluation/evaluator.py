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
    preds, trues = zip(*[cosine_sim(model(s1), model(s2)) for (s1, s2, sc) in task])
    return spearmanr(preds, trues)
