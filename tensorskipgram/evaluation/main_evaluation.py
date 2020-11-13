"""Evaluate some models on evaluation tasks."""
from tensorskipgram.tasks.datasets \
    import create_ml2008, create_ml2010, create_gs2011, create_ks2013
from tensorskipgram.tasks.datasets \
    import create_ks2014, create_elldis, create_ellsim


def evaluate_all_models() -> None:
    """Load all tasks, and models, and compute spearman correlations."""
    pass


def main() -> None:
    pass
