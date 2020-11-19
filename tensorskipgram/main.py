import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorskipgram.trainer import train_epoch
from tensorskipgram.data.preprocessing import Preprocessor
from tensorskipgram.config import subj_data_fn, obj_data_fn, noun_space_fn
from tensorskipgram.data.dataset import create_noun_matrix, MatrixSkipgramDataset
from tensorskipgram.models.model import MatrixSkipgram
from tensorskipgram.data.main_preprocessing import main as preprocessing_main
from tensorskipgram.data.main_evaluation import main as evaluation_main
from tensorskipgram.data.main_training import main as training_main


def final_main():
    """Preprocess the corpus to get verb data."""
    preprocessing_main()
    """Now train matrices."""
    training_main()
    """Now evaluate the trained matrices."""
    evaluation_main()
