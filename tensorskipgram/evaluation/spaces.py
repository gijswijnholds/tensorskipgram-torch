from typing import Dict
from tqdm import tqdm
import numpy as np

Vector = np.ndarray
Matrix = np.ndarray


def read_vectors(space_fn: str) -> Dict:
    spaceFile = open(space_fn, 'r')
    space = {}
    print("Loading vectors...")
    for ln in tqdm(spaceFile.readlines()):
        ln = ln.strip().split()
        key = ln[0]
        vec = np.array([float(b) for b in ln[1:]])
        space[key] = vec
    return space


def read_matrices(space_fn: str) -> Dict:
    spaceFile = open(space_fn, 'r')
    space = {}
    print("Loading matrices...")
    for ln in tqdm(spaceFile.readlines()):
        ln = ln.strip().split()
        key = ln[0]
        vec = np.array([float(b) for b in ln[1:]])
        embed_size = int(np.sqrt(len(vec)))
        mat = vec.reshape(embed_size, embed_size)
        space[key] = mat
    return space


class VectorSpace(object):
    def __init__(self, name: str, path: str):
        self.name = name
        self.embedding = read_vectors(path)

    def embed(self, word: str):
        return self.embedding(word)


class MatrixSpace(object):
    def __init__(self, name: str, path: str):
        self.name = name
        self.embedding = read_matrices(path)

    def embed(self, word):
        return self.embedding(word)
