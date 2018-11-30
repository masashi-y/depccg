from typing import Dict
import numpy as np


def read_pretrained_embeddings(filepath: str) -> np.ndarray:
    nvocab = 0
    io = open(filepath)
    dim = len(io.readline().split())
    io.seek(0)
    for _ in io:
        nvocab += 1
    io.seek(0)
    res = np.empty((nvocab, dim), dtype=np.float32)
    for i, line in enumerate(io):
        line = line.strip()
        if len(line) == 0: continue
        res[i] = line.split()
    io.close()
    return res


def read_model_defs(filepath: str) -> Dict[str, int]:
    res = {}
    for i, line in enumerate(open(filepath)):
        word, _ = line.strip().split(' ')
        res[word] = i
    return res


