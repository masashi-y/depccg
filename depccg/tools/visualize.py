
from .reader import read_trees_guess_extension
from ..printer import to_mathml
import logging
import argparse

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('PATH', help='path to conll data file')
    args = parser.parse_args()

    trees = [[tree] for _, _, tree in read_trees_guess_extension(args.PATH)]
    print(to_mathml(trees))
