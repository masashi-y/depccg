
from .tree import Tree


def read_auto(filename, lang='en'):
    for line in open(filename):
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith("ID"):
            name = line
        else:
            tree = Tree.of_auto(line, lang)
            yield (name, tree)

