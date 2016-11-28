
import cat
import utils
from collections import defaultdict

targets = utils.read_model_defs("full/targets.txt")

cats = defaultdict(int)
feat = defaultdict(int)

def traverse(cat):
    if cat.is_functor:
        traverse(cat.left)
        traverse(cat.right)
    else:
        feat[cat.feat] += 1

for c in targets:
    c = cat.parse(c.encode("utf-8"))
    cats[c.without_feat] += 1
    traverse(c)

print len(cats), len(feat)
print cats
print feat
