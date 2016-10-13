# -*- coding: utf-8 -*-

from tagger import EmbeddingTagger
from astar import AStarParser
import chainer

# print "loading tagger"
# tagger = EmbeddingTagger("data/train")
#
# print "loading tagger model"
# chainer.serializers.load_npz("data/train/model_iter_80000", tagger)

print "loading parser"
parser = AStarParser("model")

print "parsing"
input_text = "this is a new sentence ."
res = parser.parse(input_text)
print res

parse, cost = res
print cost
parse.show_derivation()
print parse

