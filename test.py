
from tagger import EmbeddingTagger
from astar import AStarParser
import chainer

tagger = EmbeddingTagger("data/train",
        "data/embeddings/embeddings-scaled.EMBEDDING_SIZE=50.vectors",
        20, 30)

chainer.serializers.load_npz("data/train/model_iter_10000", tagger)
parser = AStarParser(tagger)
parser.test()
