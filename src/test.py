
import parser
import chainer
from py.ja_lstm_parser import JaLSTMParser
from py.lstm_parser import LSTMParser
from py.tagger import EmbeddingTagger
from py.japanese_tagger import JaCCGEmbeddingTagger
from py.ja_lstm_tagger import JaLSTMTagger
from py.lstm_tagger import LSTMTagger
from py.lstm_tagger_ph import PeepHoleLSTMTagger
from py.ja_lstm_parser_ph import PeepHoleJaLSTMParser
from py.lstm_parser_bi import BiaffineLSTMParser
from py.precomputed_parser import PrecomputedParser
from py.lstm_parser_bi_fast import FastBiaffineLSTMParser
from py.ja_lstm_parser_bi import BiaffineJaLSTMParser

model = "/home/masashi-y/myccg/models/tri_headfirst"
sent = "this is test sentence ."

tagger = FastBiaffineLSTMParser(model)
chainer.serializers.load_npz(model + "/tagger_model", tagger)

parser = parser.PyAStarParser(model)

_, _, tag = tagger.predict_doc([sent.split(" "), sent])[0]
# print [tagger.cats[i] for i in tag.argmax(1)]
tree = parser.parse(sent, tag)
print tree.deriv
print tree.auto
