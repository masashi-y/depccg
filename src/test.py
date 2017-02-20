
import sys
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
doc = [l.strip() for l in open(sys.argv[1])]

tagger = FastBiaffineLSTMParser(model)
chainer.serializers.load_npz(model + "/tagger_model", tagger)

parser = parser.PyAStarParser(model)

tag = tagger.predict_doc([s.split(" ") for s in doc])
# print [tagger.cats[i] for i in tag.argmax(1)]
res = parser.parse_doc(doc, tag)
for r in res:
    print r.deriv
