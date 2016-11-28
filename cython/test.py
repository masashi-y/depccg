# -*- coding: utf-8 -*-
import chainer
import numpy as np
from japanese_tagger import JaCCGEmbeddingTagger
tagger = JaCCGEmbeddingTagger("ja_model2")
chainer.serializers.load_npz("ja_model2/model_iter_15000", tagger)


sent = u"これ は テスト です ．".split(" ")
res = np.argmax(tagger.predict(sent), 1)
cats = tagger.cats
for w, c in zip(sent, [cats[i] for i in list(res)]):
    print "{}\t-->\t{}".format(w.encode("utf-8"), c)
