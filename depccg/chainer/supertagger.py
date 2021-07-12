from typing import Union
import os
import json
import logging
import chainer

from depccg.chainer.lstm_parser_bi_fast import FastBiaffineLSTMParser
from depccg.chainer.ja_lstm_parser_bi import BiaffineJaLSTMParser


logger = logging.getLogger(__name__)


def load_chainer_tagger(
    model_path: str,
    device: int = -1
) -> Union[FastBiaffineLSTMParser, BiaffineJaLSTMParser]:

    model_file = os.path.join(model_path, 'tagger_model')
    def_file = os.path.join(model_path, 'tagger_defs.txt')

    assert os.path.exists(model_file) and os.path.exists(def_file), \
        (f'Failed in initialization. Directory "{model_path}" must contain both'
            '"tagger_model" and "tagger_defs.txt" files')

    with open(def_file) as f:
        tagger = eval(json.load(f)['model'])(model_path)

    logger.info(f'initializing supertagger with parameters at {model_file}')
    chainer.serializers.load_npz(model_file, tagger)

    if device >= 0:
        logger.info(f'sending the supertagger to gpu: {device}')
        tagger.to_gpu(device)

    return tagger
