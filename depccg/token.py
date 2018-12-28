import os
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple, List

from depccg.download import MODEL_DIRECTORY
from depccg.morpha import MorphaStemmer
from depccg.printer import logger


class Token(NamedTuple):
    word: str
    lemma: str
    pos: str
    chunk: str
    entity: str

    @classmethod
    def from_piped(cls, string: str) -> 'Token':
        # WORD|POS|NER or WORD|LEMMA|POS|NER
        # or WORD|LEMMA|POS|NER|CHUCK
        items = string.split('|')
        if len(items) == 5:
            w, l, p, n, c = items
            return cls(w, l, p, c, n)
        elif len(items) == 4:
            w, l, p, n = items
            return cls(w, l, p, 'XX', n)
        else:
            w, p, n = items
            return cls(w, 'XX', p, 'XX', n)

    @classmethod
    def from_word(cls, word: str) -> 'Token':
        return cls(word, 'XX', 'XX', 'XX', 'XX')


candc_cmd = "cat \"{0}\" | {1}/bin/pos --model {1}/models/pos | {1}/bin/ner --model {1}/models/ner"


def try_annotate_using_candc(sentences: List[List[str]]) -> List[List[Token]]:
    candc_dir = os.environ.get('CANDC', None)
    fail = False
    if candc_dir:
        candc_dir = Path(candc_dir)
        if (candc_dir / 'bin' /'pos').exists() and \
                (candc_dir / 'bin' /'ner').exists() and \
                   (candc_dir / 'models' / 'pos').exists() and \
                (candc_dir / 'models' / 'ner').exists():
            pass
        else:
            logger.info('CANDC environmental variable may not be configured correctly.')
            logger.info('$CANDC/bin/{pos,ner} and $CANDC/models/{pos,ner} are expected to exist.')
            fail = True
    else:
        fail = True

    if fail:
        logger.info('Did not find C&C parser at CANDC environmental variable.')
        logger.info('Fill POS tag etc. using XX tag.')
        return [[Token.from_word(word) for word in sentence]
                for sentence in sentences]

    logger.info('Find C&C parser at CANDC environmental variable.')
    logger.info('Use C&C pipeline to annotate POS and NER infos.')

    stemmer = MorphaStemmer(str(MODEL_DIRECTORY / 'verbstem.list'))

    tmpfile = tempfile.mktemp()
    with open(tmpfile, 'w') as f:
        for sentence in sentences:
            print(' '.join(sentence), file=f)

    command = candc_cmd.format(tmpfile, candc_dir)
    proc = subprocess.Popen(command,
                            shell=True,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    res, error = proc.communicate()
    try:
        tagged_sentences = res.decode('utf-8').strip().split('\n')
        tagged_sentences = [[tuple(token.split('|')) for token in sentence.strip().split(' ')]
                            for sentence in tagged_sentences]
    except:
        raise RuntimeError('failed to process C&C output. there might have been some problem '
                           'during running C&C pipeline?\n'
                           f'stderr:\n {error}')

    res = []
    for sentence in tagged_sentences:
        words, poss = zip(*[(word, pos) for word, pos, _ in sentence])
        lemmas = stemmer.analyze(list(words), list(poss))
        tokens = [Token(word=word, pos=pos, entity=ner, lemma=lemma, chunk='XX')
                  for (word, pos, ner), lemma in zip(sentence, lemmas)]
        res.append(tokens)
    return res