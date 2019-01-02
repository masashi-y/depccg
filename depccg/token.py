import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

from depccg.download import MODEL_DIRECTORY
from depccg.morpha import MorphaStemmer
from depccg.printer import logger


class Token(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, item):
        return self[item]

    def __repr__(self):
        res = super().__repr__()
        return f'Token({res})'

    @classmethod
    def from_piped(cls, string: str) -> 'Token':
        # WORD|POS|NER or WORD|LEMMA|POS|NER
        # or WORD|LEMMA|POS|NER|CHUCK
        items = string.split('|')
        if len(items) == 5:
            word, lemma, pos, entity, chunk = items
        elif len(items) == 4:
            word, lemma, pos, entity = items
            chunk = 'XX'
        else:
            assert len(items) == 3
            word, pos, entity = items
            lemma = 'XX'
            chunk = 'XX'

        return Token(word=word,
                     lemma=lemma,
                     pos=pos,
                     entity=entity,
                     chunk=chunk)

    @classmethod
    def from_word(cls, word: str) -> 'Token':
        return Token(word=word,
                     lemma='XX',
                     pos='XX',
                     entity='XX',
                     chunk='XX')


candc_cmd = "cat \"{0}\" | {1}/bin/pos --model {1}/models/pos | {1}/bin/ner --model {1}/models/ner"


def annotate_XX(sentences: List[List[str]], tokenize=False) -> List[List[Token]]:
    if tokenize:
        raise NotImplementedError('no tokenizer implemented')

    return [[Token.from_word(word) for word in sentence]
            for sentence in sentences]


def try_annotate_using_candc(sentences: List[List[str]], tokenize=False) -> List[List[Token]]:
    if tokenize:
        raise NotImplementedError('no tokenizer implemented in the C&C pipeline')

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
        logger.info('did not find C&C parser at CANDC environmental variable.')
        logger.info('fill POS tag etc. using XX tag.')
        return annotate_XX(sentences)

    logger.info('find C&C parser at CANDC environmental variable.')
    logger.info('use C&C pipeline to annotate POS and NER infos.')

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


def annotate_using_spacy(sentences, tokenize=False, n_threads=2, batch_size=10000):
    import spacy
    nlp = spacy.load('en', disable=['parser'])
    logger.info('use spacy to annotate POS and NER infos.')

    if tokenize:
        docs = [nlp.tokenizer(' '.join(sentence)) for sentence in sentences]
        raw_sentences = [[str(token) for token in doc] for doc in docs]
    else:
        docs = [nlp.tokenizer.tokens_from_list(sentence) for sentence in sentences]
    for name, proc in nlp.pipeline:
        docs = proc.pipe(docs,
                         n_threads=n_threads,
                         batch_size=batch_size)

    res = []
    for sentence in docs:
        tokens = []
        for token in sentence:
            if token.ent_iob_ == 'O':
                ner = token.ent_iob_
            else:
                ner = token.ent_iob_ + '-' + token.ent_type_

            tokens.append(
                Token(word=str(token),
                      pos=token.tag_,
                      entity=ner,
                      lemma=token.lemma_,
                      chunk='XX'))
        res.append(tokens)
    if tokenize:
        return res, raw_sentences
    else:
        return res


def annotate_using_janome(sentences, tokenize=False):
    assert tokenize, 'no support for using janome with pre-tokenized inputs'
    from janome.tokenizer import Tokenizer
    logger.info('use Janome to tokenize and annotate POS infos.')
    tokenizer = Tokenizer()
    res = []
    raw_sentences = []
    for sentence in sentences:
        sentence = ''.join(sentence)
        tokenized = tokenizer.tokenize(sentence)
        tokens = []
        for token in tokenized:
            pos, pos1, pos2, pos3 = token.part_of_speech.split(',')
            token = Token(surf=token.surface,
                          pos=pos,
                          pos1=pos1,
                          pos2=pos2,
                          pos3=pos3,
                          inflectionForm=token.infl_form,
                          inflectionType=token.infl_type,
                          reading=token.reading,
                          base=token.base_form)
            tokens.append(token)
        raw_sentence = [token.surface for token in tokenized]
        res.append(tokens)
        raw_sentences.append(raw_sentence)
    return res, raw_sentences


english_annotator = {
    'candc': try_annotate_using_candc,
    'spacy': annotate_using_spacy,
}


japanese_annotator = {
    'janome': annotate_using_janome
}
