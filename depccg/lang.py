
from .combinator import (
    UnknownCombinator,
    HeadfirstCombinator,
    HeadfinalCombinator,
    EnForwardApplication,
    EnBackwardApplication,
    ForwardComposition,
    BackwardComposition,
    GeneralizedForwardComposition,
    GeneralizedBackwardComposition,
    Conjunction,
    Conjunction2,
    RemovePunctuation,
    RemovePunctuationLeft,
    CommaAndVerbPhraseToAdverb,
    ParentheticalDirectSpeech,
    Conjoin,
    JaForwardApplication,
    JaBackwardApplication,
    JaGeneralizedForwardComposition0,
    JaGeneralizedForwardComposition1,
    JaGeneralizedForwardComposition2,
    JaGeneralizedBackwardComposition0,
    JaGeneralizedBackwardComposition1,
    JaGeneralizedBackwardComposition2,
    JaGeneralizedBackwardComposition3
)
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


GLOBAL_LANG_NAME = 'en'


def set_global_language_to(lang: str) -> None:
    global GLOBAL_LANG_NAME
    logger.info('Setting the global language config to: %s', lang)
    GLOBAL_LANG_NAME = lang


en_default_binary_rules = [
    HeadfirstCombinator(EnForwardApplication()),
    HeadfirstCombinator(EnBackwardApplication()),
    HeadfirstCombinator(ForwardComposition('/', '/', '/')),
    HeadfirstCombinator(BackwardComposition('/', '\\', '/')),
    HeadfirstCombinator(GeneralizedForwardComposition('/', '/', '/')),
    HeadfirstCombinator(GeneralizedBackwardComposition('/', '/', '/')),
    HeadfirstCombinator(Conjunction()),
    HeadfirstCombinator(Conjunction2()),
    HeadfirstCombinator(RemovePunctuation(False)),
    HeadfirstCombinator(RemovePunctuation(True)),
    HeadfirstCombinator(RemovePunctuationLeft()),
    HeadfirstCombinator(CommaAndVerbPhraseToAdverb()),
    HeadfirstCombinator(ParentheticalDirectSpeech())
]


ja_default_binary_rules = [
    HeadfinalCombinator(Conjoin()),
    HeadfinalCombinator(JaForwardApplication()),
    HeadfinalCombinator(JaBackwardApplication()),
    HeadfinalCombinator(JaGeneralizedForwardComposition0('/', '/', '/', '>B')),
    HeadfinalCombinator(
        JaGeneralizedBackwardComposition0('\\', '\\', '\\', '<B1')),
    HeadfinalCombinator(
        JaGeneralizedBackwardComposition1('\\', '\\', '\\', '<B2')),
    HeadfinalCombinator(
        JaGeneralizedBackwardComposition2('\\', '\\', '\\', '<B3')),
    HeadfinalCombinator(
        JaGeneralizedBackwardComposition3('\\', '\\', '\\', '<B4')),
    HeadfinalCombinator(
        JaGeneralizedForwardComposition0('/', '\\', '\\', '>Bx1')),
    HeadfinalCombinator(
        JaGeneralizedForwardComposition1('/', '\\', '\\', '>Bx2')),
    HeadfinalCombinator(
        JaGeneralizedForwardComposition2('/', '\\', '\\', '>Bx3')),
]


UNK_COMBINATOR = UnknownCombinator()
BINARY_RULES = defaultdict(
    lambda: [UNK_COMBINATOR],
    {
        'en': en_default_binary_rules,
        'ja': ja_default_binary_rules
    }
)
