
import logging

logger = logging.getLogger(__name__)


GLOBAL_LANG_NAME = 'en'


def set_global_language_to(lang: str) -> None:
    global GLOBAL_LANG_NAME
    logger.info('Setting the global language config to: %s', lang)
    GLOBAL_LANG_NAME = lang


def get_global_language() -> str:
    return GLOBAL_LANG_NAME
