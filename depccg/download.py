from typing import Dict, Tuple, NamedTuple, Optional
import tarfile
import logging
from pathlib import Path
from collections import defaultdict

from depccg.chainer.supertagger import load_chainer_tagger
from depccg.allennlp.supertagger import load_allennlp_tagger


logger = logging.getLogger(__name__)

MODEL_DIRECTORY = Path(__file__).parent / 'models'


class ModelConfig(NamedTuple):
    framework: str
    name: str
    url: str
    config: Path
    semantic_templates: Path


SEMANTIC_TEMPLATES: Dict[str, Path] = {
    'en': MODEL_DIRECTORY / 'semantic_templates_en_event.yaml',
    'ja': MODEL_DIRECTORY / 'semantic_templates_ja_event.yaml'
}

MODELS: Dict[str, ModelConfig] = {
    'en': ModelConfig(
        'chainer',
        'tri_headfirst',
        '1mxl1HU99iEQcUYhWhvkowbE4WOH0UKxv',
        MODEL_DIRECTORY / 'config_en.jsonnet',
        SEMANTIC_TEMPLATES['en'],
    ),
    'en[elmo]': ModelConfig(
        'allennlp',
        'lstm_parser_elmo',
        '1UldQDigVq4VG2pJx9yf3krFjV0IYOwLr',
        MODEL_DIRECTORY / 'config_en.jsonnet',
        SEMANTIC_TEMPLATES['en'],
    ),
    'en[rebank]': ModelConfig(
        'allennlp',
        'lstm_parser_char_rebanking',
        '1Az840uCW8QuAkNCZq_Y8VOkW5j0Vtcj9',
        MODEL_DIRECTORY / 'config_rebank.jsonnet',
        SEMANTIC_TEMPLATES['en'],
    ),
    'en[elmo_rebank]': ModelConfig(
        'allennlp',
        'lstm_parser_elmo_rebanking',
        '1deyCjSgCuD16WkEhOL3IXEfQBfARh_ll',
        MODEL_DIRECTORY / 'config_rebank.jsonnet',
        SEMANTIC_TEMPLATES['en'],
    ),
    'ja': ModelConfig(
        'chainer',
        'ja_headfinal',
        '1bblQ6FYugXtgNNKnbCYgNfnQRkBATSY3',
        MODEL_DIRECTORY / 'config_ja.jsonnet',
        SEMANTIC_TEMPLATES['ja'],
    )
}


def _lang_and_variant(model: str):
    if '[' in model and ']' in model:
        assert model[-1] == ']'
        return model[:-1].split('[')
    else:
        return model, None


AVAILABLE_MODEL_VARIANTS = defaultdict(list)
for model in MODELS:
    lang, variant = _lang_and_variant(model)
    AVAILABLE_MODEL_VARIANTS[lang].append(variant)


def download(lang: str, variant: Optional[str]) -> None:
    config = MODELS[f'{lang}[{variant}]' if variant else lang]

    from google_drive_downloader import GoogleDriveDownloader as gdd
    logging.info(f'start downloading from {config.url}')
    filename = (MODEL_DIRECTORY / config.name).with_suffix('.tar.gz')
    gdd.download_file_from_google_drive(
        file_id=config.url,
        dest_path=filename,
        unzip=False,
        overwrite=True
    )

    if config.framework == 'chainer':
        logging.info('extracting files')
        tf = tarfile.open(filename)
        tf.extractall(MODEL_DIRECTORY)
    logging.info('finished')


def load_model_directory(model_name: str) -> Tuple[Path, ModelConfig]:
    config = MODELS[model_name]
    model_path = MODEL_DIRECTORY / config.name
    if config.framework == 'allennlp':
        model_path = model_path.with_suffix('.tar.gz')
    if not model_path.exists():
        lang, variant = _lang_and_variant(model_name)
        if variant is None:
            variant = ''
        raise RuntimeError(
            f'please download the model by doing \'depccg_{lang} download {variant}\'.')
    return model_path, config


def model_is_available(model_name: str) -> bool:
    return model_name in MODELS.keys()


def load_model(model_name: str, device: int = -1):
    model_path, config = load_model_directory(model_name)
    if config.framework == 'allennlp':
        supertagger = load_allennlp_tagger(model_path, device)
    elif config.framework == 'chainer':
        supertagger = load_chainer_tagger(model_path, device)
    else:
        raise KeyError(
            f'unsupported framework: {config.framework}'
        )
    return supertagger, config
