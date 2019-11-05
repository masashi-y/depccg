
import sys
import tarfile
from urllib.request import urlretrieve
import logging
import time
from pathlib import Path
from collections import defaultdict


logger = logging.getLogger(__name__)

MODEL_DIRECTORY = Path(__file__).parent / 'models'

MODELS = {
    'en': (
        'chainer',
        'tri_headfirst',
        '1mxl1HU99iEQcUYhWhvkowbE4WOH0UKxv',
        MODEL_DIRECTORY / 'config_en.json'
    ),
    'en[elmo]': (
        'allennlp',
        'lstm_parser_elmo',
        '1UldQDigVq4VG2pJx9yf3krFjV0IYOwLr',
        MODEL_DIRECTORY / 'config_en.json'
    ),
    'en[rebank]': (
        'allennlp',
        'lstm_parser_char_rebanking',
        '1Az840uCW8QuAkNCZq_Y8VOkW5j0Vtcj9',
        MODEL_DIRECTORY / 'config_rebank.json'
    ),
    'en[elmo_rebank]': (
        'allennlp',
        'lstm_parser_elmo_rebanking',
        '1deyCjSgCuD16WkEhOL3IXEfQBfARh_ll',
        MODEL_DIRECTORY / 'config_rebank.json'
    ),
    'ja': (
        'chainer',
        'ja_headfinal',
        '1bblQ6FYugXtgNNKnbCYgNfnQRkBATSY3',
        MODEL_DIRECTORY / 'config_ja.json'
    )
}


AVAILABLE_MODEL_VARIANTS = defaultdict(list)
for model in MODELS:
    if '[' in model and ']' in model:
        assert model[-1] == ']'
        lang, variant = model[:-1].split('[')
        AVAILABLE_MODEL_VARIANTS[lang].append(variant)


SEMANTIC_TEMPLATES = {
    'en': MODEL_DIRECTORY / 'semantic_templates_en_event.yaml',
    'ja': MODEL_DIRECTORY / 'semantic_templates_ja_event.yaml'
}


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download(lang, variant):
    model_name = f'{lang}[{variant}]' if variant else lang
    framework, basename, url, _ = MODELS[model_name]
    from google_drive_downloader import GoogleDriveDownloader as gdd
    logging.info(f'start downloading from {url}')
    filename = (MODEL_DIRECTORY / basename).with_suffix('.tar.gz')
    gdd.download_file_from_google_drive(file_id=url,
                                        dest_path=filename,
                                        unzip=False,
                                        overwrite=True)
    if framework == 'chainer':
        logging.info(f'extracting files')
        tf = tarfile.open(filename)
        tf.extractall(MODEL_DIRECTORY)
    logging.info(f'finished')


def load_model_directory(model_name):
    framework, basename, _, config = MODELS[model_name]
    model_path = MODEL_DIRECTORY / basename
    if framework == 'allennlp':
        model_path = model_path.with_suffix('.tar.gz')
    if not model_path.exists():
        lang, variant = model_name[:-1].split('[')
        raise RuntimeError(f'please download the model by doing \'depccg_{lang} download VARIANT\'.')
    return model_path, config


def model_is_available(model_name):
    return model_name in list(MODELS.keys())
