
import sys
import tarfile
import urllib
import logging
import time
from pathlib import Path


logger = logging.getLogger(__name__)

MODELS = {
    'en': ('tri_headfirst', 'http://cl.naist.jp/~masashi-y/resources/depccg/en_hf_tri.tar.gz'),
    'ja': ('ja_headfinal', 'http://cl.naist.jp/~masashi-y/resources/depccg/ja_hf_ccgbank.tar.gz')
}

MODEL_DIRECTORY = Path(__file__).parent / 'models'


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


def download(model_name):
    basename, url = MODELS[model_name]
    logging.info(f'start downloading from {url}')
    filename = (MODEL_DIRECTORY / basename).with_suffix('.tar.gz')
    urllib.request.urlretrieve(url, filename, reporthook)
    logging.info(f'extracting files')
    tf = tarfile.open(filename)
    tf.extractall(MODEL_DIRECTORY)
    logging.info(f'finished')


def load_model_directory(model_name):
    basename, url = MODELS[model_name]
    model_dir = MODEL_DIRECTORY / basename
    if not model_dir.exists():
        raise RuntimeError(f'please download the model by doing \'depccg_{model_name} download\'.')
    return model_dir
