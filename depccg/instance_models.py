from __future__ import annotations

import logging
import os
import tarfile
from collections import defaultdict
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from depccg.lang import get_global_language
from depccg.torch_supertagger import MODEL_FILENAME, load_torch_tagger
from depccg.types import ModelConfig

logger = logging.getLogger(__name__)
RESOURCE_DIRECTORY = Path(__file__).parent / "models"
MODEL_DIRECTORY = Path(
    os.environ.get("DEPCCG_HOME", Path.home() / ".cache" / "depccg")
).expanduser()

SEMANTIC_TEMPLATES: dict[str, Path] = {
    "en": RESOURCE_DIRECTORY / "semantic_templates_en_event.yaml",
    "ja": RESOURCE_DIRECTORY / "semantic_templates_ja_event.yaml",
}

MODELS: dict[str, ModelConfig] = {
    "en": ModelConfig(
        "tri_headfirst",
        "19ksMKnW6ExoRzn88HkbBH-Yy41FwUomu",
        RESOURCE_DIRECTORY / "grammar_en.json",
        SEMANTIC_TEMPLATES["en"],
    ),
    "ja": ModelConfig(
        "ja_headfinal",
        "1KjG9iSUGAZvR13vJls5nZ_NQRG5_dxuh",
        RESOURCE_DIRECTORY / "grammar_ja.json",
        SEMANTIC_TEMPLATES["ja"],
    ),
}

AVAILABLE_MODEL_VARIANTS = defaultdict(list)
AVAILABLE_MODEL_VARIANTS.update({"en": [None], "ja": [None]})


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        if member.issym() or member.islnk():
            raise RuntimeError(f"links are not allowed in model archive: {member.name}")
        target = (destination / member.name).resolve()
        if destination not in target.parents and target != destination:
            raise RuntimeError(f"unsafe path in model archive: {member.name}")
    archive.extractall(destination)


def _download_from_google_drive(file_id: str, destination: Path) -> None:
    url = "https://drive.usercontent.google.com/download"
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        with session.get(
            url,
            params={"id": file_id, "export": "download", "confirm": "t"},
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)


def download(lang: str, variant: str | None = None) -> None:
    if variant is not None:
        raise ValueError("model variants are no longer supported")
    config = MODELS[lang]
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    archive_path = MODEL_DIRECTORY / f"{config.name}.tar.gz"
    logger.info("downloading %s model", lang)
    _download_from_google_drive(config.url, archive_path)
    try:
        with tarfile.open(archive_path) as archive:
            _safe_extract(archive, MODEL_DIRECTORY)
    finally:
        archive_path.unlink(missing_ok=True)
    logger.info("downloaded %s PyTorch model", lang)


def load_model_directory(model: str | None) -> tuple[Path, ModelConfig]:
    lang = get_global_language()
    config = MODELS[lang]
    if model is None:
        model_path = MODEL_DIRECTORY / config.name
    else:
        model_path = Path(model).expanduser()
    if not (model_path / MODEL_FILENAME).exists():
        raise RuntimeError(f"model is not available; run 'depccg_{lang} download'")
    return model_path, config


def model_is_available(model_name: str) -> bool:
    return model_name in MODELS


def load_model(model: str | None, device: int = -1):
    model_path, config = load_model_directory(model)
    return load_torch_tagger(model_path, device), config
