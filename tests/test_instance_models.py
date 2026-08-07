import io
import tarfile

from depccg import instance_models
from depccg.lang import set_global_language_to


def test_download_extracts_flat_allennlp_archive(monkeypatch, tmp_path):
    def fake_download(_file_id, destination):
        with tarfile.open(destination, "w:gz") as archive:
            for name, contents in {
                "config.json": b"{}",
                "weights.th": b"weights",
            }.items():
                info = tarfile.TarInfo(name)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))

    monkeypatch.setattr(instance_models, "MODEL_DIRECTORY", tmp_path)
    monkeypatch.setattr(instance_models, "_download_from_google_drive", fake_download)

    instance_models.download("en", "elmo")

    assert (tmp_path / "lstm_parser_elmo" / "weights.th").read_bytes() == b"weights"


def test_download_ignores_appledouble_archive_root(monkeypatch, tmp_path):
    def fake_download(_file_id, destination):
        with tarfile.open(destination, "w:gz") as archive:
            for name, contents in {
                "tri_headfirst/tagger_model.pt": b"model",
                "._tri_headfirst": b"metadata",
            }.items():
                info = tarfile.TarInfo(name)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))

    monkeypatch.setattr(instance_models, "MODEL_DIRECTORY", tmp_path)
    monkeypatch.setattr(instance_models, "_download_from_google_drive", fake_download)

    instance_models.download("en")

    assert (tmp_path / "tri_headfirst" / "tagger_model.pt").is_file()


def test_named_rebank_model_uses_rebank_grammar(monkeypatch, tmp_path):
    model_path = tmp_path / "lstm_parser_char_rebanking"
    model_path.mkdir()
    (model_path / "weights.th").touch()
    monkeypatch.setattr(instance_models, "MODEL_DIRECTORY", tmp_path)
    set_global_language_to("en")

    resolved_path, config = instance_models.load_model_directory("rebank")

    assert resolved_path == model_path
    assert config.config.name == "grammar_en_rebank.json"
