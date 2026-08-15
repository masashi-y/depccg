# Changelog

This changelog covers depccg v3 and later releases.

## [3.0.0] - Unreleased

### Added

- PyTorch inference implementations for the English and Japanese pretrained
  parsers.
- PyTorch-only compatibility code for the English ELMo and Rebank checkpoints,
  allowing them to run without AllenNLP.
- Named model selection and downloads for English `basic`, `elmo`, and
  `rebank` models.
- Support for Python 3.10 and newer, including Python 3.14 and current NumPy
  releases.
- Modern project metadata in `pyproject.toml`, an uv lockfile, and a pytest
  development dependency group.
- Safe model archive extraction that rejects links and paths outside the target
  directory.

### Changed

- Replaced Chainer inference with PyTorch while preserving the pretrained model
  outputs. On the parity corpus, all supported models matched their v2
  counterparts on supertag and dependency-head argmaxes and on all 50 complete
  parse trees per model.
- Packaged the English, Japanese, and Rebank grammar data as static JSON
  resources instead of generating it from Jsonnet files at runtime.
- Pretrained models are downloaded to `~/.cache/depccg` by default. The location
  can be overridden with `DEPCCG_HOME`.
- Installation no longer requires manually installing Cython and NumPy before
  installing depccg.

### Removed

- Chainer and AllenNLP runtime dependencies and their integration modules.
- AllenNLP-based training and predictor integration.
- The English `elmo_rebank` model variant because its original model archive is
  no longer available. The English `elmo` and `rebank` variants remain
  supported.

### Compatibility notes

- Python versions older than 3.10 are no longer supported.
- The v2 CLI unintentionally disabled category-dictionary and seen-rule filters.
  To reproduce that effective behavior in v3, pass
  `--disable-category-dictionary --disable-seen-rules`.
