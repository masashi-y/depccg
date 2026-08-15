# Releasing depccg

Releases are published to PyPI by `.github/workflows/publish.yml` when a version
tag is pushed. The workflow tests the tagged revision, builds a source
distribution, and publishes with PyPI Trusted Publishing. It does not use a
stored API token.

## One-time setup

1. Create a GitHub environment named `pypi`. Adding required reviewers and
   restricting deployments to protected version tags is recommended.
2. In the PyPI `depccg` project settings, add a GitHub Actions Trusted Publisher
   with these values:

   - Owner: `masashi-y`
   - Repository: `depccg`
   - Workflow: `publish.yml`
   - Environment: `pypi`

## Publishing a release

1. Update the version in `pyproject.toml` and run `uv lock`.
2. Replace `Unreleased` in `CHANGELOG.md` with the release date.
3. Open and merge a release preparation pull request. Confirm that CI and the
   pretrained model smoke tests pass.
4. Tag the merge commit with the same version, prefixed by `v`, and push it:

   ```sh
   git tag -a v3.0.0 -m "depccg 3.0.0"
   git push origin v3.0.0
   ```

5. Approve the `pypi` environment deployment if protection rules require it.
6. Verify the new files and metadata on PyPI, then test installation in a fresh
   environment.

The workflow rejects a tag whose name does not exactly match the version in
`pyproject.toml`. Published PyPI files are immutable, so never reuse a released
version number. depccg has historically published source distributions because
it contains a native extension. Do not publish a wheel produced directly on a
GitHub-hosted runner; add a platform-wheel build using a compatible toolchain
first.
